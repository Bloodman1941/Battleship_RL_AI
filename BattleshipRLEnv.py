#Battleship Reinforcement Leanring Eviornment and Trainer.
#The code supports training the AI against a Strandard Population Distribution AI
# and an AI that fires randomly. Having the choice of whether or not to display visuals.

from pathlib import Path
import random
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import ai as spd_ai
from game import (
    BattleshipGame,
    HIT_EMPTY_CODE,
    HIT_UNSANK_SHIP_CODE,
    UNHIT_CODE,
)

BOARD_SIZE = 10
DEFAULT_SHIP_LENGTHS = [5, 4, 3, 3, 2]


class BattleshipRLEnv:
    def __init__(self, board_size: int = BOARD_SIZE, seed: Optional[int] = None, opponent_type: str = "spd"):
        #Initializes the RL Enviornment
        #The board size is a 10x10
        #The seed (where the ships are placed) is randomly produced
        #You are able to select either an "spd" or "random" opponent 
        self.board_size = board_size
        self.rng = random.SystemRandom() if seed is None else random.Random(seed)
        self.game = BattleshipGame(size=board_size)
        self.done = False
        self.winner = -1
        self.prev_ship_count = len(DEFAULT_SHIP_LENGTHS)
        self.last_was_hit = False
        self.opponent_type = opponent_type
        self.reset()

    def _all_sunk(self, player_id: int) -> bool:
        #Checks if all game pieces have been sunk for either player.
        pieces = self.game.player_1_pieces if player_id == 1 else self.game.player_2_pieces
        return all(ship["is_sank"] for ship in pieces.values())

    def reset(self):
        #Resets the game envirnment for a new episode.
        self.game.reset_boards(
            seed1=self.rng.randint(0, 1_000_000),
            seed2=self.rng.randint(0, 1_000_000),
        )
        self.done = False
        self.winner = -1
        self.last_was_hit = False
        self.prev_ship_count = self.game.get_remaining_ship_count(2)
        return self.observe()

    def can_place(self, board, x, y, length, horizontal):
        #Checks if a ship of a given length can be placed at the specified coordinates.
        for i in range(length):
            nx, ny = (x + i, y) if horizontal else (x, y + i)
            if nx >= 10 or ny >= 10 or board[ny][nx].ship_id !=0:
                return False
        return True

    def observe(self):
        #Checks the current player's view  of the opponent's board
        board = self.game.get_view_board_opponent(2)
        return np.array(board, dtype=np.float32)

    def get_tensor_observation(self):
        #Converts the current board observation into a multi-channel tensor for the neural network.
        #Those channels are Hits, Misses, Unknown, and Fully Sunk Ships
        obs = self.observe()

        ship_hit = (obs == HIT_UNSANK_SHIP_CODE).astype(np.float32)
        miss = (obs == HIT_EMPTY_CODE).astype(np.float32)
        unknown = (obs == UNHIT_CODE).astype(np.float32)

        sunk_ships = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        for ship in self.game.player_2_pieces.values():
            if ship["is_sank"]:
                for tile in ship["tiles"]:
                    ty = tile.y_position
                    tx = tile.x_position
                    sunk_ships[ty][tx] = 1.0

        stacked = np.stack([ship_hit, miss, unknown, sunk_ships], axis=-1)
        return torch.tensor(stacked, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    def _adjacent_miss_penalty(self, x: int, y: int) -> float:
        #Calculates a penalty for firing near known misses
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        misses = 0
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                tile = self.game.player_2_board[ny][nx]
                if tile.is_hit and tile.ship_id == 0:
                    misses += 1
        return -0.1 * (misses / 4.0)

    def _target_bonus(self, x: int, y: int, board: np.ndarray) -> float:
        #Rewards firing at adjacent tiles to known, unsunk ships.
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if board[ny][nx] == HIT_UNSANK_SHIP_CODE:
                    return 1.5
        return 0.0

    def _hunt_penalty(self, x: int, y: int, board: np.ndarray) -> float:
        #Penalizes firing at tiles that aren't near unsunk ships.
        has_unsunk = np.any(board == HIT_UNSANK_SHIP_CODE)
        if not has_unsunk:
            return 0.0
        return -0.5 if self._target_bonus(x, y, board) == 0 else 1.0

    def _ship_fit_bonus(self, x: int, y: int) -> float:
        #Rewards firing at tiles where remaning enemy ships could fit.
        remaining = self.game.get_remaining_ship_lengths(1)
        if not remaining:
            return 0.0

        board = self.game.get_view_board_opponent(2)
        max_bonus = 0.0

        for length in remaining:
            h_fit = True
            for dx in range(length):
                nx = x + dx
                if nx >= self.board_size or board[y][nx] != UNHIT_CODE:
                    h_fit = False
                    break
            if h_fit:
                max_bonus = max(max_bonus, 0.15)

            v_fit = True
            for dy in range(length):
                ny = y + dy
                if ny >= self.board_size or board[ny][x] != UNHIT_CODE:
                    v_fit = False
                    break
            if v_fit:
                max_bonus = max(max_bonus, 0.15)

        return max_bonus

    def step(self, action_idx: int):
        #Executes a step in the game and gives the rewards and penalites for performing various actions.
        if self.done:
            return self.observe(), 0.0, True, {"winner": self.winner}

        x, y = action_idx % self.board_size, action_idx // self.board_size
        board = self.observe()

        #Invalid move penalty
        if self.game.player_2_board[y][x].is_hit:
            return self.observe(), -2.0, False, {"invalid": True}

        #Calculates rewards
        s_bonus = self._ship_fit_bonus(x, y)
        t_bonus = self._target_bonus(x, y, board) * 1.5
        h_penalty = self._hunt_penalty(x, y, board)
        reward = -0.1 + t_bonus + h_penalty + s_bonus

        #Survival/progress rewards
        rl_ships_left = self.game.get_remaining_ship_count(1)
        spd_ships_left = self.game.get_remaining_ship_count(2)
        if rl_ships_left < spd_ships_left:
            reward -= 0.1 * (spd_ships_left - rl_ships_left)

        c_bonus = 1.0 if self.last_was_hit else 0.0
        hit = self.game.fire_at_tile(opponent_id=2, x_position=x, y_position=y)
        self.last_was_hit = hit

        #Resolve hit/miss outcomes
        if hit:
            reward += 2.0 + c_bonus
            current_ships = self.game.get_remaining_ship_count(2)
            if current_ships < self.prev_ship_count:
                reward += 10.0 #Reward for sinking ships
            self.prev_ship_count = current_ships

            if self.game.is_game_over():
                self.done, self.winner = True, 1
                reward += 50 #Reward for winning
        else:
            reward -= 0.5

        #Opponent's turn
        if not self.done:
            opp_view = self.game.get_view_board_opponent(1)
            opp_lengths = self.game.get_remaining_ship_lengths(1)

            if self.opponent_type == "random":
                _, rx, ry = spd_ai.RandomAI.find_best_move(opp_view, opp_lengths)
            else:
                _, rx, ry = spd_ai.find_best_move(opp_view, opp_lengths)

            if rx != -1:
                self.game.fire_at_tile(opponent_id=1, x_position=rx, y_position=ry)
                if self.game.is_game_over():
                    self.done, self.winner = True, 2
                    reward -= 30.0 #Penalty for losing

        return self.observe(), reward, self.done, {"hit": hit, "winner": self.winner}


class PolicyNet(nn.Module):
    #Convolutional Neural Network acting as the Actor (Policy) in the RL setup.
    #Outputs action logits for each tile on the board.
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )

    def forward(self, x):
        logits = self.features(x)
        return logits.view(-1, BOARD_SIZE * BOARD_SIZE)


class ValueNet(nn.Module):
    #Convolutional Neural Network acting as the Critic (Value) in the RL setup.
    #Outputs a scalar baseline value evaluating the current board state.
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.features(x)


@dataclass
class RLConfig:
    #Configures the parameters for the RL Training process
    episodes: int = 10000
    checkpoint_interval: int = 1000
    gamma: float = 0.99
    lr: float = 3e-4
    entropy_coef: float = 0.05
    value_coef: float = 0.5
    max_steps: int = 300
    seed: int = 42
    save_path: str = "output/rl_battleship.pt"
    visualize: bool = False

class RLTrainer:
    #Handles the training loop, memory management and network updates
    #for the RL agent using an Actor-Critic architecture
    def __init__(self, config: RLConfig, resume_from: str = None):
        #Initializes the trainer.
        #config: Training configuration parameters.
        #resume_from: is the optional path saved checkpoint file to resume training.
        self.config = config
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.env = BattleshipRLEnv(seed=config.seed)
        self.policy = PolicyNet()
        self.value_net = ValueNet()
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=config.lr)
        self.value_opt = optim.Adam(self.value_net.parameters(), lr=config.lr)

        self.win_count = 0
        self.total_episodes = 0
        self.total_turns = 0
        self.total_misses = 0
        self.total_reward = 0.0
        self.start_episode = 0

        #Load checkpoint if provided
        if resume_from and Path(resume_from).exists():
            print(f"Loading checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from, map_location="cpu")
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.value_net.load_state_dict(checkpoint["value_state_dict"])

            stats = checkpoint.get("stats", {})
            self.start_episode = checkpoint.get("episode", 0)
            self.total_episodes = stats.get("total_episodes", 0)
            self.win_count = stats.get("win_count", 0)
            self.total_turns = stats.get("total_turns", 0)
            self.total_misses = stats.get("total_misses", 0)
            self.total_reward = stats.get("total_reward", 0.0)
            print(f"Resumed at episode {self.start_episode}")
        else:
            print("Starting fresh.")

    def _masked_dist(self, logits: torch.Tensor, obs: np.ndarray):
        mask = torch.tensor((obs.reshape(-1) == UNHIT_CODE).astype(np.float32), dtype=torch.float32).unsqueeze(0)
        logits = logits.masked_fill(mask == 0, 0.2)
        return Categorical(logits=logits)

    def save_checkpoint(self, global_episode: int, stats: dict, mode: str, is_final: bool = False):
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        #Saves model weights and training stats to disk.
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value_net.state_dict(),
            "stats": stats,
            "episode": global_episode,
            "total_episodes": stats.get("total_episodes", 0),
            "config": self.config.__dict__,
            "timestamp": datetime.now().isoformat(),
        }
        torch.save(checkpoint, output_dir / "rl_battleship.pt")

        run_start = self.start_episode + 1
        if is_final:
            interval_start = run_start
        else:
            interval_start = get_interval_start(global_episode, self.config.checkpoint_interval, run_start)

        save_summary_log(mode, interval_start, global_episode, stats, is_final=is_final)
        print(f"[{mode.upper()}] Checkpoint saved at global episode {global_episode}")

    def train(self):
        #Executes the main RL training loop.
        mode_label = f"train_{self.env.opponent_type}"
        session_end_episode = self.start_episode + self.config.episodes
        print(f"Starting {mode_label} for {self.config.episodes} episodes from global {self.start_episode}...")
    
        for local_idx in range(self.config.episodes):
            global_episode = self.start_episode + local_idx + 1
            print(f"Episode {global_episode}/{self.start_episode + self.config.episodes}")
        
            obs = self.env.reset()
            done = False
            logprobs, values, rewards, entropies = [], [], [], []
            turns = 0

            #Run episode
            while not done and turns < self.config.max_steps:
                x = self.env.get_tensor_observation()
                logits = self.policy(x)
                dist = self._masked_dist(logits, obs)

                action = dist.sample()
                logprob = dist.log_prob(action)
                value = self.value_net(x).squeeze(-1)

                next_obs, reward, done, info = self.env.step(action.item())

                logprobs.append(logprob)
                values.append(value)
                rewards.append(reward)
                entropies.append(dist.entropy())

                obs = next_obs
                turns += 1

            #Tally stats
            self.total_episodes = global_episode
            if self.env.winner == 1: 
                self.win_count += 1
            self.total_turns += turns
            self.total_misses += turns - sum(1 for r in rewards if r > 0) 
            self.total_reward += sum(rewards)

            self._update_networks(rewards, logprobs, values, entropies)

            #Handle checkpointing
            if (local_idx + 1) % self.config.checkpoint_interval == 0:
                stats = {
                    "total_episodes": self.total_episodes,
                    "win_count": self.win_count,
                    "total_turns": self.total_turns,
                    "total_misses": self.total_misses,
                    "total_reward": self.total_reward,
                }
                self.save_checkpoint(global_episode, stats, mode_label)

        #Final save
        final_stats = {
            "total_episodes": self.total_episodes,
            "win_count": self.win_count,
            "total_turns": self.total_turns,
            "total_misses": self.total_misses,
            "total_reward": self.total_reward,
        }
        final_episode = self.start_episode + self.config.episodes
        self.save_checkpoint(final_episode, final_stats, f"{mode_label}_final", is_final=True)
    
        print("Training Complete.")

    def _update_networks(self, rewards, logprobs, values, entropies):
        returns = []
        g = 0
        for r in reversed(rewards):
            g = r + self.config.gamma * g
            returns.insert(0, g)

        returns = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.cat(values).squeeze(-1)
        adv = returns - values_t.detach()

        #Calculate losses
        policy_loss = -(torch.cat(logprobs) * adv).mean() - self.config.entropy_coef * torch.cat(entropies).mean()
        value_loss = F.mse_loss(values_t, returns)
        
        #Optimize networks
        self.policy_opt.zero_grad()
        self.value_opt.zero_grad()
        (policy_loss + self.config.value_coef * value_loss).backward()
        self.policy_opt.step()
        self.value_opt.step()

        return self.policy, self.value_net

def watch_training_from_start(num_episodes=30, delay=0.005, resume_from="output/rl_battleship.pt"):
    #Opens a GUI to visualize the agent playing Battleship.
    import time
    import tkinter as tk
    from PIL import Image, ImageTk
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from torch.distributions import Categorical

    print("Battleship RL Training Visualization")

    policy_net = PolicyNet()
    value_net = ValueNet()
    policy_opt = optim.Adam(policy_net.parameters(), lr=3e-4)
    value_opt = optim.Adam(value_net.parameters(), lr=3e-4)

    stats = {
        "total_episodes": 0,
        "win_count": 0,
        "total_turns": 0,
        "total_misses": 0,
        "total_reward": 0.0,
    }
    checkpoint_win_count = stats.get("win_count", 0)
    start_episode = stats.get("total_episodes", 0)
    save_freq = 10

    if Path(resume_from).exists():
        print(f"Loading trained model from {resume_from}")
        checkpoint = torch.load(resume_from, map_location="cpu")
        policy_net.load_state_dict(checkpoint["policy_state_dict"])
        value_net.load_state_dict(checkpoint["value_state_dict"])
        stats = checkpoint.get("stats", stats)
        start_episode = checkpoint.get("episode", 0)
        checkpoint_win_count = stats.get("win_count", 0)
        print(f"Resuming watch from episode {start_episode}")
    else:
        print("No saved model found. Starting from episode 0.")

    end_episode = start_episode + num_episodes
    print(f"Watch mode: Episodes {start_episode} to {end_episode}")

    def masked_dist(logits, obs):
        mask = torch.tensor((obs.reshape(-1) == UNHIT_CODE).astype(np.float32), dtype=torch.float32).unsqueeze(0)
        logits = logits.masked_fill(mask == 0, -1e9)
        return Categorical(logits=logits)

    config = type("Config", (), {
        "gamma": 0.99,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "max_steps": 300,
    })()

    root = tk.Tk()
    root.title("Battleship RL Training Visualization")
    root.geometry("1200x900")

    ref_size = 28

    def load_img(path):
        #Helper to load and resize images for the grid.
        return ImageTk.PhotoImage(Image.open(path).convert("RGB").resize((ref_size, ref_size)))

    image_dict = {
        "water": load_img("./images/water.jpeg"),
        "water_hit": load_img("./images/water_hit.jpeg"),
        "ship": load_img("./images/ship.jpeg"),
        "ship_hit": load_img("./images/ship_hit.jpeg"),
    }

    #GUI Layout framing
    left_frame = tk.Frame(root)
    left_frame.grid(row=0, column=0, sticky="n")

    right_frame = tk.Frame(root)
    right_frame.grid(row=0, column=1, sticky="n")

    tk.Label(left_frame, text="RL Board (Your Ships)", font=("Arial", 14, "bold")).grid(row=0, column=0, pady=5)
    p1_frame = tk.Frame(left_frame)
    p1_frame.grid(row=1, column=0, padx=10, pady=5)

    p1_labels = [[tk.Label(p1_frame, image=image_dict["water"]) for _ in range(10)] for _ in range(10)]
    for row in range(10):
        for col in range(10):
            p1_labels[row][col].grid(row=row, column=col, padx=1, pady=1)

    tk.Label(left_frame, text="SPD Board (Enemy - RL View)", font=("Arial", 14, "bold")).grid(row=2, column=0, pady=5)
    p2_frame = tk.Frame(left_frame)
    p2_frame.grid(row=3, column=0, padx=10, pady=5)

    p2_buttons = [[tk.Button(p2_frame, image=image_dict["water"], width=ref_size, height=ref_size) for _ in range(10)] for _ in range(10)]
    for row in range(10):
        for col in range(10):
            p2_buttons[row][col].grid(row=row, column=col, padx=1, pady=1)

    tk.Label(right_frame, text="SPD View Attacking RL", font=("Arial", 14, "bold")).pack(pady=5)

    #Matplotlib graph setup for the heatmap
    graph_frame = tk.Frame(right_frame, relief=tk.SOLID, borderwidth=1)
    graph_frame.pack(padx=10, pady=5, fill="both", expand=True)

    graph_figure = Figure(figsize=(6, 6), dpi=100)
    subplot = graph_figure.add_subplot(111)
    graph_canvas = FigureCanvasTkAgg(graph_figure, master=graph_frame)
    graph_canvas.get_tk_widget().pack(fill="both", expand=True)

    cbar = None

    def refresh_graph(env):
        #Updates tje heatmap graph showing the AI's confidence levels for tiles.
        nonlocal cbar
        subplot.clear()

        board = env.game.get_view_board_opponent(1)
        ship_lengths = [len(ship["tiles"]) for ship in env.game.player_1_pieces.values() if not ship["is_sank"]]
        ai_heat_map_board, _, _ = spd_ai.find_best_move(board, ship_lengths)

        heatmap = subplot.imshow(ai_heat_map_board, cmap="hot_r", interpolation="nearest")
        subplot.figure.set_facecolor("gray")
        subplot.set_title("AI Move View")
        subplot.tick_params(axis="x", labelsize=8)
        subplot.tick_params(axis="y", labelsize=8)

        if cbar is None:
            cbar = graph_figure.colorbar(heatmap)
            cbar.set_label("AI Moves Confidence (heatmap)", rotation=270, labelpad=15)

        graph_canvas.draw()

    def update_boards(env):
        #Updates the grid images to reflect the current board state
        if not root.winfo_exists():
            return

        view_board_p1 = env.game.get_view_board_opponent(1)
        for row in range(10):
            for col in range(10):
                code = view_board_p1[row][col]
                if code == -2:
                    img = image_dict["ship_hit"]
                elif code == -1:
                    img = image_dict["water_hit"]
                else:
                    img = image_dict["water"]
                p1_labels[row][col].configure(image=img)
                p1_labels[row][col].image = img

        view_board_p2 = env.game.get_view_board_opponent(2)
        for row in range(10):
            for col in range(10):
                code = view_board_p2[row][col]
                if code == -2:
                    img = image_dict["ship_hit"]
                elif code == -1:
                    img = image_dict["water_hit"]
                else:
                    img = image_dict["water"]
                p2_buttons[row][col].configure(image=img)
                p2_buttons[row][col].image = img

        refresh_graph(env)
        root.update_idletasks()
        root.update()

    episode_label = tk.Label(left_frame, text=f"Episode: {start_episode}/{end_episode}", font=("Arial", 16, "bold"), fg="blue")
    episode_label.grid(row=4, column=0, pady=10)

    for ep in range(start_episode, end_episode):
        print(f"\n=== Episode {ep + 1}/{end_episode} ===")
        episode_label.config(text=f"Episode: {ep + 1}/{end_episode}")

        env = BattleshipRLEnv(seed=None)
        obs = env.reset()

        done = False
        logprobs, values, rewards, entropies = [], [], [], []
        steps = 0
        turns = 0
        misses = 0
        episode_reward = 0.0
        global_episode = ep + 1

        while not done and steps < config.max_steps:
            x = env.get_tensor_observation()
            logits = policy_net(x)
            
            mask = torch.tensor((obs.reshape(-1) == UNHIT_CODE).astype(np.float32), dtype=torch.float32).unsqueeze(0)
            logits = logits.masked_fill(mask == 0, -1e9)
            dist = masked_dist(logits, obs)
            
            action = dist.sample()
            logprob = dist.log_prob(action)
            entropy = dist.entropy()
            value = value_net(x).squeeze(-1)

            next_obs, reward, done, info = env.step(action.item())
            episode_reward += reward

            if not info.get("hit", False):
                misses += 1 #
            turns += 1 #

            logprobs.append(logprob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)
            
            obs = next_obs
            steps += 1

            update_boards(env)
            time.sleep(delay)

        stats["total_episodes"] = global_episode
        stats["total_turns"] += turns
        stats["total_misses"] += misses
        stats["total_reward"] += episode_reward
        if env.winner == 1:
            stats["win_count"] += 1

        returns = []
        g = 0
        for r in reversed(rewards):
            g = r + config.gamma * g
            returns.insert(0, g)

        returns = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.cat(values).squeeze(-1)
        adv = returns - values_t.detach()

        policy_loss = -(torch.cat(logprobs) * adv).mean() - config.entropy_coef * torch.cat(entropies).mean()
        value_loss = F.mse_loss(values_t, returns)
        loss = policy_loss + config.value_coef * value_loss

        policy_opt.zero_grad()
        value_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
        policy_opt.step()
        value_opt.step()

        session_episodes = stats["total_episodes"] - start_episode
        session_win_rate = ((stats["win_count"] - checkpoint_win_count)) if session_episodes > 0 else 0.0
        cumulative = (stats["win_count"] / stats["total_episodes"] * 100) if stats["total_episodes"] > 0 else 0.0
        current_total = global_episode

        print(f"WATCH Ep {ep + 1}: Session {session_win_rate:.1f}% (Cumulative: {cumulative:.1f}%)")
        
        #Periodic saves
        if (ep + 1 - start_episode) % save_freq == 0:
            stats["total_episodes"] = current_total
            watch_mode = f"watch_{env.opponent_type}"
            perform_checkpoint(policy_net, value_net, stats, global_episode, save_freq, mode=watch_mode)

            interval_start = get_interval_start(global_episode, save_freq, start_episode + 1)
            save_summary_log(watch_mode, interval_start, global_episode, stats, is_final=True)

    print("Watch mode complete. Saving final checkpoint.")
    final_checkpoint = {
        "policy_state_dict": policy_net.state_dict(),
        "value_state_dict": value_net.state_dict(),
        "stats": stats,
        "episode": global_episode,
    }
    torch.save(final_checkpoint, resume_from)
    root.destroy()

def get_interval_start(current_episode: int, interval: int, run_start_episode: int = 1) -> int:
    #Calculates what the starting episode is for specific loggin interval.
    if interval <= 0:
        return run_start_episode
    return ((current_episode - run_start_episode) // interval) * interval + run_start_episode

def save_summary_log(mode, start_ep, end_ep, stats, is_final=False):
    #Writes a plain text file to summarize performance metrics
    folder = "summaries" if is_final else "logs"
    log_dir = Path(folder)
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    episode_range = f"{start_ep}-{end_ep}"
    filename = log_dir / f"{mode}_{folder}_{episode_range}_{timestamp}.txt"

    total_eps = max(1, stats.get("total_episodes", 0))

    with open(filename, "w", encoding="utf-8") as f:
        header = f"--- {mode.upper()} {'SESSION FINAL' if is_final else 'INTERVAL'} ---"
        f.write(f"{header}\n")
        f.write(f"Range: {episode_range}\n")
        f.write(f"Total Cumulative Episodes: {total_eps}\n")
        f.write(f"Cumulative Wins: {stats.get('win_count', 0)}\n")
        f.write(f"Cumulative Win Rate: {stats.get('win_count', 0) / total_eps * 100:.1f}%\n")
        f.write(f"Avg Reward: {stats.get('total_reward', 0.0) / total_eps:.2f}\n")
        f.write(f"Avg Turns: {stats.get('total_turns', 0) / total_eps:.2f}\n")
        f.write(f"Avg Misses: {stats.get('total_misses', 0) / total_eps:.2f}\n")

    print(f"[{'SUMMARY' if is_final else 'LOG'}] saved to {folder}/{filename.name}")


def perform_checkpoint(policy_net, value_net, stats, episode, save_freq, mode="train", is_final=False):
    #A standalone helper function to save metrics during operation
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    checkpoint = {
        "policy_state_dict": policy_net.state_dict(),
        "value_state_dict": value_net.state_dict(),
        "stats": stats,
        "episode": episode,
        "timestamp": datetime.now().isoformat(),
    }
    torch.save(checkpoint, output_dir / "rl_battleship.pt")

    start_ep = get_interval_start(episode, save_freq, 1)
    save_summary_log(mode, start_ep, episode, stats, is_final=is_final)
    print(f"[{mode.upper()}] Checkpoint saved: {episode} total episodes completed.")

def train_default():
    #Initializes the standard training sessions using the default configurations
    config = RLConfig()
    resume_path = config.save_path if Path(config.save_path).exists() else None
    trainer = RLTrainer(config, resume_from=resume_path)
    return trainer.train()


if __name__ == "__main__":
    import sys

    #Handles what mode is being executed "random", "watch", or default.
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "watch":
        watch_training_from_start(num_episodes=30, delay=0.005)
    else:
        opp = "random" if mode == "random" else "spd"
        print(f"Starting training against {opp.upper()} AI...")

        config = RLConfig()
        env = BattleshipRLEnv(seed=config.seed, opponent_type=opp)

        resume_path = config.save_path if Path(config.save_path).exists else None
        trainer = RLTrainer(config, resume_from=resume_path)

        trainer.env = env
        trainer.train()