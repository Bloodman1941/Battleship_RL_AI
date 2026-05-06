The Environment:
There is a 10x10 board with the possible states of a ship/miss/unknown
With a 10x10 board, that is 100 possible shots.
The RL AI is rewarded for hits, consecutive hits, and wins. Loses points for missing and losing.

Neural Networks:
It utilizes both a Policy Network and Value Network
The Policy Network is learning strategies and maps possible states and best actions
The Value Network defines the specific value of a single action
With these working together, the Policy finds the best actions and the Values finds the best reward estimations.

Checkpoints:
We have a checkpoint every 500 episodes to log the RL AIs performance through various metrics of wins, misses, rewards.
the logs folder stores checkpoint data
the summaries folder stores final results

How to run:
cd "file path"
python3 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\activate.ps1
pip install -r requirements.txt
If your powershell now shows ">>>" instead of your file path, enter "exit()"
python3 -c "import torch; print(torch.__version__)"

Run the various versions:
mkdir output,logs,summaries, images -Force (Verify all needed folders are present)
python3 main.py (Player vs SPD AI)
python3 BattleshipRLEnv.py watch (RL vs SPD with visuals)
python3 BattleshipRLEnv.py random (RL vs Random Firing AI without visuals)
python3 BattleshipRLEnv.py (RL vs SPD without visuals)