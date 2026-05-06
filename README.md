# Battleship_RL_AI
This project takes the SPD Battleship AI created by JTexpo and uses it to train a Reinforcement Learning AI.

Is it possible to train an AI that can beat a mathematically perfect SPD AI? No, no it can't. But it is fun to train the AI and see it improve. My code is not perfect, but this was a fun project for me. Do show the guy who made the original SPD AI, https://github.com/JTexpo, some love. He also made a video that explains the SPD AI https://www.youtube.com/watch?v=LE315dE81Bo&t=114s. I am to lazy to make a whole YouTube video to show off this RL AI.

The below instructions are done in Windows Powershell, so some commands may need to be edited if you're on a different platform.

How to run:
1) cd "file path"
2) python3 -m venv .venv
3) Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
4) .venv\Scripts\activate.ps1
5) pip install -r requirements.txt
6) If your powershell now shows ">>>" instead of your file path, enter "exit()"
7) python3 -c "import torch; print(torch.__version__)"

Run the various versions:
1) mkdir output,logs,summaries, images -Force (Verify all needed folders are present)
2) python3 main.py (Player vs SPD AI)
3) python3 BattleshipRLEnv.py watch (RL vs SPD with visuals)
4) python3 BattleshipRLEnv.py random (RL vs Random Firing AI without visuals)
5) python3 BattleshipRLEnv.py (RL vs SPD without visuals)
