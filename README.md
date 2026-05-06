# Battleship_RL_AI
This project takes the SPD Battleship AI created by JTexpo and uses it to train a Reinforcement Learning AI.

How to run:
-cd "file path"
-python3 -m venv .venv
-Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
-.venv\Scripts\activate.ps1
-pip install -r requirements.txt
-If your powershell now shows ">>>" instead of your file path, enter "exit()"
-python3 -c "import torch; print(torch.__version__)"

Run the various versions:
-mkdir output,logs,summaries, images -Force (Verify all needed folders are present)
-python3 main.py (Player vs SPD AI)
-python3 BattleshipRLEnv.py watch (RL vs SPD with visuals)
-python3 BattleshipRLEnv.py random (RL vs Random Firing AI without visuals)
-python3 BattleshipRLEnv.py (RL vs SPD without visuals)
