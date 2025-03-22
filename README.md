# SLAI - Self-Learning Autonomous Intelligence

An open-source AGI prototype that evolves, learns, and rewrites itself.
SLAI combines **Reinforcement Learning**, **Meta-Learning**, and **Recursive Self-Improvement** into an autonomous research agent.

---

## What It Does
- Evolves deep neural networks (AutoML / NAS)  
- Reinforcement learning agents (DQN, PPO)  
- Meta-learning agents (MAML, Reptile) for few-shot task adaptation  
- Recursive self-improvement: code generation, evaluation, and rewriting  
- Multi-task RL framework  
- Sandbox execution for safety  

---

# How to run

## Clone this repo.
   ```bash
   git clone https://github.com/The-Outsider-97/SLAI.git
   cd SLAI
   ```
2a. Set Up a Virtual Environment (Recommended): **For Linux**
   ```console
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```
2b. Set Up a Virtual Environment (Recommended): **For Windows**
   ```console
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   Note: If you see an error about execution policy:
   ```console
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   .\venv\Scripts\Activate.ps1
   ```
3. Install requirements:
   ```console
   pip install torch
   pip install -r requirements.txt
   pip install pyyaml
   ```
4. Run the main loop:
   ```console
   python main.py
   ```

   If user experiencing errors at this stage, run this command to install PyTorch inside the virtual environment.
   CPU-Only Version (lighter):
   ```console
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

   Confirm Torch Installed:
   ```console
   pip list
   ```
   
6. Run the Tests (Optional but Recommended):
   ```console
   pytest tests/
   ```

## Roadmap
- [x] Basic evolutionary agent
- [ ] Add reinforcement learning agents
- [ ] Implement meta-learning strategy
- [ ] Build interactive dashboard for monitoring

## License
This project is licensed under the MIT License.
