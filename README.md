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

2. Set Up a Virtual Environment (Recommended):

On **Linux/MacOS**:
   ```console
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

On **Windows (PowerShell)**:
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
   pip install gymnasium[box2d]
   pip install numpy
   ```
Optional: For CUDA (GPU), install PyTorch with the correct CUDA version. See: https://pytorch.org/get-started/locally/
4. Run a basic reinforcement learning task (CartPole with DQN):

   ```console
   python main_cartpole.py
   ```

5. Run Evolutionary Hyperparameter Optimization (CartPole + Evolution):

   ```console
   python main_cartpole_evolve.py
   ```

6. Run multi-task learning agent:

   ```console
   python main_multitask.py
   ```

7. Run Meta-Learning (MAML):):

   ```console
   python main_maml.py
   ```

8. Run Meta-Learning (MAML):):
Docker is required for sandboxing.
Start Docker daemon first.

   ```console
   python main_rsi.py
   ```

## Continuous Integration
- GitHub Actions workflow runs on each push/PR.
- To trigger manually:
  ```bash
  gh workflow run test.yml
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
- [x] Multi-Task RL with shared policies
- [x] Meta-Learning (MAML / Reptile)
- [x] Recursive Self-Improvement (Codegen + Evaluation Loop)
- [ ] Safe AI & Alignment Checks
- [ ] Collaborative Agents & Task Routing
- [ ] Automated R&D Loop

## License
This project is licensed under the MIT License.

## License
Built by The-Outsider-97
With contributions from the SLAI Open AGI Initiative
