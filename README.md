# Self-Improving AI

An open-source experiment in building an AI that evolves its own models.

## What it does
- Evolves simple neural networks.
- Evaluates performance.
- Iterates improvements over generations.

## How to run
1. Clone this repo.
   ```console
   git clone https://github.com/The-Outsider-97/SLAI.git
   cd SLAI
   ```
2. Set Up a Virtual Environment (Recommended): **For Linux**
   ```console
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```
2. Set Up a Virtual Environment (Recommended): **For Windows**
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
   ```
4. Run the main loop:
   ```console
   python main.py
   ```
5. Run the Tests (Optional but Recommended):
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
