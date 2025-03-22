import os
import sys
import subprocess
import shutil
import time
import logging
import tempfile

from utils.logger import setup_logger

logger = setup_logger('SLAI-RSI', level=logging.DEBUG)

# ===============================
# RSI CONFIGURATION
# ===============================
RSI_CONFIG = {
    'target_file': 'agents/dqn_agent.py',
    'backup_folder': 'logs/rsi_backups',
    'evaluation_script': 'evaluate_generated_agent.py',
    'max_iterations': 10,
    'performance_threshold': 0.01  # Replace if you want target score
}

# ===============================
# Generate New Code Function
# (Placeholder for codegen model)
# ===============================
def generate_new_code(existing_code):
    """
    Dummy code generator. Replace this with a language model call in RSI phase 2.
    """
    logger.info("Generating new code from existing agent...")
    
    # Example mutation: change learning rate if found
    new_code = existing_code.replace('learning_rate = 0.001', 'learning_rate = 0.0005')
    
    # You can extend this by adding layers, activations, etc.
    
    return new_code

# ===============================
# Validate Python Code Function
# ===============================
def validate_code(code_string):
    try:
        compile(code_string, RSI_CONFIG['target_file'], 'exec')
        logger.info("New code validated: Syntax OK")
        return True
    except SyntaxError as e:
        logger.error(f"Syntax Error in generated code:\n{e}")
        return False

# ===============================
# Evaluate New Code Function
# ===============================
def evaluate_new_code():
    """
    Runs the evaluation script for the agent and parses performance.
    """
    logger.info("Evaluating the new code...")

    try:
        result = subprocess.run(
            [sys.executable, RSI_CONFIG['evaluation_script']],
            capture_output=True,
            text=True,
            timeout=300
        )
        logger.info(f"Evaluation Script Output:\n{result.stdout}")

        # Parse the performance from the output
        for line in result.stdout.splitlines():
            if "Average Reward:" in line:
                reward_str = line.split(":")[-1].strip()
                return float(reward_str)

    except subprocess.SubprocessError as e:
        logger.error(f"Evaluation failed: {e}")
        return None

    return None

# ===============================
# Backup and Overwrite Function
# ===============================
def backup_and_overwrite(new_code, iteration):
    # Ensure backup folder exists
    os.makedirs(RSI_CONFIG['backup_folder'], exist_ok=True)

    # Backup current file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = os.path.join(RSI_CONFIG['backup_folder'], f'dqn_agent_iter{iteration}_{timestamp}.py')

    shutil.copy(RSI_CONFIG['target_file'], backup_path)
    logger.info(f"Backup created at {backup_path}")

    # Overwrite target file with new code
    with open(RSI_CONFIG['target_file'], 'w') as f:
        f.write(new_code)
    logger.info(f"Overwritten {RSI_CONFIG['target_file']} with new code.")

# ===============================
# Recursive Self-Improvement Loop
# ===============================
def recursive_self_improvement():
    logger.info("Starting Recursive Self-Improvement (RSI)...")

    # Load the current agent code
    with open(RSI_CONFIG['target_file'], 'r') as f:
        current_code = f.read()

    best_performance = evaluate_new_code()
    logger.info(f"Initial Performance: {best_performance}")

    for iteration in range(1, RSI_CONFIG['max_iterations'] + 1):
        logger.info(f"====== RSI Iteration {iteration} ======")

        # Generate new candidate code
        new_code = generate_new_code(current_code)

        # Validate new code syntax
        if not validate_code(new_code):
            logger.warning("Code rejected due to validation failure. Skipping iteration.")
            continue

        # Backup current file and overwrite
        backup_and_overwrite(new_code, iteration)

        # Evaluate new agent performance
        new_performance = evaluate_new_code()

        if new_performance is None:
            logger.warning("Evaluation returned None. Rolling back to last working version.")
            continue

        # Check if performance improves
        if new_performance > best_performance + RSI_CONFIG['performance_threshold']:
            logger.info(f"New code improved! Reward: {new_performance} > Previous: {best_performance}")
            best_performance = new_performance
            current_code = new_code  # Promote the new code
        else:
            logger.warning(f"No improvement. Rolling back. Reward: {new_performance}")
            
            # Rollback to backup (restore previous code)
            backup_path = sorted(os.listdir(RSI_CONFIG['backup_folder']))[-1]
            rollback_file = os.path.join(RSI_CONFIG['backup_folder'], backup_path)

            shutil.copy(rollback_file, RSI_CONFIG['target_file'])
            logger.info(f"Rolled back to {rollback_file}")

    logger.info("RSI Process Completed.")


# ===============================
# Main Function Entry Point
# ===============================
if __name__ == "__main__":
    recursive_self_improvement()
