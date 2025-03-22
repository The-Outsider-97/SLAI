import os
import sys
import subprocess
import shutil
import time
import logging
import tempfile
from recursive_improvement.sandbox.runner import run_code_and_tests
from utils.logger import setup_logger
from recursive_improvement.codegen.codegen import generate_code

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
# ===============================
def generate_new_code(existing_code):
    return generate_code(existing_code)

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
def evaluate_new_code(code):
    passed, output = run_code_and_tests(code)
    if not passed:
        logger.warning("Tests failed during evaluation.")
        return None
    # Parse reward from output or use your custom reward function
    return calculate_reward(output)

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
