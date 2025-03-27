import os
import sys
import subprocess
import shutil
import time
import logging
import tempfile

from logs.logger import get_logger
from recursive_improvement.rewriter import Rewriter
from recursive_improvement.sandbox.runner import run_code_and_tests_docker
from utils.logger import setup_logger
from agents.rsi_agent import RSI_Agent
from recursive_improvement.codegen.codegen import generate_code

logger = get_logger("RSIAgent")

# ===============================
# RSI CONFIGURATION
# ===============================
RSI_CONFIG = {
    'target_file': 'agents/dqn_agent.py',
    'backup_folder': 'logs/rsi_backups',
    'evaluation_script': 'evaluate_generated_agent.py',
    'max_iterations': 10,
    'performance_threshold': 0.01
}

# ===============================
# AST-BASED SMART CODE REWRITER
# ===============================
def rule_based_generate_code(existing_code: str) -> str:
    """
    Uses AST to locate and modify hyperparameters in the agent code.
    Supports learning_rate, epsilon, hidden sizes, and more.
    """
    try:
        tree = ast.parse(existing_code)
        changes_made = False

        class RewriteHyperparams(ast.NodeTransformer):
            def visit_Assign(self, node):
                nonlocal changes_made
                if isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id

                    # Rule 1: Adjust learning rate
                    if name == "learning_rate" and isinstance(node.value, ast.Constant):
                        new_val = round(min(node.value.value * 1.1, 0.01), 5)
                        node.value = ast.Constant(value=new_val)
                        changes_made = True
                        logger.info(f"Updated learning_rate → {new_val}")

                    # Rule 2: Lower epsilon
                    elif name == "epsilon" and isinstance(node.value, ast.Constant):
                        new_val = round(max(node.value.value - 0.1, 0.01), 2)
                        node.value = ast.Constant(value=new_val)
                        changes_made = True
                        logger.info(f"Updated epsilon → {new_val}")

                return node

        tree = RewriteHyperparams().visit(tree)
        ast.fix_missing_locations(tree)

        if not changes_made:
            logger.warning("No hyperparameters matched for AST rewrite.")
        return astor.to_source(tree)

    except Exception as e:
        logger.error(f"AST rewrite failed: {e}")
        return existing_code

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
    logger.info("Evaluating code via sandbox...")
    try:
        passed, output = run_code_and_tests_docker(code, RSI_CONFIG['evaluation_script'])
        if not passed:
            logger.warning("Sandbox tests failed.")
            return None
        reward = extract_reward_from_output(output)
        logger.info(f"Evaluation complete: Reward = {reward}")
        return reward
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return None

def extract_reward_from_output(output: str) -> float:
    import re
    match = re.search(r"Reward:\s*([\d\.]+)", output)
    if match:
        return float(match.group(1))
    raise ValueError("Reward not found in evaluation output")

# ===============================
# Backup and Overwrite Function
# ===============================
def backup_and_overwrite(new_code, iteration):
    os.makedirs(RSI_CONFIG['backup_folder'], exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = os.path.join(RSI_CONFIG['backup_folder'], f'dqn_agent_iter{iteration}_{timestamp}.py')
    shutil.copy(RSI_CONFIG['target_file'], backup_path)
    logger.info(f"Backup created: {backup_path}")

    with open(RSI_CONFIG['target_file'], 'w') as f:
        f.write(new_code)
    logger.info("New agent code written to target file.")

# ===============================
# Recursive Self-Improvement Loop
# ===============================
def recursive_self_improvement():
    logger.info("Starting RSI Loop")

    with open(RSI_CONFIG['target_file'], 'r') as f:
        current_code = f.read()

    best_score = evaluate_new_code(current_code)
    if best_score is None:
        logger.warning("Initial evaluation failed. Aborting RSI.")
        return

    for iteration in range(1, RSI_CONFIG['max_iterations'] + 1):
        logger.info(f"🔁 Iteration {iteration}")
        new_code = rule_based_generate_code(current_code)

        if not validate_code(new_code):
            logger.warning("❌ Invalid syntax in generated code. Skipping.")
            continue

        backup_and_overwrite(new_code, iteration)
        new_score = evaluate_new_code(new_code)

        if new_score is None:
            logger.warning("⚠️ Evaluation failed. Rolling back.")
            rollback_to_latest_backup()
            continue

        if new_score > best_score + RSI_CONFIG['performance_threshold']:
            logger.info(f"✅ Improvement! {new_score:.3f} > {best_score:.3f}")
            best_score = new_score
            current_code = new_code
        else:
            logger.warning(f"No gain. Rolling back. Score: {new_score:.3f}")
            rollback_to_latest_backup()

    logger.info("RSI Complete")

# ===============================
# Rollback Handler
# ===============================
def rollback_to_latest_backup():
    files = sorted(os.listdir(RSI_CONFIG['backup_folder']))
    if not files:
        logger.error("No backups found.")
        return
    rollback_file = os.path.join(RSI_CONFIG['backup_folder'], files[-1])
    shutil.copy(rollback_file, RSI_CONFIG['target_file'])
    logger.info(f"Rolled back to {rollback_file}")

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    recursive_self_improvement()
