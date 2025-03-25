import os
import sys
import yaml
import json
import time
import torch
import logging
import subprocess
from logs.logger import get_logger
from logs.logs_parser import LogsParser
from alignment_checks.bias_detection import BiasDetection
from alignment_checks.ethical_constraints import EthicalConstraints
from alignment_checks.fairness_evaluator import FairnessEvaluator
from evaluators.behavioral_tests import BehavioralTests
from evaluators.reward_function import RewardFunction
from evaluators.static_analysis import StaticAnalysis
from deployment.git.rollback_handler import RollbackHandler
from hyperparam_tuning.bayesian_search import BayesianSearch
from hyperparam_tuning.grid_search import GridSearch
from hyperparam_tuning.tuner import HyperParamTuner
from logging.handlers import RotatingFileHandler

file_handler = RotatingFileHandler('logs/run.log', maxBytes=10*1024*1024, backupCount=5)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Strategy selection for hyperparam tuning
strategy = config["tuning"]["strategy"]

if strategy == "grid":
    config_file = config["configs"]["grid_config"]
else:
    config_file = config["configs"]["bayesian_config"]

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Create logger
logger = logging.getLogger('AutoTuneOrchestrator')
logger.setLevel(logging.INFO)

# Console handler (stdout)
console_handler = logging.StreamHandler(stream=sys.stdout)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# File handler (run.log)
file_handler = logging.FileHandler('logs/run.log', mode='a', encoding='utf-8')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add both handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

class AutoTuneOrchestrator:
    """
    Orchestrator for training, evaluation, behavioral testing, reward-based evaluation,
    static code analysis, and automated corrective actions including rollback and retrain.
    """

    def __init__(self):
        # Configs
        self.bias_threshold = 0.1
        self.reward_threshold = 70.0
        self.max_retries = 5

        # Core Handlers
        self.rollback_handler = RollbackHandler(
            models_dir='models/',
            backup_dir='models/backups/'
        )        
        self.hyperparam_tuner = HyperParamTuner(
            config_path='config.yaml',
            evaluation_function=self.rl_agent_evaluation
        )

        # Components
        self.logs_parser = LogsParser(
            log_dir='logs/',
            bias_threshold=self.bias_threshold,
            reward_threshold=self.reward_threshold,
            rollback_handler=self.rollback_handler,
            hyperparam_tuner=self.hyperparam_tuner
        )

        self.behavioral_tests = BehavioralTests(
            rollback_handler=self.rollback_handler,
            hyperparam_tuner=self.hyperparam_tuner
        )

        self.reward_function = RewardFunction(
            rollback_handler=self.rollback_handler,
            hyperparam_tuner=self.hyperparam_tuner,
            safety_thresholds={
                'negative_reward_limit': -50,
                'alignment_violation_limit': 3
            }
        )

        self.static_analyzer = StaticAnalysis(
            codebase_path='src/',
            rollback_handler=self.rollback_handler,
            hyperparam_tuner=self.hyperparam_tuner,
            thresholds={
                'max_warnings': 5,
                'critical_issues': True
            }
        )
        self.bayesian_optimizer = BayesianSearch(
            config_file=config["configs"]["bayesian_config"],
            evaluation_function=self.rl_agent_evaluation,
            n_calls=10,
            n_random_starts=2
        )

        self.grid_optimizer = GridSearch(
            config_file='hyperparam_tuning/example_grid_config.json',
            evaluation_function=self.rl_agent_evaluation
        )

        self._init_behavioral_tests()

    def _init_behavioral_tests(self):
        """
        Define behavioral test cases for the agent.
        """
        def validate_greet(response):
            return response == "Hello!"

        def validate_farewell(response):
            return response == "Goodbye!"

        self.behavioral_tests.add_test_case("greet", "Agent should greet politely", validate_greet)
        self.behavioral_tests.add_test_case("farewell", "Agent should say goodbye", validate_farewell)

    def run_training_pipeline(self):
        """
        Full training pipeline with evaluation, reward monitoring, static code analysis,
        and corrective actions.
        """
        logger.info(" Starting AutoTune Training Pipeline")

        retry_count = 0
        while retry_count < self.max_retries:
            logger.info(f" Training Run {retry_count + 1} / {self.max_retries}")

            # STEP 1: Run Static Code Analysis before training
            logger.info("🛠️ Running Static Code Analysis...")
            self.static_analyzer.run_static_analysis()

            # STEP 2: Train the AI agent
            self.train_agent()

            # STEP 3: Parse Logs & Evaluate Metrics
            report = self.logs_parser.parse_logs()

            # STEP 4: Behavioral Tests
            logger.info(" Running Behavioral Tests...")
            self.behavioral_tests.run_tests(self.simulated_agent_function)

            # STEP 5: Evaluate Reward Function
            logger.info(" Evaluating Reward Function...")
            state = {'user': 'UserABC'}
            action = 'recommend_product'
            outcome = {
                'reward': 10,
                'harm': False,
                'bias_detected': False,
                'discrimination_detected': False
            }
            reward = self.reward_function.compute_reward(state, action, outcome)
            logger.info(f"Reward Function Computed Reward: {reward}")

            # STEP 6: Hyperparameter Optimization
            logger.info("🔎 Triggering Hyperparameter Tuning from CLI...")

            # STEP 7: Run Grid Hyperparameter Optimization (after agent evaluation cycle)
            logger.info("🔎 Running Bayesian Hyperparameter Search...")
            best_params, best_score, _ = self.bayesian_optimizer.run_search()
            logger.info(f"Best hyperparameters from Bayesian optimization: {best_params}") 
            
            # STEP 8: Run Bayesian Hyperparameter Optimization (after agent evaluation cycle)
            logger.info("🔎 Running Grid Hyperparameter Search...")
            best_grid_params = self.grid_optimizer.run_search()
            logger.info(f"Best hyperparameters from GridSearch: {best_grid_params}")
            
            # STEP 9: Decide on Retraining or Rollback
            action_taken = self.decision_policy(report)

            if not action_taken:
                logger.info(" Model passed all checks. Ending pipeline.")
                break

            retry_count += 1

        if retry_count >= self.max_retries:
            logger.warning("⚠️ Max retries reached. Manual intervention recommended.")

    def train_agent(self):
        """
        Placeholder for the actual agent training logic.
        """
        logger.info(" Training AI Agent...")
        time.sleep(2)
        logger.info(" Training complete. Logs and metrics generated.")

    def simulated_agent_function(self, input_data):
        """
        Simulates the agent's response for testing purposes.
        """
        if input_data == "greet":
            return "Hello!"
        elif input_data == "farewell":
            return "Goodbye!"
        else:
            return "I don't understand."

    def rl_agent_evaluation(self, params):
        """
        Dummy evaluation function that simulates RL agent performance
        (used by Bayesian optimizer).
        """
        learning_rate = params['learning_rate']
        num_layers = params['num_layers']
        activation_function = params['activation']

        logger.info(f"Evaluating RL agent with: lr={learning_rate}, layers={num_layers}, activation={activation_function}")

        # Dummy score based on distance from ideal params
        score = -((learning_rate - 0.01) ** 2 + (num_layers - 3) ** 2)
        return score

    def decision_policy(self, report=None):
        """
        Evaluate parsed logs and decide whether to retrain or rollback.
        """
        logger.info("🔎 Evaluating performance and alignment metrics...")

        try:
            with open('logs/parsed_metrics.json', 'r') as f:
                report = json.load(f)
        except Exception as e:
            logger.error(f" Failed to load parsed metrics report: {e}")
            return False

        parity_diff = abs(report.get('statistical_parity', {}).get('parity_difference', 0.0))
        tpr_diff = abs(report.get('equal_opportunity', {}).get('tpr_difference', 0.0))
        reward_score = report.get('performance', {}).get('best_reward', 0.0)

        corrective_action = False

        if parity_diff > self.bias_threshold or tpr_diff > self.bias_threshold:
            logger.warning(f"⚠️ Bias thresholds breached: ParityDiff={parity_diff}, TPRDiff={tpr_diff}")
            self.rollback_handler.rollback_model()
            corrective_action = True

        if reward_score < self.reward_threshold:
            logger.warning(f"⚠️ Reward threshold breached: {reward_score}")
            self.hyperparam_tuner.run_tuning_pipeline()
            corrective_action = True

        return corrective_action

if __name__ == "__main__":
    orchestrator = AutoTuneOrchestrator()

    tuner = HyperParamTuner(
        config_path=config_file,
        evaluation_function=orchestrator.rl_agent_evaluation,
        strategy=strategy,
        n_calls=config["tuning"]["n_calls"],
        n_random_starts=config["tuning"]["n_random_starts"]
    )

    orchestrator.run_training_pipeline()
