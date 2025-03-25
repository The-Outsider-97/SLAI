import os
import sys
import yaml
import torch
import logging
from modules.data_handler import DataHandler
from modules.model_trainer import ModelTrainer
from modules.security_manager import SecurityManager
from modules.monitoring import Monitoring
from modules.compliance_auditor import ComplianceAuditor
from rnd_loop.evaluator import Evaluator
from rnd_loop.experiment_manager import ExperimentManager
from rnd_loop.hyperparam_tuner import HyperparamTuner
from deployment.git.rollback_handler import RollbackHandler
from agent.safe_ai_agent import SafeAI_Agent
from collaborative.shared_memory import SharedMemory

config_file = "config.yaml"
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Configure logging
os.makedirs(config['run']['output_dir'], exist_ok=True)
logging.basicConfig(
    filename=config['logging']['log_file'],
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=getattr(logging, config['logging']['level'].upper(), logging.INFO)
)
logger = logging.getLogger('SafeAI')

def main():
    logger.info(" Starting Safe AI Pipeline...")

    # Initialize Components
    data_handler = DataHandler()
    model_trainer = ModelTrainer()
    security_manager = SecurityManager()
    monitoring = Monitoring()
    compliance_auditor = ComplianceAuditor()
    rollback_handler = RollbackHandler(models_dir=config['run']['output_dir'], backup_dir=config['rollback']['backup_dir'])

    try:
        # Step 1: Data Handling & Fairness Check
        logger.info("Loading and preprocessing data...")
        raw_data = data_handler.load_data(config['paths']['data_source'])

        if config['fairness'].get('enforce_fairness', False):
            data_handler.check_data_fairness(raw_data)

        clean_data = data_handler.preprocess_data(raw_data)

        # Step 2: Model Training
        logger.info(" Training model...")
        model = model_trainer.train_model(clean_data)

        # Step 3: Security Hardening
        if config['security'].get('encrypt_models', False):
            logger.info("Applying model security...")
            security_manager.secure_model(model)

        if config['security'].get('enable_threat_detection', False):
            security_manager.check_for_threats()

        # Step 4: Compliance Audit
        if config['compliance'].get('enable_audit', False):
            logger.info("⚖ Running compliance audit...")
            compliance_auditor.run_audit()

        # Step 5: Monitoring
        if config['monitoring'].get('enable_monitoring', False):
            logger.info("Starting monitoring...")
            monitoring.start(model, data_handler)

        # Step 6: Evaluation (SafeAI Agent)
        logger.info("Evaluating SafeAI agent...")
        evaluator = Evaluator(shared_memory=SharedMemory())

        eval_result = evaluator.evaluate_agent(
            agent=SafeAI_Agent(shared_memory=SharedMemory()),
            task_data={
                "policy_risk_score": 0.32,
                "task_type": "meta_learning"
            },
            metadata={"experiment": "baseline_risk_check"}
        )

        logger.info(f"SafeAI Evaluation Result: {eval_result}")

        logger.info(" Safe AI Pipeline completed successfully!")

        tuner = HyperparamTuner(
            agent_class=SafeAI_Agent,
            search_space={
                "risk_threshold": [0.3, 0.2, 0.1],
                "compliance_weight": [0.5, 1.0]  # Optional if your agent supports it
            },
            base_task={
                "policy_risk_score": 0.27,
                "task_type": "reinforcement_learning"
            },
            shared_memory=shared_memory,
            max_trials=6
        )

        best_result = tuner.run_grid_search()
        logger.info(f"Best Tuning Result: {best_result}")
    
    except Exception as e:
        logger.error(f" Pipeline error: {e}", exc_info=True)
        if config['rollback'].get('enabled', False):
            logger.info(" Rolling back...")
            rollback_handler.rollback_model()

        logger.info("Running multiple SafeAI experiments...")

        manager = ExperimentManager(shared_memory=shared_memory)

        results = manager.run_experiments(
            agent_configs=[
                {
                    "agent_class": SafeAI_Agent,
                    "init_args": {"shared_memory": shared_memory, "risk_threshold": 0.2},
                    "name": "safe_v1"
                },
                {
                    "agent_class": SafeAI_Agent,
                    "init_args": {"shared_memory": shared_memory, "risk_threshold": 0.1},
                    "name": "safe_strict"
                }
            ],
            task_data={
                "policy_risk_score": 0.27,
                "task_type": "reinforcement_learning"
            }
        )

        top = manager.summarize_results(sort_key="risk_score", minimize=True)[0]
        logger.info(f"🏆 Best Agent: {top['agent']} with score {top['result']['risk_score']}")
    
    finally:
        logger.info("Pipeline finished.")

if __name__ == "__main__":
    main()

print("Safe AI Pipeline completed successfully!")
