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

        logger.info(" Safe AI Pipeline completed successfully!")

    except Exception as e:
        logger.error(f" Pipeline error: {e}", exc_info=True)
        if config['rollback'].get('enabled', False):
            logger.info(" Rolling back...")
            rollback_handler.rollback_model()

    finally:
        logger.info("Pipeline finished.")

if __name__ == "__main__":
    main()

print("Safe AI Pipeline completed successfully!")
