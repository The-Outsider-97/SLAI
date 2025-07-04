"""
Implements STPA (Leveson, 2011) and ISO 21448 (SOTIF) risk frameworks
with dynamic Bayesian adaptation based on:
- Barber, D. (2012) Bayesian Reasoning and Machine Learning
- Kaplan, S. (1997) The Words of Risk Analysis
"""

import os
import threading
import time
import numpy as np
import yaml, json
import jsonschema

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from src.agents.evaluators.utils.config_loader import load_global_config, get_config_section
# from src.agents.evaluators.utils.background_scheduler import BackgroundScheduler
from src.agents.evaluators.evaluators_memory import EvaluatorsMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Adaptive Risk")
printer = PrettyPrinter

SAFETY_CASE_SCHEMA = {
    "type": "object",
    "properties": {
        "metadata": {"type": "object"},
        "system_description": {"type": "object"},
        "control_structure": {"type": "object"},
        "safety_requirements": {"type": "object"},
        "hazard_analysis": {"type": "object"},
        "evidence_base": {"type": "object"},
        "validation": {"type": "object"}
    },
    "required": ["metadata", "hazard_analysis", "safety_requirements"]
}

class BackgroundScheduler:
    """Custom thread-based scheduler for risk adaptation tasks"""
    
    def __init__(self):
        self.jobs = []
        self.thread = None
        self.active = False

    def add_job(self, func, trigger, hours=None, next_run_time=None):
        """Add a periodic job to the scheduler"""
        if trigger != 'interval':
            raise ValueError("Only interval trigger is supported")
        
        self.jobs.append({
            'func': func,
            'interval': timedelta(hours=hours),
            'next_run': next_run_time or datetime.now()
        })

    def start(self):
        """Start the scheduler in a background thread"""
        if self.active:
            return
            
        self.active = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        """Main scheduler loop"""
        while self.active:
            now = datetime.now()
            for job in self.jobs:
                if now >= job['next_run']:
                    try:
                        job['func']()
                    except Exception as e:
                        logger.error(f"Job execution failed: {str(e)}")
                    job['next_run'] = now + job['interval']
            time.sleep(60)  # Check every minute

    def stop(self):
        """Stop the scheduler"""
        self.active = False
        if self.thread:
            self.thread.join(timeout=5)

class RiskAdaptation:
    """Real-time risk assessment engine with Bayesian updating"""
    
    def __init__(self):
        self.config = load_global_config()
        self.max_safety_case_versions = self.config.get('documentation', {}).get('versioning', {}).get('max_versions', 7) 

        self.risk_config = get_config_section('risk_adaptation')
        self.learning_rate = self.risk_config.get('learning_rate')
        self.uncertainty_window = self.risk_config.get('uncertainty_window')

        self.hazard_config = get_config_section('initial_hazard_rates')
        self.system_failure = self.hazard_config.get('system_failure')
        self.sensor_failure = self.hazard_config.get('sensor_failure')
        self.unexpected_behavior = self.hazard_config.get('unexpected_behavior')

        self.memory = EvaluatorsMemory()
        self.risk_model = self._initialize_model()
        self.observation_history = []
        self.safety_case_versions = {}
        self._init_report_scheduler()

        logger.info(f"Risk Adaptation succesfully initialized")

    def _initialize_model(self) -> Dict[str, Tuple[float, float]]:
        """Create prior distributions for each hazard"""
        self.observation_history = [
            ({h: 0 for h in self.hazard_config}, 1.0)
            for _ in range(2)
        ]
        
        return {
            hazard: (float(rate), float(rate) * 0.1)
            for hazard, rate in self.hazard_config.items()
        }

    def _init_report_scheduler(self):
        """Initialize automated report generation with custom scheduler"""
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(
            func=self.generate_automated_report,
            trigger='interval',
            hours=24,
            next_run_time=datetime.now()
        )
        self.scheduler.start()
        logger.info("Background scheduler started for daily reports")

    def update_model(self, observations: Dict[str, int], operational_time: float):
        """
        Bayesian update of risk parameters using operational data
        
        Args:
            observations: Count of observed hazards
            operational_time: Total operational hours
        """
        if operational_time <= 0:
            logger.warning("Invalid operational time - skipping update")
            return
            
        if not any(count > 0 for count in observations.values()):
            logger.debug("No relevant observations - skipping update")
            return
        uncertainty_window = self.uncertainty_window
        learning_rate = self.learning_rate
        for hazard, count in observations.items():
            if hazard not in self.risk_model:
                continue
                
            prior_mean, prior_var = self.risk_model[hazard]
            
            # Calculate posterior
            obs_rate = count / operational_time
            weight = min(1.0, len(self.observation_history)/uncertainty_window)

            new_mean = (1-weight)*prior_mean + weight*obs_rate
            new_var = prior_var * (1 - learning_rate)
            
            self.risk_model[hazard] = (new_mean, new_var)
        
        self.observation_history.append((observations, operational_time))
    
    def get_current_risk(self, hazard: str) -> Dict[str, Any]:
        """Returns comprehensive risk analysis for a specific hazard"""
        if hazard not in self.risk_model:
            logger.warning(f"Hazard '{hazard}' not found in risk model")
            return {}
        
        mean, var = self.risk_model[hazard]
        std_dev = np.sqrt(var)
        
        # Calculate trend data
        historical_means = [obs[0].get(hazard, 0) for obs in self.observation_history]
        if len(historical_means) >= 2:
            try:
                trend = np.polyfit(range(len(historical_means)), historical_means, 1)[0]
            except np.linalg.LinAlgError:
                trend = 0.0  # Fallback for numerical instability
        else:
            trend = 0.0  # Insufficient data
        return {
            "hazard_id": hazard,
            "risk_metrics": {
                "current_mean": float(mean),
                "variance": float(var),
                "standard_deviation": float(std_dev),
                "confidence_intervals": {
                    "90%": (mean - 1.645*std_dev, mean + 1.645*std_dev),
                    "95%": (mean - 1.96*std_dev, mean + 1.96*std_dev),
                    "99%": (mean - 2.576*std_dev, mean + 2.576*std_dev)
                }
            },
            "trend_analysis": {
                "historical_data_points": len(historical_means),
                "slope": float(trend),
                "stability": "improving" if trend < 0 else "degrading" if trend > 0 else "stable"
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def generate_safety_case(self) -> Dict:
        """STAMP-based safety argument structure with JSON template"""
        template = {
            "metadata": {
                #**template['metadata'],
                "system": "Autonomous Decision System",
                "version": f"1.{len(self.safety_case_versions)+1}",
                "generation_date": datetime.now().isoformat(),
                "standard_compliance": ["STPA", "ISO 21448"]
            },
            "system_description": {
                "purpose": "AI-driven operational decision system",
                "boundary_conditions": "Urban environment operations",
                "operational_domain": "Mixed traffic scenarios"
            },
            "control_structure": {
                "controllers": ["Planning Module", "Risk Assessment Engine"],
                "actuators": ["Motion Controller", "Communication Interface"],
                "sensors": ["Lidar", "Camera", "V2X"],
                "controlled_processes": ["Vehicle Trajectory", "Collision Avoidance"]
            },
            "safety_requirements": {
                "goals": [
                    {
                        "id": f"SG-{hazard.upper()}",
                        "description": f"Maintain {hazard} rate below {2*mean:.1e}/hr",
                        "verification_method": "Operational monitoring",
                        "target_value": 2*mean
                    }
                    for hazard, (mean, _) in self.risk_model.items()
                ]
            },
            "hazard_analysis": {
                "identified_hazards": [
                    {
                        "id": f"HAZ-{hazard.upper()}",
                        "description": hazard,
                        "current_risk": self.get_current_risk(hazard),
                        "mitigation_strategy": "Dynamic risk adaptation",
                        "safety_constraints": [
                            f"System shall maintain {hazard} occurrence below {2*self.risk_model[hazard][0]:.1e}/hr"
                        ]
                    }
                    for hazard in self.risk_model.keys()
                ]
            },
            "evidence_base": {
                "operational_history": {
                    "total_operational_hours": sum(t for _, t in self.observation_history),
                    "observed_events": [
                        {
                            "hazard": hazard,
                            "total_occurrences": sum(obs.get(hazard, 0) for obs, _ in self.observation_history),
                            "last_occurrence": next(
                                (datetime.fromtimestamp(ts).isoformat()
                                 for obs, ts in reversed(self.observation_history)
                                 if hazard in obs), "Never")
                        }
                        for hazard in self.risk_model.keys()
                    ]
                },
                "model_parameters": {
                    "learning_rate": self.learning_rate,
                    "update_window": self.uncertainty_window,
                    "model_version": "Bayesian-1.3"
                }
            },
            "validation": {
                "assumptions": [
                    "Environmental conditions remain within operational design domain",
                    "Sensor inputs maintain minimum required accuracy"
                ],
                "limitations": [
                    "Does not account for unknown-unknown failure modes",
                    "Assumes continuous power supply"
                ]
            }
        }
        try:
            jsonschema.validate(instance=template, schema=SAFETY_CASE_SCHEMA)
        except jsonschema.ValidationError as e:
            logger.error(f"Safety case validation failed: {str(e)}")
            raise

        # Store in memory with versioning
        case_id = f"safety_case_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.memory.add(
            entry=template,
            tags=["safety_case", "automated_report"],
            priority="high"
        )
        self.safety_case_versions[case_id] = template
        self._generate_documentation(template)

        return template

    def _generate_documentation(self, safety_case: Dict):
        """Generate Markdown documentation"""
        doc_content = f"""# Safety Case Documentation\n\nVersion {safety_case['metadata']['version']}\n\n"""
        
        # System Description
        doc_content += "## System Description\n"
        doc_content += f"**Purpose**: {safety_case['system_description']['purpose']}\n\n"
        
        # Hazard Analysis
        doc_content += "## Hazard Analysis\n"
        for hazard in safety_case['hazard_analysis']['identified_hazards']:
            doc_content += f"### {hazard['description']}\n"
            doc_content += f"**Current Risk Level**: {hazard['current_risk']['risk_metrics']['current_mean']}\n\n"
        
        # Save to docs directory
        os.makedirs("src/agents/evaluators/docs/safety_cases", exist_ok=True)
        filename = f"src/agents/evaluators/docs/safety_cases/case_v{safety_case['metadata']['version']}.md"
        with open(filename, 'w') as f:
            f.write(doc_content)
            
        logger.info(f"Generated documentation at {filename}")

    def generate_automated_report(self):
        """Generate comprehensive report package"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "safety_cases": list(self.safety_case_versions.values())[-3:],
            "current_risks": [
                self.get_current_risk(hazard)
                for hazard in self.risk_model.keys()
            ],
            "system_status": {
                "operational_hours": sum(t for _, t in self.observation_history),
                "memory_usage": self.memory.get_statistics()
            }
        }
        
        # Save report
        filename = f"src/agents/evaluators/reports/risk_report_{datetime.now().strftime('%Y%m%d')}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Store report in memory
        self.memory.add(
            entry=report,
            tags=["automated_report"],
            priority="medium"
        )
        logger.info(f"Generated automated report: {filename}")
        return filename
    
    def reset_model(self,
                    retain_last_n: int = 0,
                    override_hazard_rates: Dict[str, float] = None,
                    reset_reason: str = "manual reset"):
        """
        Resets the Bayesian risk model to its initial state with optional configuration overrides.
        
        Args:
            retain_last_n (int): Number of most recent observation entries to retain (default: 0)
            override_hazard_rates (Dict[str, float]): New initial hazard priors if overriding defaults
            reset_reason (str): Text label describing the reason for reset (logged for audit trail)
        """
        printer.status("RESET", f"Risk model reset triggered: {reset_reason}", "warning")

        # Optionally override initial priors
        if override_hazard_rates:
            logger.info(f"Overriding hazard priors with: {override_hazard_rates}")
            self.initial_hazard_rates = override_hazard_rates

        # Retain last N observation entries (if specified)
        retained_history = self.observation_history[-retain_last_n:] if retain_last_n > 0 else []

        # Reset observation history
        self.observation_history = retained_history
        logger.info(f"Retained last {len(self.observation_history)} observations after reset")

        # Reset risk model using updated (or default) priors
        self.risk_model = {
            hazard: (float(rate), float(rate) * 0.1)
            for hazard, rate in self.initial_hazard_rates.items()
        }

        # Log reset event in memory
        self.memory.add(
            entry={
                "event": "risk_model_reset",
                "timestamp": datetime.now().isoformat(),
                "reason": reset_reason,
                "retained_observations": len(retained_history),
                "new_hazard_rates": self.initial_hazard_rates
            },
            tags=["risk_reset", "system_event"],
            priority="medium"
        )

        logger.info("Risk model successfully reinitialized.")

    def get_safety_case_history(self) -> List[Dict]:
        """Retrieve historical safety cases directly from versions"""
        # Return most recent cases (up to max_versions)
        versions = list(self.safety_case_versions.values())
        return versions[-self.max_safety_case_versions :]

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Adaptive Risk ===\n")
    risk = RiskAdaptation()

    print(f"{risk}")
    print(f"\n* * * * * Phase 2 * * * * *\n")

    risk.generate_safety_case()

    logger.info("Audit Report:\n" + json.dumps(risk.generate_safety_case(), indent=4))
    print(json.dumps (risk.generate_safety_case(), indent=4))
    print(f"\n* * * * * Phase 3 * * * * *\n")
    # Generate and store safety case
    safety_case = risk.generate_safety_case()
    
    # Access documentation
    risk._generate_documentation(safety_case)
    
    # Get historical cases
    history = risk.get_safety_case_history()
    
    # Manual report trigger
    risk.generate_automated_report()
    
    # Validate existing case
    jsonschema.validate(instance=history[0], schema=SAFETY_CASE_SCHEMA)

    print("\n=== Successfully Ran Adaptive Risk ===\n")
