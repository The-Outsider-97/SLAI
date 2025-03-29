"""
Implements STPA (Leveson, 2011) and ISO 21448 (SOTIF) risk frameworks
with dynamic Bayesian adaptation based on:
- Barber, D. (2012) Bayesian Reasoning and Machine Learning
- Kaplan, S. (1997) The Words of Risk Analysis
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class RiskModelParameters:
    """Configuration for dynamic risk assessment"""
    initial_hazard_rates: Dict[str, float]
    learning_rate: float = 0.01
    uncertainty_window: int = 1000

class RiskAdaptation:
    """Real-time risk assessment engine with Bayesian updating"""
    
    def __init__(self, params: RiskModelParameters):
        self.params = params
        self.risk_model = self._initialize_model()
        self.observation_history = []
        
    def _initialize_model(self) -> Dict[str, Tuple[float, float]]:
        """Create prior distributions for each hazard"""
        return {
            hazard: (rate, rate*0.1)  # (mean, variance)
            for hazard, rate in self.params.initial_hazard_rates.items()
        }
    
    def update_model(self, observations: Dict[str, int], operational_time: float):
        """
        Bayesian update of risk parameters using operational data
        
        Args:
            observations: Count of observed hazards
            operational_time: Total operational hours
        """
        for hazard, count in observations.items():
            if hazard not in self.risk_model:
                continue
                
            prior_mean, prior_var = self.risk_model[hazard]
            
            # Calculate posterior
            obs_rate = count / operational_time
            weight = min(1.0, len(self.observation_history)/self.params.uncertainty_window)
            
            new_mean = (1-weight)*prior_mean + weight*obs_rate
            new_var = prior_var * (1 - self.params.learning_rate)
            
            self.risk_model[hazard] = (new_mean, new_var)
        
        self.observation_history.append((observations, operational_time))
    
    def get_current_risk(self, hazard: str) -> Tuple[float, float]:
        """Returns (mean, 95th percentile) of hazard rate"""
        mean, var = self.risk_model.get(hazard, (0.0, 0.0))
        return mean, mean + 1.96*np.sqrt(var)
    
    def generate_safety_case(self) -> Dict:
        """STAMP-based safety argument structure"""
        return {
            "goals": [
                f"Maintain {hazard} rate below {2*mean:.1e}/hr" 
                for hazard, (mean, _) in self.risk_model.items()
            ],
            "assumptions": [
                f"Operational environment remains within design parameters"
            ],
            "evidence": [
                f"Observed {sum(obs[hazard] for obs, _ in self.observation_history)} {hazard} events"
                for hazard in self.risk_model.keys()
            ]
        }
