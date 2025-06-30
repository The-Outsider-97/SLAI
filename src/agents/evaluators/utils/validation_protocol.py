
import json

from pathlib import Path
from typing import Dict
from dataclasses import dataclass, field

from src.agents.evaluators.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Validation Protocol")
printer = PrettyPrinter

@dataclass
class ValidationProtocol:
    """
    Comprehensive validation configuration based on:
    - ISO/IEC 25010 (Software Quality Requirements)
    - UL 4600 (Standard for Safety for Autonomous Vehicles)
    - EU AI Act (Risk-based Classification)
    """
    
    # Static Analysis Configuration
    static_analysis: Dict = field(default_factory=lambda: {
        'enable': True,
        'security': {
            'owasp_top_10': True,
            'cwe_top_25': True,
            'max_critical': 0,
            'max_high': 3
        },
        'code_quality': {
            'tech_debt_threshold': 0.15,
            'test_coverage': 0.8,
            'complexity': {
                'cyclomatic': 15,
                'cognitive': 20
            }
        }
    })
    
    # Dynamic Testing Parameters
    behavioral_testing: Dict = field(default_factory=lambda: {
        'test_types': ['unit', 'integration', 'adversarial', 'stress'],
        'sample_size': {
            'nominal': 1000,
            'edge_cases': 100,
            'adversarial': 50
        },
        'failure_tolerance': {
            'critical': 0,
            'high': 0.01,
            'medium': 0.05
        }
    })
    
    # Safety & Ethics Configuration
    safety_constraints: Dict = field(default_factory=lambda: {
        'operational_design_domain': {
            'geography': 'global',
            'speed_range': (0, 120),  # km/h
            'weather_conditions': ['clear', 'rain', 'snow']
        },
        'risk_mitigation': {
            'safety_margins': {
                'positional': 1.5,  # meters
                'temporal': 2.0      # seconds
            },
            'fail_operational': True
        },
        'ethical_requirements': {
            'fairness_threshold': 0.8,
            'bias_detection': ['gender', 'ethnicity', 'age'],
            'transparency': ['decision_logging', 'explanation_generation']
        }
    })
    
    # Performance Benchmarks
    performance_metrics: Dict = field(default_factory=lambda: {
        'accuracy': {
            'min_precision': 0.95,
            'min_recall': 0.90,
            'f1_threshold': 0.925
        },
        'efficiency': {
            'max_inference_time': 100,  # ms
            'max_memory_usage': 512,    # MB
            'energy_efficiency': 0.5    # Joules/inference
        },
        'robustness': {
            'noise_tolerance': 0.2,
            'adversarial_accuracy': 0.85,
            'distribution_shift': 0.15
        }
    })
    
    # Compliance & Certification
    compliance: Dict = field(default_factory=lambda: {
        'regulatory_frameworks': ['ISO 26262', 'EU AI Act', 'SAE J3016'],
        'certification_level': 'ASIL-D',
        'documentation': {
            'required': ['safety_case', 'test_reports', 'risk_assessment'],
            'format': 'ISO/IEC 15288'
        }
    })
    
    # Operational Parameters
    operational: Dict = field(default_factory=lambda: {
        'update_policy': {
            'retrain_threshold': 0.10,
            'rollback_strategy': 'versioned',
            'validation_frequency': 'continuous'
        },
        'resource_constraints': {
            'max_compute_time': 3600,  # seconds
            'allowed_hardware': ['CPU', 'GPU'],
            'privacy': ['differential_privacy', 'on-device_processing']
        }
    })

    def __init__(self):
        self.config = load_global_config()
        self.template_path = self.config.get('template_path')

        self.protocol_config = get_config_section('validation_protocol')
        self.static_analysis = self.protocol_config.get('static_analysis', ValidationProtocol.__dataclass_fields__['static_analysis'].default_factory())
        self.behavioral_testing = self.protocol_config.get('behavioral_testing', ValidationProtocol.__dataclass_fields__['behavioral_testing'].default_factory())
        self.safety_constraints = self.protocol_config.get('safety_constraints', ValidationProtocol.__dataclass_fields__['safety_constraints'].default_factory())
        self.performance_metrics = self.protocol_config.get('performance_metrics', ValidationProtocol.__dataclass_fields__['performance_metrics'].default_factory())
        self.compliance = self.protocol_config.get('compliance', ValidationProtocol.__dataclass_fields__['compliance'].default_factory())
        self.operational = self.protocol_config.get('operational', ValidationProtocol.__dataclass_fields__['operational'].default_factory())
        self.full_evaluation_flow = self.protocol_config.get('full_evaluation_flow', {})

    @property
    def validation_protocol(self):
        return self.protocol_config

    def validate_configuration(self):
        """
        Formally verify that validation protocol settings are internally consistent,
        complete, and compatible with certification templates.
        Raises ValueError if inconsistencies are found.
        """
        errors = []
    
        # 1. Validate Static Analysis Config
        if not isinstance(self.static_analysis, dict):
            errors.append("Static analysis configuration must be a dictionary.")
    
        sec = self.static_analysis.get('security', {})
        if sec:
            for key in ['owasp_top_10', 'cwe_top_25', 'max_critical', 'max_high']:
                if key not in sec:
                    errors.append(f"Missing security config: {key}")
    
        code_quality = self.static_analysis.get('code_quality', {})
        if code_quality:
            for key in ['tech_debt_threshold', 'test_coverage', 'complexity']:
                if key not in code_quality:
                    errors.append(f"Missing code quality config: {key}")
    
        complexity = code_quality.get('complexity', {})
        for c_key in ['cyclomatic', 'cognitive']:
            if c_key not in complexity:
                errors.append(f"Missing complexity threshold: {c_key}")
    
        # 2. Validate Behavioral Testing
        bt = self.behavioral_testing
        if not isinstance(bt.get('test_types', []), list):
            errors.append("Test types must be a list.")
        if 'sample_size' not in bt or 'failure_tolerance' not in bt:
            errors.append("Behavioral testing must define sample_size and failure_tolerance.")
    
        # 3. Validate Safety Constraints
        sc = self.safety_constraints
        if 'operational_design_domain' not in sc:
            errors.append("Safety constraints must define an operational design domain.")
        if 'ethical_requirements' not in sc:
            errors.append("Safety constraints must define ethical requirements.")
    
        # 4. Validate Performance Metrics
        perf = self.performance_metrics
        required_perf = ['accuracy', 'efficiency', 'robustness']
        for metric in required_perf:
            if metric not in perf:
                errors.append(f"Performance metric missing: {metric}")
    
        # 5. Validate Compliance Targets
        comp = self.compliance
        if 'regulatory_frameworks' not in comp:
            errors.append("Compliance section missing regulatory frameworks.")
        if 'certification_level' not in comp:
            errors.append("Compliance section missing certification level.")
    
        # 6. Validate Operational Parameters
        op = self.operational
        if 'update_policy' not in op or 'resource_constraints' not in op:
            errors.append("Operational section must define update_policy and resource_constraints.")
    
        # 7. Cross-check with Certification Templates
        try:
            cert_path = Path(self.template_path)
            if not cert_path.exists():
                cert_path = Path(self.template_path)
            templates = json.loads(cert_path.read_text())
    
            domain = "automotive"  # Default fallback domain
            domains = templates.keys()
    
            if domain not in domains:
                errors.append(f"Domain '{domain}' not found in certification templates.")
    
            levels = ['DEVELOPMENT', 'PILOT', 'DEPLOYMENT', 'CRITICAL']
            for lvl in levels:
                if lvl not in templates.get(domain, {}):
                    errors.append(f"Certification template missing phase: {lvl}")
        except Exception as e:
            errors.append(f"Certification template verification failed: {str(e)}")
    
        # 8. Final Decision
        if errors:
            raise ValueError(f"ValidationProtocol consistency check failed:\n" + "\n".join(errors))
        else:
            print("[ValidationProtocol] Configuration validated successfully.")
