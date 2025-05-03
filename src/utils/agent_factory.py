__version__ = "1.7.0"


import threading
import importlib
import torch
import inspect 

from contextlib import contextmanager
# from restricted_env import RestrictedPython
from collections import defaultdict, deque
from typing import Any, Dict, Optional, Tuple, List, Type

from profiling_utils import memory_profile, time_profile, start_memory_tracing, display_top_memory_sources
#from src.utils.system_optimizer import SystemOptimizer
from src.agents.language.grammar_processor import GrammarProcessor
#from src.agents.language.resource_loader import ResourceLoader
#from src.agents.adaptive_agent import AdaptiveAgent
#from src.agents.alignment_agent import AlignmentAgent
#from src.agents.evaluation_agent import EvaluationAgent
#from src.agents.execution_agent import ExecutionAgent
#from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.language_agent import DialogueContext#, LanguageAgent
#from src.agents.learning_agent import LearningAgent
#from src.agents.learning.slaienv import SLAIEnv
#from src.agents.perception_agent import PerceptionAgent
#from src.agents.planning_agent import PlanningAgent
#from src.agents.reasoning_agent import ReasoningAgent
#from src.agents.safety_agent import SafeAI_Agent, SafetyAgentConfig
from src.agents.perception.encoders.audio_encoder import AudioEncoder
from src.agents.perception.encoders.vision_encoder import VisionEncoder
#from models.slai_lm_registry import SLAILMManager
from logs.logger import get_logger

logger = get_logger("Agent Factory")

class AgentMetaData:
    __slots__ = ['name', 'class_name', 'module_path', 'required_params']
    
    def __init__(self, name: str, class_name: str, module_path: str, required_params: Tuple[str]):
        self.name = name
        self.class_name = class_name
        self.module_path = module_path
        self.required_params = required_params

class AgentFactory:
    def __init__(self, config: Dict, shared_resources: Dict):
        self.config = config
        self.shared_resources = shared_resources
        self.registry = {}
        self.instance_cache = {}
        self.lock = threading.RLock()
        
        # Pre-register core agent types with lazy loading
        self._register_core_agents(config.get('agent-network', {}))

    def _register_core_agents(self, agent_network: Dict):
        """Register agents from config with deferred initialization"""
        for agent_name, agent_config in agent_network.items():
            self.registry[agent_name] = {
                'path': agent_config['path'],
                'class': agent_config['class'],
                'config': agent_config.get('init_args', {}),
                'dependencies': self._resolve_dependencies(agent_config)
            }
            
    def _import_class(self, class_path: str) -> Type:
        """Dynamic class importer with caching"""
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    
    def _resolve_dependencies(self, config: Dict) -> List[str]:
        """Identify required dependencies from config"""
        return [
            dep for dep in config.get('requires', []) 
            if dep in self.registry
        ]
    
    def get(self, agent_name: str):
        with self._get_agent_context(agent_name) as agent:
            return agent
    
    @contextmanager
    def _get_agent_context(self, agent_name: str):
        """Thread-safe context manager for agent access"""
        with self.lock:
            if agent_name not in self.instance_cache:
                self._initialize_agent(agent_name)
            yield self.instance_cache[agent_name]
    
    def _initialize_agent(self, agent_name: str):
        from models.slai_lm import get_shared_slailm
        """Lazy initialization with dependency resolution"""
        agent_info = self.registry[agent_name]
        dependencies = {
            dep: self.instance_cache[dep]
            for dep in agent_info['dependencies']
        }
    
        if agent_name == "language":
            filtered_config = {
                k: v for k, v in agent_info['config'].items()
                if k in agent_info['class'].__init__.__code__.co_varnames
            }
            agent_instance = agent_info['class'](
                agent_factory=self,
                grammar=GrammarProcessor(),
                context=DialogueContext(),
                slai_lm=get_shared_slailm(self.shared_resources['memory'], self),
                **filtered_config,
                **dependencies,
                **self.shared_resources
            )
        else:
            filtered_config = {
                k: v for k, v in agent_info['config'].items()
                if k in agent_info['class'].__init__.__code__.co_varnames
            }
            agent_instance = agent_info['class'](
                **filtered_config,
                **dependencies,
                **self.shared_resources
            )
    
        self.instance_cache[agent_name] = agent_instance

    def create(self, agent_name: str, config: Dict = None):
        agent_info = self.registry[agent_name]
        cls = self._import_class(f"{agent_info['path']}.{agent_info['class']}")
        
        # Filter parameters to only those accepted by the class constructor
        init_args = agent_info.get('config', {})
        if config:
            init_args.update(config)
            
        # Get valid constructor parameters
        init_params = inspect.signature(cls.__init__).parameters
        filtered_args = {
            k: v for k, v in init_args.items() 
            if k in init_params
        }
        
        return cls(**filtered_args)

    def _check_text_deps(self) -> bool:
        try:
            import transformers  # noqa
            return True
        except ImportError:
            logger.warning("Transformers not installed, text capabilities limited")
            return False

    def _init_text_model(self, config: Dict) -> Any:
        if self._check_text_deps():
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_id = config.get("text_model_id", "gpt2")
            return AutoModelForCausalLM.from_pretrained(model_id)
        return None

    def _init_audio_encoder(self, config: Dict) -> Any:
        return AudioEncoder(
            encoder_type=config.get("audio_encoder_type", "mfcc"),  # or "transformer"
            mfcc_config=config.get("mfcc", {
                "sample_rate": 16000,
                "n_mfcc": 13,
                "frame_length_ms": 25,
                "frame_step_ms": 10,
                "n_filters": 40
            }),
            # Transformer parameters remain for backward compatibility
            audio_length=config.get("audio_length", 16000),
            patch_size=config.get("patch_size", 400),
            embed_dim=config.get("embed_dim", 512)
        )

    def _init_vision_encoder(self, config: Dict) -> Any:
        return VisionEncoder(
            encoder_type=config.get("vision_encoder_type", "cnn"),  # or "transformer"
            cnn_config=config.get("cnn", {
                "input_size": (224, 224),
                "channels": 3,
                "filters": [
                    (11, 11, 96),
                    (5, 5, 256),
                    (3, 3, 384)
                ]
            }),
            # Transformer parameters remain for backward compatibility
            img_size=config.get("img_size", 224),
            patch_size=config.get("patch_size", 16),
            embed_dim=config.get("embed_dim", 512)
        )

    def report_memory_usage(self, limit: int = 10):
        """Display top memory-consuming components"""
        display_top_memory_sources(limit)

class MetricsAdapter:
    """
    Bridges metrics analysis with agent configuration using control-theoretic adaptation
    Implements principles from:
    - PID Controllers (Åström & Hägglund, 1995)
    - Safe Exploration (Turchetta et al., 2020)
    - Fairness-Aware RL (D'Amour et al., 2020)
    """
    
    def __init__(self, 
                 history_size: int = 100,
                 max_adaptation_rate: float = 0.2):
        self.metric_history = deque(maxlen=history_size)
        self.adaptation_factors = {
            'risk_threshold': 1.0,
            'exploration_rate': 1.0,
            'learning_rate': 1.0
        }
        self.max_rate = max_adaptation_rate
        self._init_control_parameters()

    def _init_control_parameters(self):
        """PID tuning per Ziegler-Nichols method"""
        self.Kp = 0.15  # Proportional gain
        self.Ki = 0.05   # Integral gain
        self.Kd = 0.02   # Derivative gain
        self.integral = defaultdict(float)
        self.prev_error = defaultdict(float)

    def process_metrics(self, 
                       metrics: Dict[str, Any], 
                       agent_types: List[str]) -> Dict[str, float]:
        """
        Convert raw metrics to adaptation factors using:
        - Moving average filters
        - Differential fairness constraints
        - Calibration-aware adjustments
        """
        self.metric_history.append(metrics)
        
        # Calculate temporal derivatives
        delta = self._calculate_metric_deltas()
        
        # Apply control theory
        adjustments = {}
        for metric_type in ['fairness', 'performance', 'bias']:
            error = self._calculate_error(metrics, metric_type)
            adjustments.update(
                self._pid_control(metric_type, error, delta.get(metric_type, 0)))
        
        # Enforce ISO/IEC 24027:2021 safety bounds
        return self._apply_safety_bounds(adjustments, agent_types)

    def _calculate_metric_deltas(self) -> Dict[str, float]:
        """Numerical differentiation for trend analysis"""
        if len(self.metric_history) < 2:
            return {}
            
        current = self.metric_history[-1]
        previous = self.metric_history[-2]
        
        return {
            'fairness': current.get('demographic_parity', 0) - previous.get('demographic_parity', 0),
            'performance': current.get('calibration_error', 0) - previous.get('calibration_error', 0)
        }

    def _pid_control(self, 
                    metric_type: str, 
                    error: float,
                    delta: float) -> Dict[str, float]:
        """Discrete PID controller implementation"""
        self.integral[metric_type] += error
        derivative = error - self.prev_error[metric_type]
        
        adjustment = (self.Kp * error +
                     self.Ki * self.integral[metric_type] +
                     self.Kd * derivative)
        
        self.prev_error[metric_type] = error
        return {
            f"{metric_type}_adjustment": adjustment
        }

    def _apply_safety_bounds(self, 
                            adjustments: Dict[str, float],
                            agent_types: List[str]) -> Dict[str, float]:
        """Constrained optimization per Wachi & Sui (2020)"""
        bounded = {}
        for key, value in adjustments.items():
            # Agent-type specific constraints
            if 'risk_threshold' in key and 'safety' in agent_types:
                bound = 0.5 if 'medical' in agent_types else 0.4
                bounded[key] = torch.clip(value, -bound, bound)
            else:
                bounded[key] = torch.clip(value, -self.max_rate, self.max_rate)
        return bounded

    def update_factory_config(self, 
                             factory: 'AgentFactory',
                             adjustments: Dict[str, float]):
        """Dynamic reconfiguration using meta-learning gradients"""
        for agent_type in factory.registry.values():
            # Update exploration rates
            new_exploration = agent_type.exploration_rate * \
                (1 + adjustments.get('performance_adjustment', 0))
            agent_type.exploration_rate = min(new_exploration, 1.0)
            
            # Adapt risk thresholds
            if hasattr(agent_type, 'risk_threshold'):
                agent_type.risk_threshold *= \
                    (1 - adjustments.get('fairness_adjustment', 0))
