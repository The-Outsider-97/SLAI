__version__ = "1.7.0"

import ast
import math
import inspect
import pickle
import psutil
import importlib
import tracemalloc
import numpy as np
from pathlib import Path
from inspect import Parameter
# from restricted_env import RestrictedPython
from collections import defaultdict, deque
from typing import Any, Dict, Optional, Tuple, List
from profiling_utils import memory_profile, time_profile, start_memory_tracing, display_top_memory_sources
from src.utils.system_optimizer import SystemOptimizer
from src.agents.language.grammar_processor import GrammarProcessor
from src.agents.language.resource_loader import ResourceLoader
from src.agents.adaptive_agent import AdaptiveAgent
from src.agents.alignment_agent import AlignmentAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.agents.execution_agent import ExecutionAgent
from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.language_agent import LanguageAgent, DialogueContext
from src.agents.learning_agent import LearningAgent
from src.agents.learning.slaienv import SLAIEnv
from src.agents.perception_agent import PerceptionAgent
from src.agents.planning_agent import PlanningAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.safety_agent import SafeAI_Agent, SafetyAgentConfig
from src.agents.perception.encoders.audio_encoder import AudioEncoder
from src.agents.perception.encoders.vision_encoder import VisionEncoder
from models.slai_lm_registry import SLAILMManager
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
    def __init__(self, shared_resources: Optional[Dict[str, Any]] = None, optimizer: 'SystemOptimizer' = None, **config):
        self.agent_registry: Dict[str, AgentMetaData] = {}
        self.shared_resources = shared_resources or {}
        self._memorized_agents: Dict[str, Any] = {}
        self._fallback_map = self._create_fallback_map()
        self._discover_agents()
        self.registry = {}
        self.metrics_adapter = MetricsAdapter()
        self.optimizer = optimizer
        self.lazy_registry = {}


        if not tracemalloc.is_tracing():
            start_memory_tracing()
        
        slailm_instance = SLAILMManager.get_instance("default",
                                                     shared_memory=self.shared_resources.get("shared_memory"),
                                                     agent_factory=self)
        #safety_agent = self._memorized_agents.get("safety")
        #if "safety" not in self.registry:
        #    self._register_safety_agent

        self._pool = {}  # Add this line
        self.agent_registry: Dict[str, AgentMetaData] = {}
        self.shared_resources = shared_resources or {}

        self.register("adaptive", lambda config: AdaptiveAgent(
            shared_memory=self.shared_resources.get("shared_memory"),
            agent_factory=self,
            **config.get("init_args", {})
        ))
        self.register("alignment", lambda config: AlignmentAgent(
            shared_memory=self.shared_resources.get("shared_memory"),
            agent_factory=self,
            sensitive_attrs=config.get("init_args", {}).get("sensitive_attrs", []),            
            config=config.get("init_args", {}),
            monitor_config=config.get("init_args", {}).get("monitor"),
            correction_policy=config.get("correction_policy")
        ))
        self.register("evaluation", lambda config: self._safe_create_evaluation_agent(config))

        self.register("execution", lambda config: ExecutionAgent(
            shared_memory=self.shared_resources.get("shared_memory"),
            agent_factory=self,
            **config.get("init_args", {})
        ))
        self.register("knowledge", lambda config: KnowledgeAgent(
            shared_memory=self.shared_resources.get("shared_memory"),
            agent_factory=self,
            **config.get("init_args", {})
        ))
        slailm_instance = SLAILMManager.get_instance("default",
            shared_memory=self.shared_resources.get("shared_memory"),
            agent_factory=self
        )
        self.register("language", lambda config: LanguageAgent(
            shared_memory=self.shared_resources.get("shared_memory"),
            agent_factory=self,
            grammar=slailm_instance.grammar_processor,
            context=slailm_instance.dialogue_context,
            slai_lm=slailm_instance,
            config=config.get("init_args", {})
        ))

        self.register("learning", lambda config: LearningAgent(
            slai_lm=slailm_instance,
            agent_factory=self,
            safety_agent=self.registry.get("safety", lambda _: None)({}),
            env=self.shared_resources.get("env"),
            config=config.get("init_args", {}),
            shared_memory=self.shared_resources.get("shared_memory")
        ))
        self.register("learner", self.registry["learning"]
        )
        self.register("perception", lambda config: PerceptionAgent(
            config={
                **{
                    "modalities": ["vision", "text", "audio"],
                    "embed_dim": 100,
                    "projection_dim": 256,
                    "batch_size": 8,
                    "learning_rate": 0.001,
                    "epochs": 20
                },
                **config.get("init_args", {})
            },
            shared_memory=self.shared_resources.get("shared_memory"),
            agent_factory=self,
        ))
        self.register("planning", lambda config: PlanningAgent(
            shared_memory=self.shared_resources.get("shared_memory"),
            agent_factory=self,
            **config.get("init_args", {})
        ))
        self.register("reasoning", lambda config: ReasoningAgent(
            # 1. Get shared_memory and agent_factory from the config's init_args
            shared_memory=config.get("init_args", {}).get("shared_memory"),
            agent_factory=config.get("init_args", {}).get("agent_factory"),
            
            # 2. CONFIG-DRIVEN PARAMS
            tuple_key=tuple(
                config.get("init_args", {}).get("tuple_key", "subject|predicate|object").split("|")
            ) if isinstance(config.get("init_args", {}).get("tuple_key", "subject|predicate|object"), str) 
              else tuple(config.get("init_args", {}).get("tuple_key", ("subject", "predicate", "object"))),
            storage_path=config.get("init_args", {}).get("storage_path", "src/agents/knowledge/knowledge_db.json"),
            contradiction_threshold=config.get("init_args", {}).get("contradiction_threshold", 0.25),
            rule_validation=config.get("init_args", {}).get("rule_validation", {}),
            nlp_integration=config.get("init_args", {}).get("nlp_integration", {}), 
            inference=config.get("init_args", {}).get("inference", {}), 
            llm=config.get("init_args", {}).get("llm", self.shared_resources.get("llm")),  # Fallback to factory's llm if not provided
            language_agent=config.get("init_args", {}).get("language_agent", self._memorized_agents.get("language"))
        ))
        self.register("safety", lambda config: SafeAI_Agent(
            shared_memory=self.shared_resources.get("shared_memory"),
            agent_factory=self,
            alignment_agent_cls=AlignmentAgent,
            config=SafetyAgentConfig(
                constitutional_rules=config.get("init_args", {}).get("constitutional_rules", {}),
                risk_thresholds=config.get("init_args", {}).get("risk_thresholds", {
                    "safety": 0.01, "security": 0.001, "privacy": 0.05
                }),
                audit_level=config.get("init_args", {}).get("audit_level", 2),
                enable_rlhf=config.get("init_args", {}).get("enable_rlhf", True)
            )
        ))

    def _safe_create_evaluation_agent(self, config):
        try:
            full_config = {
                **self.shared_resources.get('config', {}),
                **config.get("init_args", {})
            }
            reasoning_args = {
                "shared_memory": self.shared_resources.get("shared_memory"),  # From factory's shared_resources
                "agent_factory": self,  # Pass the current AgentFactory instance
                "tuple_key": "subject|predicate|object",
                "storage_path": "src/agents/knowledge/knowledge_db.json",
                "contradiction_threshold": 0.25,
                "rule_validation": {},
                "nlp_integration": {},
            }
            full_config["reasoning_config"] = {
                "init_args": reasoning_args
            }
            
            return EvaluationAgent(
                shared_memory=self.shared_resources.get("shared_memory"),
                agent_factory=self,
                config=full_config
            )
        except Exception as e:
            logger.warning(f"Failed to instantiate EvaluationAgent: {str(e)}")
            return None

    @staticmethod
    def _validate_init_signature(cls, config):
        sig = inspect.signature(cls.__init__)
        missing = [
            p.name for p in sig.parameters.values()
            if p.default == Parameter.empty
            and p.name not in config
            and p.name != "self"
            and p.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
        ]
        if missing:
            raise ValueError(f"Missing params for {cls.__name__}: {missing}")

    @memory_profile
    @time_profile
    def create(self, agent_type: str, config: dict, system_metrics: dict = None):
        if agent_type in self._pool:
            return self._pool[agent_type].reconfigure(config)
        if agent_type in self.lazy_registry and not self.lazy_registry[agent_type][1]:
            init_fn, _ = self.lazy_registry[agent_type]
            init_fn()
            self.lazy_registry[agent_type] = (init_fn, True)

        if not isinstance(config, dict):
            raise TypeError(f"[AgentFactory] Config must be a dict, got {type(config)} with value: {config}")
        """Optimized agent creation"""
        if self.optimizer and system_metrics:
            system_metrics = {
                "cpu_usage": psutil.cpu_percent(),
                "model_size": 512,  # Approx. MB usage
                "latency_slo": 0.15,
                "throughput": 5,
                "alloc_history": [psutil.virtual_memory().used / 1e6] * 5
            }
            recommendations = self.optimizer.optimize_throughput(system_metrics)
            config["init_args"]["batch_size"] = recommendations["max_batch_size"]

        build_config_fn = getattr(self, f"_build_{agent_type}_config", None)
        if agent_type in self.registry:
            factory_fn = self.registry[agent_type]

        # Get the actual agent class from the factory function
        if isinstance(factory_fn, (type, classmethod)):
            agent_cls = factory_fn
        else:
            # Extract class from lambda closure
            agent_cls = factory_fn.__closure__[0].cell_contents.__class__

        # Build final configuration
        final_config = build_config_fn(config) if build_config_fn else config
        
        # Validate configuration against init signature
        self._validate_init_signature(agent_cls, final_config.get("init_args", {}))

        if callable(self.registry[agent_type]):
            return self.registry[agent_type](final_config)
    
        if agent_type in self.registry:
            if callable(self.registry[agent_type]):
                return self.registry[agent_type](config)

            # === Modified validation ===
            init_sig = inspect.signature(agent_cls.__init__)
            missing_args = []
            for name, param in init_sig.parameters.items():
                if name == "self" or \
                param.kind in (inspect.Parameter.VAR_POSITIONAL, 
                                inspect.Parameter.VAR_KEYWORD):
                    continue
                if param.default == inspect.Parameter.empty and name not in final_config:
                    missing_args.append(name)
                    
            if missing_args:
                raise ValueError(f"[AgentFactory] Missing required arguments for agent '{agent_type}': {missing_args}")
    
            if agent_type in self.registry:
                if callable(self.registry[agent_type]):
                    return self.registry[agent_type](config)

                # === Validate config before instantiation ===
                init_sig = inspect.signature(agent_cls.__init__)
                missing_args = []
                for name, param in init_sig.parameters.items():
                    if name == "self":
                        continue
                    if param.default == inspect.Parameter.empty and name not in final_config:
                        missing_args.append(name)
                    if agent_type == "perception":
                        assert "modalities" in config, "PerceptionAgent requires 'modalities' in config"
                if missing_args:
                    raise ValueError(f"[AgentFactory] Missing required arguments for agent '{agent_type}': {missing_args}")


            init_args = final_config.get("init_args", {})
            return agent_cls(
                shared_memory=self.shared_resources.get("shared_memory"),
                agent_factory=self,
                **init_args
                )

        logger.info(f"[AgentFactory] Using fallback for unregistered agent type: {agent_type}")
        return self._create_fallback_agent(agent_type, config)
          
    def reconfigure_from_metrics(self, metrics: Dict[str, Any]):
        """Public interface for metric-driven adaptation"""
        adjustments = self.metrics_adapter.process_metrics(
            metrics,
            agent_types=list(self.registry.keys())
        )
        self.metrics_adapter.update_factory_config(self, adjustments)

    def register(self, name: str, agent_cls):
        self.registry[name] = agent_cls
        
    def adapt_from_metrics(self, metrics_data: Dict[str, Any]) -> None:
        """
        Update factory configurations based on metric feedback
        Implements inverse RL strategy from Ng & Russell (2000)
        """
        # Fairness adaptation
        if metrics_data.get('fairness_violations', 0) > 0:
            new_risk_threshold = self._calculate_risk_adjustment(
                violations=metrics_data['fairness_violations'],
                base_threshold=0.35
            )
            self._update_agent_configs(
                'risk_threshold', 
                max(0.1, min(0.5, new_risk_threshold)))
            
        # Performance adaptation
        if metrics_data.get('calibration_error', 0) > 0.1:
            self._adjust_learning_rates(
                factor=1.5 if metrics_data['calibration_error'] > 0.15 else 1.2
            )
            
    def _calculate_risk_adjustment(self, violations: int, base_threshold: float) -> float:
        """Control-theoretic adjustment using PI (Proportional-Integral) control.
        
        Implements the formula:
        adjusted_threshold = base_threshold + Kp * violations + Ki * integral_term
        
        Where:
        - Kp: Proportional gain (immediate response to current violations)
        - Ki: Integral gain (response to accumulated past violations)
        - integral_term: Sum of all past violations (discrete integration)
        
        Reference: Åström, K. J., & Hägglund, T. (1995). PID Controllers: Theory, Design, and Tuning.
        
        Args:
            violations: Number of current fairness violations detected
            base_threshold: Initial risk threshold before adjustment
            
        Returns:
            Adjusted risk threshold based on control law
        """
        Kp = 0.05  # Proportional gain (immediate response)
        Ki = 0.01   # Integral gain (historical accumulation)

        integral = self._integral_term(violations) # Calculate integral term using accumulated violations
        adjusted_threshold = base_threshold + (Kp * violations) + (Ki * integral) # PI control formula
        return max(0.1, min(0.5, adjusted_threshold)) # Clamp values to prevent extreme adjustments

    def _integral_term(self, violations: int) -> float:
        """Maintains and updates the integral of historical violations.
        
        Implements discrete-time integration using accumulation:
        integral += current_violations
        
        Includes anti-windup protection by enforcing maximum bounds
        
        Returns:
            Accumulated integral value (sum of all past violations)
        """
        # Initialize integral storage if not exists
        if not hasattr(self, '_risk_integral'):
            self._risk_integral = 0.0

        self._risk_integral += violations # Update integral with current violations
        
        # Anti-windup: Prevent excessive integral accumulation
        self._risk_integral = min(self._risk_integral, 1000)  # Max 1000 violation-history
        
        return self._risk_integral

    def _create_fallback_map(self) -> Dict[str, Tuple[str, str]]:
        return {
            agent_type: (f"src.agents.{agent_type}_agent", f"{agent_type.title()}Agent")
            for agent_type in [
                'language', 'learning', 'perception', 'adaptive',
                'evaluation', 'execution', 'knowledge', 'alignment',
                'planning', 'reasoning', 'safety'
            ]
        }

    def _discover_agents(self) -> None:
        cache_path = Path(f"agent_discovery_v{__version__}.cache")
        if cache_path.exists():
            self.agent_registry = pickle.load(cache_path.open('rb'))
            return
        base_dir = Path(__file__).parent.parent / "agents"
        for agent_file in base_dir.glob("*_agent.py"):
            try:
                module_path = f"src.agents.{agent_file.stem}"
                agent_name = agent_file.stem.replace("_agent", "")
                
                with open(agent_file, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())
                    
                class_def = next(
                    (node for node in tree.body 
                     if isinstance(node, ast.ClassDef)), None
                )
                
                if class_def:
                    required_params = self._parse_init_params(class_def)
                    self.agent_registry[agent_name] = AgentMetaData(
                        name=agent_name,
                        class_name=class_def.name,
                        module_path=module_path,
                        required_params=required_params
                    )
                    
            except Exception as e:
                logger.warning(f"Agent discovery failed for {agent_file}: {str(e)}")
        
        pickle.dump(self.agent_registry, cache_path.open('wb'))

    def _parse_init_params(self, class_def: ast.ClassDef) -> Tuple[str]:
        init_method = next(
            (node for node in class_def.body 
             if isinstance(node, ast.FunctionDef) and node.name == "__init__"),
            None
        )
        
        if not init_method or not init_method.args.args:
            return ()
            
        return tuple(
            arg.arg for arg in init_method.args.args[1:]  # Skip self
            if not (arg.annotation and self._is_optional(arg.annotation))
        )

    def _is_optional(self, annotation: ast.AST) -> bool:
        return isinstance(annotation, ast.Subscript) and \
               isinstance(annotation.value, ast.Name) and \
               annotation.value.id == "Optional"

    def parse_intent(self, prompt):
        # Add input safety checks
        if not prompt or len(prompt.strip()) < 1:
            return {"intent": "unknown", "confidence": 0.0}
        
        # Limit input length and handle encoding
        clean_prompt = prompt.strip()[:200]  # Truncate long inputs
        try:
            # Add fallback for empty token lists
            tokens = self.llm.tokenizer.tokenize(clean_prompt)
            if not tokens:
                return {"intent": "generic", "confidence": 0.5}

        except Exception as e:
            logger.error(f"Intent parsing error: {str(e)}")
            return {"intent": "error", "confidence": 0.0}

    def _create_fallback_agent(self, agent_name: str, config: Dict) -> Any:
        #with RestrictedPython():
        #    return agent_class(**config)    
        if agent_name not in self._fallback_map:
            raise ValueError(f"Unknown agent: {agent_name}")
            
        module_path, class_name = self._fallback_map[agent_name]
        try:
            module = importlib.import_module(module_path)
            agent_class = getattr(module, class_name)
            
            # Apply agent-specific config builder if available
            build_config_fn = getattr(self, f"_build_{agent_name}_config", None)
            if build_config_fn:
                final_config = build_config_fn(config)
            else:
                final_config = config
                
            return self._initialize_agent(agent_class, final_config)
        except ModuleNotFoundError as e:
            logger.critical(f"Missing dependency for {agent_name}: {e}")
            raise
        except ImportError as e:
            logger.error(f"Import error in {module_path}: {e}")
            raise

    def _validate_config(self, metadata: AgentMetaData, config: Dict) -> None:
        missing = [param for param in metadata.required_params if param not in config]
        if missing:
            raise ValueError(
                f"Missing required parameters for {metadata.name}: {missing}"
            )

    def _instantiate_agent(self, metadata: AgentMetaData, config: Dict) -> Any:
        try:
            module = importlib.import_module(metadata.module_path)
            agent_class = getattr(module, metadata.class_name)
            return self._initialize_agent(agent_class, config)
        except ImportError as e:
            raise RuntimeError(f"Module import failed: {str(e)}")
        except AttributeError:
            raise RuntimeError(f"Class {metadata.class_name} not found in {metadata.module_path}")

    def _initialize_agent(self, agent_class: type, config: Dict) -> Any:
        if agent_class.__name__ == "DialogueContext":
            from models.slai_lm import SLAILM
            return agent_class(
                llm=config.get("llm", SLAILM()),
                history=config.get("history", []),
                summary=config.get("summary", None),
                memory_limit=config.get("memory_limit", 1000),
                enable_summarization=config.get("enable_summarization", True),
                summarizer=config.get("summarizer", None)
            )
        clean_config = {k: v for k, v in config.items() if k != "class" and k != "path"}
        return agent_class(**config)

    def _agent_specific_config(self, agent_class: type, config: Dict) -> Dict:
        agent_type = agent_class.__name__.lower()
        config_builder = {
            'adaptive': self._build_adaptive_config,
            'alignment': self._build_alignment_config,
            'evaluation': self._build_evaluation_config,
            'execution': self._build_execution_config,
            'knowledge': self._build_knowledge_config,
            'language': self._build_language_config,
            'learning': self._build_learning_config,
            'perception': self._build_perception_config,
            'planning': self._build_planning_config,
            'reasoning': self._build_reasoning_config,
            'safety': self._build_safety_config
        }.get(agent_type.split('agent')[0], lambda _: {})
        
        return config_builder(config)

    def _build_adaptive_config(self, config: dict) -> dict:
        return {
            "args": config.get("args", ()),
            "kwargs": config.get("kwargs", {}),
            # Add other necessary parameters
        }

    def _build_alignment_config(self, config: dict) -> dict:
        return {
            "args": config.get("args", ()),
            "kwargs": config.get("kwargs", {}),
            # Add other parameters
        }

    def _build_evaluation_config(self, config):
        return {
            "init_args": {
                "evaluation_type": "standard",
                "output_path": "logs/eval.jsonl"
            }
        }

    def _build_execution_config(self, config: dict) -> dict:
        return {
            "args": config.get("args", ()),
            "kwargs": config.get("kwargs", {}),
            # Add other parameters
        }

    def _build_knowledge_config(self, config: dict) -> dict:
        return {
            "args": config.get("args", ()),
            "kwargs": config.get("kwargs", {}),
            # Add other necessary parameters
        }

    def _build_language_config(self, config: dict) -> dict:
        from models.slai_lm import SLAILM
        llm_instance = config.get("llm", SLAILM(
            shared_memory=self.shared_resources.get("shared_memory"),
            agent_factory=self
        ))
        dialogue_context = DialogueContext(llm=llm_instance)

        return {
            "llm": llm_instance,
            "args": config.get("args", ()),
            "kwargs": config.get("kwargs", {}),
            "config": {
                "history": config.get("history", []),
                "summary": config.get("summary", None),
                "memory_limit": config.get("memory_limit", 1000),
                "enable_summarization": config.get("enable_summarization", True),
                "summarizer": config.get("summarizer", None),
                "cache_size": config.get("cache_size", 1000)
            }

        }
    
    def _build_learning_config(self, config: Dict) -> Dict:
        return {
            "algorithm": config.get("algorithm", "dqn"),
            "epsilon": 0.1,
            "discount": 0.95,
            "memory_limit": 50000,
            "learning_rate": 0.001,
            "args": config.get("args", ()),
            "kwargs": config.get("kwargs", {})
        }

    def _build_perception_config(self, config: Dict) -> Dict:
        return {
            "args": config.get("args", ()),
            "kwargs": config.get("kwargs", {}),
            "modalities": config.get("modalities", ["vision", "text", "audio"]),
            "embed_dim": config.get("embed_dim", 512),
            "projection_dim": config.get("projection_dim", 256),
            'audio_encoder': self._init_audio_encoder(config),
            'vision_encoder': self._init_vision_encoder(config),
        }

    def _build_planning_config(self, config: dict) -> dict:
        return {
            "args": config.get("args", ()),
            "kwargs": config.get("kwargs", {}),
            # Add other necessary parameters
        }

    def _build_reasoning_config(self, config: dict) -> dict:
        init_args = config.get("init_args", {})
        return {
            "init_args": {
                "args": init_args.get("args", ()),
                "kwargs": init_args.get("kwargs", {}),
                **init_args
            }
        }

    def _build_safety_config(self, config: dict) -> dict:
        return {
            "args": config.get("args", ()),
            "kwargs": config.get("kwargs", {}),
            # Add other parameters
        }

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
                bounded[key] = np.clip(value, -bound, bound)
            else:
                bounded[key] = np.clip(value, -self.max_rate, self.max_rate)
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
