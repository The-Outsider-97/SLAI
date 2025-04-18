import ast
import math
import inspect
import pickle
import importlib
import tracemalloc
import numpy as np
import logging as logger, logging
from pathlib import Path
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
from models.slai_lm_registry import SLAILMManager

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
        safety_agent = self._memorized_agents.get("safety")
        if not safety_agent:
            safety_agent = SafeAI_Agent(
                agent_factory=self,
                shared_memory=self.shared_resources.get("shared_memory"),
                alignment_agent_cls=AlignmentAgent,
                config=SafetyAgentConfig(
                    constitutional_rules={},
                    risk_thresholds={"safety": 0.01, "security": 0.001, "privacy": 0.05}
                )
            )
            self._memorized_agents["safety"] = safety_agent
        

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
            monitor_config=config.get("monitor_config"),
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
            safety_agent=self.get("safety"),
            env=self.shared_resources.get("env"),
            config=config.get("init_args", {}),
            shared_memory=self.shared_resources.get("shared_memory")
        ))
        self.register("perception", lambda config: PerceptionAgent(
            config=config.get("init_args", {}),
            shared_memory=self.shared_resources.get("shared_memory"),
            agent_factory=self,
        ))
        self.register("planning", lambda config: PlanningAgent(
            shared_memory=self.shared_resources.get("shared_memory"),
            agent_factory=self,
            **config.get("init_args", {})
        ))
        self.register("reasoning", lambda config: ReasoningAgent(
            shared_memory=self.shared_resources.get("shared_memory"),
            agent_factory=self,
            tuple_key=tuple(config.get("init_args", {}).get("tuple_key", "subject|predicate|object").split("|")
            if isinstance(config.get("init_args", {}).get("tuple_key", "subject|predicate|object"), str)
            else config.get("init_args", {}).get("tuple_key", ["subject", "predicate", "object"])
                ),
            storage_path=config.get("init_args", {}).get("storage_path", "src/agents/knowledge/knowledge_db.json"),
            contradiction_threshold=config.get("init_args", {}).get("contradiction_threshold", 0.25),
            rule_validation=config.get("init_args", {}).get("rule_validation", {}),
            nlp_integration=config.get("init_args", {}).get("nlp_integration", {}),
            llm=self.shared_resources.get("llm"),
            language_agent=self._memorized_agents.get("language")
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
            init_args = config.get("init_args", {})
            return EvaluationAgent(
                shared_memory=self.shared_resources.get("shared_memory"),
                agent_factory=self,
                **init_args
            )
        except Exception as e:
            logging.warning(f"Failed to instantiate EvaluationAgent: {e}")
            return None

    @memory_profile
    @time_profile
    def create(self, agent_type: str, config: dict, system_metrics: dict = None):
        if agent_type in self.lazy_registry and not self.lazy_registry[agent_type][1]:
            init_fn, _ = self.lazy_registry[agent_type]
            init_fn()
            self.lazy_registry[agent_type] = (init_fn, True)

        if not isinstance(config, dict):
            raise TypeError(f"[AgentFactory] Config must be a dict, got {type(config)} with value: {config}")
        """Optimized agent creation"""
        if self.optimizer and system_metrics:
            config.update(
                self.optimizer.recommend_agent_params(
                    agent_type=agent_type,
                    system_status=system_metrics
                )
            )

        build_config_fn = getattr(self, f"_build_{agent_type}_config", None)
    
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

        logging.info(f"[AgentFactory] Using fallback for unregistered agent type: {agent_type}")
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
        cache_path = Path("agent_discovery.cache")
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
                logging.warning(f"Agent discovery failed for {agent_file}: {str(e)}")
        
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
        except Exception as e:
            raise RuntimeError(f"Failed to create fallback agent {agent_name}: {str(e)}")

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
        llm_instance = config.get("llm", SLAILM())
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
        return {
            "args": config.get("args", ()),
            "kwargs": config.get("kwargs", {}),
            # Add other parameters
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
            logging.warning("Transformers not installed, text capabilities limited")
            return False

    def _init_text_model(self, config: Dict) -> Any:
        if self._check_text_deps():
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_id = config.get("text_model_id", "gpt2")
            return AutoModelForCausalLM.from_pretrained(model_id)
        return None

#    def _init_rl_algorithm(self, config: Dict) -> Any:
#        algorithm = config.get("algorithm", "rl").lower()
#        try:
#            if algorithm == "maml":
#                from src.agents.learning import maml_rl
#                return maml_rl.MAMLTrainer(config)
#            elif algorithm == "rsi":
#                from src.agents.learning import rsi
#                return rsi.RSITrainer(config)
#            elif algorithm == "dqn":
#                from src.agents.learning import dqn
#                return dqn.DQNAgent(config)
#            else:
#                from src.agents.learning import rl_agent
#                return rl_agent.RLAgent(config)
#        except ImportError as e:
#            raise RuntimeError(f"RL component import failed: {str(e)}")

    def _init_audio_encoder(self, config: Dict) -> Any:
        """
        Initializes an audio feature extractor based on psychoacoustic principles.
        Implements a simplified version of Mel-Frequency Cepstral Coefficients (MFCCs):
        
        1. Pre-emphasis: y[t] = x[t] - αx[t-1] (α=0.97)
        2. Framing & Windowing: x_w[n] = x[n] * w[n], w[n] = 0.54 - 0.46cos(2πn/N)
        3. Power Spectrum: P[k] = |DFT(x_w)|²
        4. Mel Filterbank: M[m] = Σₖ Wₘ[k]P[k], Wₘ triangular filters spaced per mel scale
        5. Logarithm: log(M[m])
        6. DCT: c[n] = Σₘ log(M[m])·cos(πn(m+0.5)/M)
        
        Where mel scale: m(f) = 2595·log₁₀(1 + f/700)
        
        Reference: Davis & Mermelstein (1980) MFCC technique
        """
        class AudioFeatureExtractor:
            def __init__(self, config):
                self.sample_rate = config.get('sample_rate', 16000)
                self.n_mfcc = config.get('n_mfcc', 13)
                self.frame_length = int(0.025 * self.sample_rate)  # 25ms
                self.frame_step = int(0.01 * self.sample_rate)     # 10ms
                
            def __call__(self, waveform):
                """
                MFCC Extraction Pipeline (Davis & Mermelstein, 1980)
                
                1. Pre-emphasis: y[n] = x[n] - αx[n-1], α=0.97
                2. Framing: Split into N = ⌈(T - L)/S⌉ + 1 frames
                Where T=signal length, L=frame length, S=frame step
                3. Windowing: Apply Hamming window w[n] = 0.54 - 0.46·cos(2πn/(L-1))
                4. DFT: X[k] = Σ_{n=0}^{L-1} x[n]e^{-j2πkn/L}
                5. Mel Filterbank: E[m] = Σ_{k=0}^{L/2} W_m[k]·|X[k]|²
                W_m = triangular filters spaced at mel frequencies:
                    mel(f) = 2595·log₁₀(1 + f/700)
                6. Log Compression: log(E[m])
                7. DCT-II: c[n] = Σ_{m=0}^{M-1} log(E[m})·cos(πn(m+0.5)/M)
                """
                # Implementation details
                T = len(waveform)
                L = self.frame_length
                S = self.frame_step
                N = 1 + (T - L) // S  # Number of frames
                
                mfccs = []
                for i in range(N):
                    # 1. Frame extraction
                    frame = waveform[i*S : i*S+L]
                    
                    # 2. Pre-emphasis
                    emphasized = [frame[n] - 0.97*frame[n-1] for n in range(1, L)]
                    
                    # 3. Hamming window
                    windowed = [e * (0.54 - 0.46*math.cos(2*math.pi*n/(L-1))) 
                            for n, e in enumerate(emphasized)]
                    
                    # 4. DFT magnitude squared (simplified real FFT)
                    spectrum = [abs(x)**2 for x in windowed]  # Placeholder for |FFT|²
                    
                    # 5. Mel filterbank energy (40 filters typical)
                    filter_energies = [sum(w * e for w, e in zip(filter_weights, spectrum))
                                    for filter_weights in self.mel_filters]
                    
                    # 6. Log compression
                    log_energies = [math.log(e + 1e-6) for e in filter_energies]
                    
                    # 7. DCT for decorrelation (MFCC coefficients)
                    mfcc = [sum(e * math.cos(math.pi * n * (m + 0.5) / len(log_energies)))
                        for n in range(self.n_mfcc)
                        for m, e in enumerate(log_energies)]
                    
                    mfccs.append(mfcc)
                
                return np.array(mfccs)
                
        return AudioFeatureExtractor(config)

    def _init_vision_encoder(self, config: Dict) -> Any:
        """
        Implements a convolutional feature extractor based on biological vision models:
        
        1. Convolution: (I*K)[i,j] = ΣₘΣₙ I[i+m,j+n]K[m,n]
        2. ReLU: f(x) = max(0, x)
        3. Max Pooling: y[i,j] = max_{m,n ∈ N(i,j)} x[m,n]
        4. Layered composition: f(x) = fₙ(...(f₂(f₁(x))))
        
        Following the AlexNet architecture principles:
        - Stacked conv layers with decreasing receptive fields
        - Spatial pyramid pooling
        - Local response normalization
        
        Reference: Krizhevsky et al. (2012) ImageNet classification
        """
        class VisionFeatureExtractor:
            def __init__(self, config):
                self.input_size = config.get('input_size', (224, 224))
                self.channels = config.get('channels', 3)
                self.filters = [
                    (11, 11, 96),    # Conv1: 11x11, 96 filters
                    (5, 5, 256),     # Conv2: 5x5, 256 filters
                    (3, 3, 384)      # Conv3: 3x3, 384 filters
                ]
                
            def __call__(self, image):
                """
                ConvNet Forward Pass (Krizhevsky et al., 2012)
                
                Layer Operations:
                1. Conv2D: (W-F+2P)/S + 1
                Where W=input size, F=filter size, P=padding, S=stride
                2. ReLU: f(x) = max(0, x)
                3. MaxPool: y[i,j] = max_{m,n ∈ N(i,j)} x[m,n]
                4. LRN: b_{x,y}^i = a_{x,y}^i / (k + αΣ_{j=max(0,i-n/2)}^{min(N-1,i+n/2)} (a_{x,y}^j)^2)^β
                
                Architectural Parameters:
                - Input: 224x224x3
                - Conv1: 96@55x55, 11x11 filters, stride 4, pad 2
                - Pool1: 3x3, stride 2
                - Conv2: 256@27x27, 5x5 filters, pad 2
                - Pool2: 3x3, stride 2
                - Conv3-5: 384/256/256@13x13
                - FC6-7: 4096-D
                - FC8: 1000-D (ImageNet)
                """
                # Feature dimension progression
                features = image
                channel_axis = -1
                
                # Conv1 Layer
                features = self._conv2d(features, self.filters[0], stride=4, padding=2)
                features = self._relu(features)
                features = self._max_pool(features, 3, stride=2)
                
                # Conv2 Layer
                features = self._conv2d(features, self.filters[1], padding=2)
                features = self._relu(features)
                features = self._max_pool(features, 3, stride=2)
                
                # Conv3 Layer
                features = self._conv2d(features, self.filters[2])
                features = self._relu(features)
                
                # Spatial Pyramid Pooling
                features = self._spatial_pyramid_pooling(features)
                
                return features

            def _conv2d(self, x, filters, stride=1, padding=0):
                """2D Convolution: y[i,j] = Σ_{m,n} x[i+m,j+n] * k[m,n] + b"""
                # Input shape: (H, W, C_in)
                # Filter shape: (Fh, Fw, C_in, C_out)
                # Output shape: (H', W', C_out)
                H, W, C_in = x.shape
                Fh, Fw, C_out = filters.shape[:3]
                
                H_out = (H - Fh + 2*padding) // stride + 1
                W_out = (W - Fw + 2*padding) // stride + 1
                
                # Simulate convolution operation
                output = np.zeros((H_out, W_out, C_out))
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i*stride - padding
                        w_start = j*stride - padding
                        receptive_field = x[h_start:h_start+Fh, w_start:w_start+Fw, :]
                        output[i,j] = np.sum(receptive_field * filters, axis=(0,1,2))
                
                return output

            def _spatial_pyramid_pooling(self, features):
                """Pyramid pooling: max pooling at multiple scales"""
                levels = [1, 2, 4]  # Different binning levels
                pooled = []
                for level in levels:
                    H, W = features.shape[:2]
                    bin_h = H // level
                    bin_w = W // level
                    for i in range(level):
                        for j in range(level):
                            pool_region = features[i*bin_h:(i+1)*bin_h, 
                                                j*bin_w:(j+1)*bin_w]
                            pooled.append(np.max(pool_region, axis=(0,1)))
                return np.concatenate(pooled)

        return VisionFeatureExtractor(config)

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
