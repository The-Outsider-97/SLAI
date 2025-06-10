__version__ = "1.8.0"


import threading
import importlib
import torch
import inspect 

from contextlib import contextmanager
# from restricted_env import RestrictedPython
from collections import defaultdict, deque
from typing import Any, Dict, Optional, Tuple, List, Type

from models.reasoner import BasicZeroReasoner

from profiling_utils import memory_profile, time_profile, start_memory_tracing, display_top_memory_sources
#from src.utils.system_optimizer import SystemOptimizer
#from src.agents.adaptive_agent import AdaptiveAgent
from src.agents.alignment_agent import AlignmentAgent
#from src.agents.evaluation_agent import EvaluationAgent
#from src.agents.execution_agent import ExecutionAgent
#from src.agents.knowledge_agent import KnowledgeAgent
# from src.agents.language_agent import LanguageAgent
#from src.agents.learning_agent import LearningAgent
from src.agents.learning.slaienv import SLAIEnv
#from src.agents.perception_agent import PerceptionAgent
#from src.agents.planning_agent import PlanningAgent
#from src.agents.reasoning_agent import ReasoningAgent
from src.agents.safety_agent import SafetyAgent
from src.agents.factory.agent_meta_data import AgentMetaData, load_config
from src.agents.factory.metrics_adapter import MetricsAdapter
# from models.slai_lm import get_shared_slailm
from logs.logger import get_logger

logger = get_logger("Agent Factory")

class AgentFactory:
    def __init__(self, config, shared_resources):
        self.config = config
        self.shared_resources = {
            **shared_resources,
            "shared_memory": shared_resources.get("shared_memory"),
            "agent_factory": self
        }
        self.registry = {}
        self.meta_registry = {}
        self.instance_cache = {}
        self.metrics_adapter = MetricsAdapter()
        self.bzr = BasicZeroReasoner()
        
        self._register_core_agents(config.get('agent-network', {}))
        self._validate_agent_metadata()
        self.lock = threading.RLock()

    def _register_core_agents(self, agent_network: Dict):
        """Register agents with metadata validation"""
        for agent_name, agent_config in agent_network.items():
            # Create AgentMetaData instance
            meta = AgentMetaData(
                name=agent_name,
                class_name=agent_config['class'],
                module_path=agent_config['path'],
                required_params=tuple(agent_config.get('required_params', [])),
                config=agent_config.get('init_args', {})
            )

            # Store in meta registry
            self.meta_registry[agent_name] = meta

            # Original registry remains for runtime info
            self.registry[agent_name] = {
                'meta': meta,
                'dependencies': self._resolve_dependencies(agent_config),
                'instance': None
            }

    
    def validate_with_azr(self, triple):
        result = self.bzr.check_contradiction(triple)
        return result.get("contradiction_score", 0.0)

    def _validate_agent_metadata(self):
        """Validate all registered agent metadata"""
        for agent_name, meta in self.meta_registry.items():
            try:
                meta._validate(load_config().get("agent_meta", {}))
            except ValueError as e:
                logger.error(f"Invalid metadata for {agent_name}: {str(e)}")
                raise

    def _initialize_shared_resources(self):
        shared_resources_config = self.config.get('shared_resources', {})
        for name, resource_config in shared_resources_config.items():
            module_path = resource_config.get('path')
            class_name = resource_config.get('class')
            init_args = resource_config.get('init_args', {})
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            resource = cls(**init_args)
            self.shared_resources[name] = resource
            if name == 'tokenizer': #Add this
                self.shared_resources['tokenizer'] = resource #Add this
            if name == 'text_encoder': #Add this
                self.shared_resources['text_encoder'] = resource #Add this

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
        
    def set(self):
        pass
    
    @contextmanager
    def _get_agent_context(self, agent_name):
        """Thread-safe context manager for agent access"""
        with self.lock:
            if agent_name not in self.instance_cache:
                self._initialize_agent(agent_name)
            yield self.instance_cache[agent_name]
    
    def _initialize_agent(self, agent_name):
        

        agent_info = self.registry[agent_name]
        cls = self._import_class(f"{agent_info['path']}.{agent_info['class']}")
        dependencies = {dep: self.instance_cache[dep] for dep in agent_info['dependencies']}
        config = agent_info.get('config', {})
        init_params = inspect.signature(cls.__init__).parameters

        general_args = {}
        if 'shared_memory' in init_params:
            general_args['shared_memory'] = self.shared_resources['shared_memory']
        if 'agent_factory' in init_params:
            general_args['agent_factory'] = self
        if 'tokenizer' in init_params:
             general_args['tokenizer'] = self.shared_tokenizer
        if 'text_encoder' in init_params:
             general_args['text_encoder'] = self.shared_text_encoder
    
        elif agent_name == "language":
            shared_slailm_instance = get_shared_slailm(
                self.shared_resources['shared_memory'],
                shared_tokenizer=self.shared_tokenizer,
                shared_text_encoder=self.shared_text_encoder,
                agent_factory=self
            )
            # LanguageAgent initializes its own DialogueContext and GrammarProcessor internally
            agent_instance = cls(
                config=config,
                **general_args,
                **dependencies
            )
        elif agent_name == "knowledge":
            language_agent = self.get('language')
            agent_instance = cls(
                language_agent=language_agent,
                knowledge_agent_dir="data/knowledge_base",
                persist_file="data/knowledge_cache.json",
                text_encoder=self.shared_text_encoder,
                tokenizer=self.shared_tokenizer,
                config=config,
                **general_args,
                **dependencies
            )
        elif agent_name == "safety":
            safety_config = config(**config)
            agent_instance = cls(
                alignment_agent_cls=AlignmentAgent,
                config=safety_config,
                **general_args,
                **dependencies
            )
        elif agent_name == "reasoning":
            language_agent = self.get('language')
            shared_slailm_instance = get_shared_slailm(
                self.shared_resources['shared_memory'],
                shared_tokenizer=self.shared_tokenizer,
                shared_text_encoder=self.shared_text_encoder,
                agent_factory=self
            )
            agent_instance = cls(
                language_agent=language_agent,
                llm=shared_slailm_instance,
                config=config,
                **general_args,
                **dependencies
            )
        elif agent_name == "perception":
            audio_encoder = self._init_audio_encoder(config)
            agent_instance = cls(
                audio_encoder=audio_encoder,
                text_encoder=self.shared_text_encoder,
                tokenizer=self.shared_tokenizer,
                config=config,
                shared_memory=self.shared_resources['shared_memory'],
                agent_factory=self,
                **dependencies
            )
        elif agent_name == "planning":
            agent_instance = cls(
                text_encoder=self.shared_text_encoder,
                tokenizer=self.shared_tokenizer,
                config=config,
                **general_args,
                **dependencies
            )
        elif agent_name == "execution":
            agent_instance = cls(
                text_encoder=self.shared_text_encoder,
                tokenizer=self.shared_tokenizer,
                config=config,
                **general_args,
                **dependencies
            )
        elif agent_name == "browser":
            agent_instance = cls(
                text_encoder=self.shared_text_encoder,
                tokenizer=self.shared_tokenizer,
                config=config,
                **general_args,
                **dependencies
            )
        elif agent_name == "adaptive":
            agent_instance = cls(
                text_encoder=self.shared_text_encoder,
                tokenizer=self.shared_tokenizer,
                config=config,
                **general_args,
                **dependencies
            )
        elif agent_name == "alignment":
            from src.agents.alignment.alignment_monitor import MonitorConfig
            safe_agent_instance = self.get('safety')
            monitor_conf = MonitorConfig(**config.get('monitor', {}))
            correction_pol = CorrectionPolicy(**config.get('corrections', {}))
            
            agent_instance = cls(
                text_encoder=self.shared_text_encoder,
                tokenizer=self.shared_tokenizer,
                config=config,
                monitor_config=monitor_conf,
                correction_policy=correction_pol,
                safe_agent=safe_agent_instance,
                **general_args,
                **dependencies
            )
        elif agent_name == "evaluation":
            agent_instance = cls(
                text_encoder=self.shared_text_encoder,
                tokenizer=self.shared_tokenizer,
                config=config,
                **general_args,
                **dependencies
            )
        elif agent_name == "learning":
            slai_lm = self.get('language').slai_lm if self.get('language') else None
            safety_agent = self.get('safety')
            env = self._init_environment(config.get('env_name', 'CartPole-v1'))
            agent_instance = cls(
                shared_memory=self.shared_resources['shared_memory'],
                agent_factory=self,
                slai_lm=slai_lm,
                safety_agent=safety_agent,
                env=env,
                text_encoder=self.shared_text_encoder,
                tokenizer=self.shared_tokenizer,
                config=config,
                **dependencies
            )
        else:
            agent_instance = cls(
                config=config,
                **general_args,
                **dependencies
            )
    
        self.instance_cache[agent_name] = agent_instance

    def _init_environment(self, env_name: str):
        import gymnasium as gym
        return gym.make(env_name)

    def create(self, agent_name: str, config: Dict = None):
        agent_info = self.registry[agent_name]
        cls = self._import_class(f"{agent_info['path']}.{agent_info['class']}")
        
        # Get base parameters from factory
        base_params = {
            'shared_memory': self.shared_resources.get('shared_memory'),
            'agent_factory': self
        }
        
        # Merge configurations
        init_args = agent_info.get('init_args', {}).copy()
        if config and 'init_args' in config:
            init_args.update(config['init_args'])
        
        # Get valid constructor parameters
        init_params = inspect.signature(cls.__init__).parameters
        
        # Automatically inject required base parameters
        final_args = {}
        for param in init_params:
            if param == 'config':
                final_args['config'] = config or {}
            elif param == 'audio_encoder' and agent_name == "perception":
                final_args['audio_encoder'] = self._init_audio_encoder(config or {})
            elif param in base_params:
                final_args[param] = base_params[param]
            elif param in init_args:
                final_args[param] = init_args[param]
        
        # Preserve explicit config parameters
        if config:
            final_args.update({k: v for k, v in config.items() if k != 'init_args'})
        
        return cls(**final_args)

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

    def report_memory_usage(self, limit: int = 10):
        """Display top memory-consuming components"""
        display_top_memory_sources(limit)
