"""
Agent Factory for Dynamic Agent Creation

This implementation provides a robust factory pattern for creating all agent types
in the system with:
- Comprehensive error handling
- Lazy imports for better performance
- Configuration validation
- Automatic agent discovery
- Support for all agent types except BaseAgent

Academic References:
- Gamma et al. (1994) "Design Patterns: Factory Method"
- Fowler (2002) "Patterns of Enterprise Application Architecture"
"""

import importlib
import logging
from pathlib import Path
from typing import Dict, Any, Type, Optional

logger = logging.getLogger("SLAI.Factory")
logger.setLevel(logging.INFO)

class AgentMetaData:
    """Container for dynamically discovered agent metadata"""
    def __init__(self, name: str, module_path: str, class_name: str,
                 description: str, required_params: list[str]):
        self.name = name
        self.module_path = module_path
        self.class_name = class_name
        self.description = description
        self.required_params = required_params

class AgentFactory:
    def __init__(self, shared_resources: Optional[Dict] = None):
        self.shared_resources = shared_resources or {}
        self.agent_registry: Dict[str, AgentMetaData] = {}
        self._discover_agents()

    def _discover_agents(self):
        """Dynamic agent discovery with AST parsing"""
        base_dir = Path(__file__).parent.parent / "src" / "agents"
        
        # Parse top-level agents
        for agent_file in base_dir.glob("*_agent.py"):
            if "learning" in agent_file.parts:
                continue  # Skip learning subsystem
            
            metadata = self._parse_agent_file(agent_file)
            if metadata:
                self.agent_registry[metadata.name] = metadata

    def _parse_agent_file(self, file_path: Path) -> Optional[AgentMetaData]:
        """Extract metadata using AST parsing"""
        try:
            with open(file_path, "r") as f:
                tree = ast.parse(f.read())
            
            class_def = next(
                (n for n in tree.body if isinstance(
                    n, ast.ClassDef
                    )),
                    None
                    )
            
            if not class_def:
                return None

            docstring = ast.get_docstring(class_def) or "No description"
            init_method = next(
                (node for node in class_def.body 
                 if isinstance(node, ast.FunctionDef) and node.name == "__init__"),
                None
            )
            
            required_params = []
            if init_method and init_method.args.args:
                # Skip 'self' parameter
                params = init_method.args.args[1:]
                required_params = [arg.arg for arg in params 
                                 if not (arg.annotation and 
                                         ast.unparse(arg.annotation) == "Optional")]
            
            return AgentMetaData(
                name=file_path.stem.replace("_agent", ""),
                module_path=f"agents.{file_path.stem}",
                class_name=class_def.name,
                description=ast.get_docstring(class_def) or "No description",
                required_params=self._detect_required_params(class_def)
            )
        
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            return None

    def _special_init(self, agent_class: Type, config: Dict) -> Any:
        """Handles agent-specific initialization with academic grounding"""
        agent_name = agent_class.__name__.lower()
        
        # --- Perception Agent ---
        if "perception" in agent_name:
            # Initialize multimodal processing pipeline
            if "sensors" not in config:
                config["sensors"] = self._default_sensors()
                
            # Set up model ensemble (CLIP + Perceiver-like architecture)
            config["vision_encoder"] = self._init_vision_encoder(
                model_type=config.get("vision_model", "clip")
            )
            config["audio_encoder"] = self._init_audio_encoder(
                model_type=config.get("audio_model", "wav2vec2")
            )
            
            return agent_class(**config)
        
        # --- Knowledge Agent ---
        elif "knowledge" in agent_name:
            # Initialize RAG components
            config["retriever"] = self._init_retriever(
                index_path=self.shared_resources.get("knowledge_index"),
                embedding_model=config.get("embedding_model", "all-mpnet-base-v2")
            )
            
            # Add memory components (Knowledge Graph + Vector Store)
            config["graph_db"] = self.shared_resources.get("graph_db") or Neo4jInterface()
            config["vector_store"] = self.shared_resources.get("vector_store") or FAISSIndex()
            
            return agent_class(**config)
        
        # --- Planning Agent ---
        elif "planning" in agent_name:
            # HTN Planner configuration
            config["domain_knowledge"] = self.shared_resources.get("domain_ontology")
            config["planner_type"] = config.get("planner_type", "hierarchical")
            
            # ReAct-style planning setup
            if "reasoning_model" not in config:
                config["reasoning_model"] = self._load_default_reasoner()
                
            return agent_class(**config)
        
        # --- Reasoning Agent ---
        elif "reasoning" in agent_name:
            # Chain-of-Thought configuration
            config["cot_prompt_template"] = config.get(
                "cot_prompt",
                "Let's think step by step: {question}\nFirst,"
            )
            
            # Symbolic reasoning tools
            config["math_engine"] = SymPySolver()
            config["logic_prover"] = Prover9Interface()
            
            return agent_class(**config)
        
        # --- Execution Agent ---
        elif "execution" in agent_name:
            # Tool library initialization
            config["tool_library"] = self._init_tool_library(
                allow_list=config.get("allowed_tools"),
                sandbox=config.get("sandbox", True)
            )
            
            # Action validation setup
            config["validator"] = ActionValidator(
                safety_policy=self.shared_resources.get("safety_policy")
            )
            
            return agent_class(**config)
        
        # --- Language Agent ---
        elif "language" in agent_name:
            # Unified text-to-text model (T5-style)
            config["text_model"] = self._init_text_model(
                model_name=config.get("model_name", "t5-large"),
                device=config.get("device", "auto")
            )
            
            # Dialogue management
            config["conversation_memory"] = self.shared_resources.get(
                "conversation_store",
                RollingBuffer(max_turns=10)
            )
            
            return agent_class(**config)
        
        # --- Learning Agent ---
        elif "learning" in agent_name:
            # RL and Meta-learning setup
            config["rl_algorithm"] = self._init_rl_algorithm(
                algorithm=config.get("algorithm", "ppo"),
                policy_network=config.get("policy_network")
            )
            
            # Experience replay buffer
            config["replay_buffer"] = PrioritizedReplayBuffer(
                capacity=config.get("buffer_size", 10000),
                alpha=config.get("priority_alpha", 0.6)
            )
            
            return agent_class(**config)
        
        # --- Adaptation Agent ---
        elif "adaptation" in agent_name:
            # Continual learning setup
            config["ewc_lambda"] = config.get("ewc_lambda", 1000)  # Elastic Weight Consolidation
            config["replay_ratio"] = config.get("replay_ratio", 0.3)
            
            # Performance monitoring
            config["metrics_tracker"] = MultiMetricTracker(
                primary_metric=config.get("primary_metric", "success_rate")
            )
            
            return agent_class(**config)
        
        # --- Collaboration Agent ---
        elif "collaborative" in agent_name:
            # Blackboard system setup
            config["shared_memory"] = self.shared_resources.get(
                "shared_memory",
                BlackboardSystem()
            )
            
            # Message router initialization
            config["message_router"] = PubSubRouter(
                channels=list(self.agent_registry.keys())
            )
            
            return agent_class(**config)
        
        # --- Evaluation Agent ---
        elif "evaluation" in agent_name:
            # Automated evaluation models
            config["quality_predictor"] = self._init_evaluator_model(
                model_path=config.get("eval_model")
            )
            
            # A/B testing framework
            config["experiment_manager"] = ExperimentManager(
                stratify_by=config.get("stratify_metrics", ["task_type"])
            )
            
            return agent_class(**config)
        
        # --- Safety Agent ---
        elif "safety" in agent_name:
            # Constitutional AI setup
            config["harmlessness_rules"] = self._load_constitutional_rules(
                rule_set=config.get("rule_set", "default")
            )
            
            # Interpretability tools
            config["attention_visualizer"] = AttentionMapper()
            config["concept_activator"] = TCAVInterface()
            
            return agent_class(**config)
        
        # --- Default Case ---
        return agent_class(**config)

    # Helper initialization methods
    def _init_vision_encoder(self, model_type: str):
        """Initialize visual perception models"""
        if model_type == "clip":
            return CLIPWrapper(
                vision_encoder="ViT-B/32",
                text_encoder="text-embedding-ada-002"
            )
        elif model_type == "vit":
            return VisionTransformerWrapper(
                model_name="vit-large-patch16-224"
            )
        else:
            raise ValueError(f"Unknown vision model: {model_type}")

    def _init_retriever(self, index_path: str, embedding_model: str):
        """Initialize RAG components"""
        return HybridRetriever(
            dense_retriever=DPRWrapper(model_name=embedding_model),
            sparse_retriever=BM25Retriever(index_path=index_path)
        )

    def _init_tool_library(self, allow_list: list[str], sandbox: bool):
        """Initialize execution tools with safety constraints"""
        return ToolLibrary(
            allowed_tools=allow_list or DEFAULT_ALLOWED_TOOLS,
            sandbox_mode=sandbox,
            timeout=30.0  # seconds
        )

    def _load_constitutional_rules(self, rule_set: str):
        """Load alignment rules for Safety Agent"""
        rules = {
            "default": [
                "Don't provide harmful or dangerous content",
                "Don't reveal private information",
                "Maintain helpful and honest behavior"
            ],
            "strict": [
                # More comprehensive rule set
            ]
        }
        return rules.get(rule_set, rules["default"])

    def _default_sensors(self):
        """Default sensor configuration for perception agent"""
        return ["vision", "audio", "tactile"]

    def create(self, agent_name: str, config: Dict) -> Any:
        """Create agent with dynamic validation"""
        if agent_name not in self.agent_registry:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        metadata = self.agent_registry[agent_name]
        self._validate_config(metadata, config)
        
        try:
            module = importlib.import_module(metadata.module_path)
            agent_class = getattr(module, metadata.class_name)
            return self._special_init(agent_class, config)
        except Exception as e:
            logger.error(f"Agent creation failed: {str(e)}")
            raise

    def _validate_config(self, metadata: AgentMetaData, config: Dict):
        """Dynamic parameter validation"""
        missing = [param for param in metadata.required_params 
                   if param not in config]
        if missing:
            raise ValueError(
                f"Missing required parameters for {metadata.name}: {missing}"
            )

class PerceptionAgent:
    """Handles sensory input processing including:
    - Visual data
    - Audio signals
    - Tactile feedback
    """

# Example Usage
# if __name__ == "__main__":
#     factory = AgentFactory(shared_resources={
#         "shared_memory": {},
#         "task_router": {}
#     })
    
#     print("Discovered Agents:")
#     for name, meta in factory.agent_registry.items():
#         print(f"{name}: {meta.description}")
    
#     try:
#         collab_agent = factory.create("collaborative", {})
#         perception_agent = factory.create("perception", {})
#     except Exception as e:
#         print(f"Error: {str(e)}")
