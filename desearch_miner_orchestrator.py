import time
import random
import os, re
import hashlib
import json, yaml
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Dict, List
from collections import deque
from datetime import datetime
from threading import Thread
from flask import Flask, jsonify, send_from_directory

from src.agents.learning.slaienv import SLAIEnv
from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.collaborative.shared_memory import SharedMemory
from src.agents.safety_agent import SafetyAgent, SafetyAgentConfig # Import SafetyAgentConfig
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.agents.adaptive_agent import AdaptiveAgent
from src.agents.learning_agent import LearningAgent
from logs.logger import get_logger

logger = get_logger("Desearch Miner Orchestrator")

KNOWLEDGE_STORE_PATH = "data/knowledge_store.json"
KNOWLEDGE_BASE_PATH = "data/knowledge_base"

# Configuration paths
CONFIG_PATHS = {
    "collaborative": "src/agents/collaborative/configs/collaborative_config.yaml",
    "reasoning": "src/agents/reasoning/configs/reasoning_config.yaml",
    "adaptive": "src/agents/adaptive/configs/adaptive_config.yaml",
    "learning": "src/agents/learning/configs/learning_config.yaml",
    "safety": "src/agents/safety/configs/secure_config.yaml",
    "evaluation": "src/agents/evaluation/configs/evaluator_config.yaml",
    "knowledge": "src/agents/knowledge/configs/knowledge_config.yaml"
}

def load_config(config_path):
    """Load configuration from YAML file with error handling"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f: # Added encoding
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {str(e)}")
        return {}

# --- Functions ---
def fetch_data(batch_size=5) -> List[str]:
    """
    Simulates fetching new documents.
    In a real system, this would connect to data sources (e.g., web crawlers, APIs, databases).
    """
    logger.info(f"Fetching {batch_size} new documents (simulated)...")
#    docs = []

#    for i in range(batch_size):
#        doc_type = random.choice(["news_article", "research_paper", "blog_post", "forum_discussion"])
#        safety_keyword = random.choice(["safe", "secure", "vulnerability", "exploit", "privacy", "threat", "benign", "helpful"])
#        tech_keyword = random.choice(["blockchain", "AI", "machine learning", "cybersecurity", "cloud computing", "quantum", "robotics"])
#        action_keyword = random.choice(["discusses", "analyzes", "reviews", "questions", "explores"])
#        content_length = random.randint(50, 200) # Shorter for easier debugging
#        doc_content = f"Document {i+1} ({doc_type}): This document {action_keyword} {tech_keyword} and its relation to {safety_keyword}. "
#        doc_content += " ".join([f"word{j}" for j in range(content_length)])
#        if random.random() < 0.3:
#            doc_content += f" More info at http://example-domain-{random.randint(1,100)}.com/path{i} or contact researcher{i}@example-domain.org."
#        if random.random() < 0.1: 
#            # Add PII-like content sometimes
#            doc_content += f" User SSN is {random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)} and password for access is 'test{random.randint(100,999)}Key'."
#        docs.append(doc_content)
#    return docs

    tasks = []
    miner_instance = Miner(config_path="config.yaml")
    for _ in range(batch_size):
        task = miner_instance.get_task()
        if task and "text" in task:
            tasks.append(task["text"])
    return tasks
def get_recent_rewards(processed_docs_info: list) -> List[float]:
    """
    Generates reward signals based on the evaluation and safety scores of recently processed documents.
    Args:
    processed_docs_info: A list of dictionaries, where each dictionary contains
    scores for a successfully submitted document.
    Expected keys: 'eval_score', 'safety_score', 'cyber_risk'.
    Returns:
    A list of float reward values.
    """
    rewards = []
    if not processed_docs_info:
        logger.debug("No processed documents to generate rewards for.") # Changed to debug
        return rewards
    logger.debug(f"Generating rewards for {len(processed_docs_info)} recently processed documents...")
    for doc_info in processed_docs_info:
        if not isinstance(doc_info, dict):
            logger.warning(f"Skipping invalid doc_info for reward calculation: {doc_info}")
            continue
        eval_score = doc_info.get('eval_score', 0.0)
        safety_score = doc_info.get('safety_score', 0.0)
        cyber_security_contribution = 1.0 - doc_info.get('cyber_risk', 1.0) # Default cyber_risk to 1.0 (max risk)

        combined_reward = (0.5 * eval_score) + \
                        (0.3 * safety_score) + \
                        (0.2 * cyber_security_contribution)
        
        final_reward = max(0.0, min(1.0, combined_reward)) 
        rewards.append(final_reward)
        logger.debug(f"Doc ID {doc_info.get('doc_id', 'N/A')}: Eval={eval_score:.2f}, Safety={safety_score:.2f}, CyberContr={cyber_security_contribution:.2f} -> Reward={final_reward:.2f}")
    return rewards

def submit_data(doc_content: str, submission_metadata: dict):
    """
    Submits an approved document to the KnowledgeAgent and logs the submission.
    Args:
    doc_content: The content of the document.
    submission_metadata: A dictionary containing metadata about the submission,
    including 'doc_id', 'eval_score', 'safety_score', etc.
    """
    doc_id = submission_metadata.get('doc_id', 'N/A')
    logger.info(f"Attempting to submit document ID {doc_id}: {doc_content[:70]}...")
    try:
        knowledge_agent.add_document(
            text=doc_content,
            doc_id=doc_id, 
            metadata={
                "source": "desearch_miner_orchestrator",
                "submission_timestamp": submission_metadata.get("timestamp"),
                "evaluation_score": submission_metadata.get("eval_score"),
                "safety_score": submission_metadata.get("safety_score"),
                "status": "approved_and_submitted_to_kb" # More specific status
            }
        )
        # Log submission to collaborative shared_memory
        recent_submissions_log = shared_memory.get("recent_submissions_log") or deque(maxlen=200)
        if not isinstance(shared_memory.get("recent_submissions_log"), (list, deque)):
            shared_memory["recent_submissions_log"] = deque(maxlen=200)
        submission_metadata_copy = submission_metadata.copy() # Avoid modifying original dict
        submission_metadata_copy["submission_to_orchestrator_log_timestamp"] = datetime.utcnow().isoformat()
        recent_submissions_log.append(submission_metadata_copy)
        shared_memory.set("recent_submissions_log", recent_submissions_log)

        logger.info(f"Document ID {doc_id} successfully submitted and added to knowledge base.")
    except Exception as e:
        logger.error(f"Error during document submission to KnowledgeAgent (ID {doc_id}): {e}", exc_info=True)
        raise

def extract_embedding(text: str) -> np.ndarray:
    """Generates embedding for text using KnowledgeAgent's capabilities."""
    if not isinstance(text, str):
        text_to_embed = str(text)
        logger.warning(f"extract_embedding: Input was not a string (type: {type(text)}), converted to string: '{text_to_embed[:50]}...'.")
    else:
        text_to_embed = text
    if not text_to_embed.strip():
        logger.warning("extract_embedding: Input text is empty or whitespace-only.")
        # Return zero vector matching expected dimension
        embedding_dim = configs.get("learning", {}).get("embedding_dim", 512) # Default from DummyEnv
        if knowledge_agent.embedding_fallback: # Try to get actual dimension
            try:
                embedding_dim = knowledge_agent.embedding_fallback.get_sentence_embedding_dimension()
            except AttributeError: # If model doesn't have this method
                logger.debug("Could not get SBERT dimension, using config/default for zero vector.")
        logger.warning(f"Returning zero vector of dimension {embedding_dim} for empty input.")
        return np.zeros(embedding_dim)

    if knowledge_agent.embedding_fallback:
        try:
            embedding_array = knowledge_agent.embedding_fallback.encode(text_to_embed)
            return embedding_array
        except Exception as e:
            logger.error(f"Error generating embedding with knowledge_agent.embedding_fallback: {e}", exc_info=True)
    else:
        logger.warning("KnowledgeAgent embedding_fallback not available.")

    # Fallback to a random vector if primary embedding fails or isn't available
    embedding_dim = configs.get("learning", {}).get("embedding_dim", 512)
    logger.warning(f"Falling back to random embedding of dimension {embedding_dim}.")
    return np.random.rand(embedding_dim)
# --- End Functions ---


# Load all configurations
configs = {name: load_config(path) for name, path in CONFIG_PATHS.items()}

shared_memory = SharedMemory()
agent_factory_placeholder = lambda name, cfg: None


safety_agent_global_config = configs.get("safety", {})
# SafetyAgent's own config is expected under a 'safety_agent' key within that, or passed directly
safety_agent_specific_config_data = safety_agent_global_config.get("safety_agent", {})

# Ensure essential keys for SafetyAgentConfig are present with defaults
default_sa_params = {
    "constitutional_rules_path": "src/agents/safety/templates/constitutional_rules.json",
    "risk_thresholds": {"overall_safety": 0.7, "cyber_risk": 0.6, "compliance_failure_is_blocker": False},
    "audit_level": 1,
    "collect_feedback": safety_agent_global_config.get("collect_feedback", False), # Get from global or default
    "enable_learnable_aggregation": safety_agent_global_config.get("enable_learnable_aggregation", False),
    "secure_memory": safety_agent_global_config.get("secure_memory", {}) # Pass secure_memory section from global
}
final_safety_agent_params = {**default_sa_params, **safety_agent_specific_config_data}
safety_agent_config_obj = SafetyAgentConfig(**final_safety_agent_params)

safety_agent = SafetyAgent(
    shared_memory=shared_memory,
    agent_factory=agent_factory_placeholder,
    config=safety_agent_config_obj
)

reasoning_agent = ReasoningAgent(
    shared_memory=shared_memory,
    agent_factory=agent_factory_placeholder,
    config=configs.get("reasoning", {})
)

evaluation_agent = EvaluationAgent(
    shared_memory=shared_memory,
    agent_factory=agent_factory_placeholder,
    config=configs.get("evaluation", {})
)

adaptive_agent = AdaptiveAgent(
    shared_memory=shared_memory,
    agent_factory=agent_factory_placeholder,
    config=configs.get("adaptive", {})
)

# SLAIEnv setup
learning_config = configs.get("learning", {})
slai_env_state_dim = learning_config.get("embedding_dim", 512) # Match LearningAgent's expectation
slai_env_action_dim = 2 # Assuming binary classification for learning_agent.observe
slaienv = SLAIEnv(
    SLAILM=None, 
    agent_factory=agent_factory_placeholder,
    shared_memory=shared_memory,
    state_dim=slai_env_state_dim,
    action_dim=slai_env_action_dim
)

learning_agent = LearningAgent(
    shared_memory=shared_memory,
    agent_factory=agent_factory_placeholder,
    config=configs.get("learning", {}),
    env=slaienv  # Use SLAIEnv here
)

# Initialize KnowledgeAgent with its config
knowledge_config = configs.get("knowledge", {})
knowledge_memory_config = knowledge_config.get("knowledge_memory", {})
persist_file_path = knowledge_memory_config.get("persist_file", "data/knowledge_store.json") # Default

knowledge_agent = KnowledgeAgent(
    shared_memory=shared_memory,
    agent_factory=agent_factory_placeholder,
    config=knowledge_config,
    persist_file=persist_file_path
)


# Load knowledge base on startup
def initialize_knowledge_base():
    """Load knowledge from directory and database"""
    try:
        # Initialize semantic fallback and stopwords for KnowledgeAgent
        # This requires a LanguageAgent instance. If not available, KA will use defaults.
        lang_agent_for_ka = agent_factory_placeholder("LanguageAgent", configs.get("language", {})) # Or however you get it
        knowledge_agent._initialize_semantic_fallback(lang_agent_for_ka) # Call this early
        knowledge_agent.initialize_stopwords() # Call explicitly if not done in _initialize_semantic_fallback

        knowledge_dir = knowledge_config.get("knowledge_dir", "data/knowledge_base")
        if os.path.exists(knowledge_dir) and os.path.isdir(knowledge_dir): # Check if directory exists AND is a directory
            knowledge_agent.load_from_directory(knowledge_dir)
            logger.info(f"Loaded knowledge from directory: {knowledge_dir}")
        else:
            logger.warning(f"Knowledge directory not found or not a directory: {knowledge_dir}")
        
        if persist_file_path and os.path.exists(persist_file_path) and os.path.isfile(persist_file_path): # Check if file exists AND is a file
            knowledge_agent.load_knowledge_db(persist_file_path)
            logger.info(f"Loaded knowledge database: {persist_file_path}")
        elif persist_file_path:
            logger.warning(f"Knowledge DB file not found: {persist_file_path}")
        else:
            logger.warning("No persist_file path configured for knowledge DB.")

        logger.info("Knowledge base initialization attempted.")
    except Exception as e:
        logger.error(f"Knowledge base initialization failed: {str(e)}", exc_info=True)

initialize_knowledge_base()


# Initialize on import
initialize_knowledge_base()

def process_document(doc_content: str) -> Optional[Dict]:
    """Enhanced document pipeline with knowledge integration.
    Returns a dictionary with processing results if successful and submitted, else None.
    """
    doc_hash = hashlib.sha256(doc_content.encode('utf-8', 'ignore')).hexdigest() # Ensure encoding
    logger.info(f"Processing document ID {doc_hash}: {doc_content[:70]}...")
    context = {
        "timestamp": datetime.utcnow().isoformat(),
        "source": "miner_loop",
        "doc_id": doc_hash
    }
    current_safety_assessment = None
    current_eval_result = None
    reasoning_output_for_eval = {}

    # 1. Safety Check
    try:
        current_safety_assessment = safety_agent.perform_task(doc_content, context=context)
        if not (isinstance(current_safety_assessment, dict) and current_safety_assessment.get("is_safe")):
            reason = current_safety_assessment.get("overall_recommendation", "Unknown safety reason") if isinstance(current_safety_assessment, dict) else "Safety check failed"
            logger.warning(f"[BLOCKED ID {doc_hash}] Document failed safety check: {reason}")
            shared_memory.set(f"processing_status:{doc_hash}", {"status": "rejected_safety", "reason": reason})
            # Increment safety block counter
            safety_blocked_count = shared_memory.get("safety_blocked", 0) or 0
            shared_memory.set("safety_blocked", safety_blocked_count + 1)
            shared_memory.set("last_block_reason", f"Safety ({doc_hash}): {reason}")
            return None
    except Exception as e:
        logger.error(f"Error during safety check for ID {doc_hash}: {e}", exc_info=True)
        shared_memory.set(f"processing_status:{doc_hash}", {"status": "error_safety", "error": str(e)})
        return None


    # 2. Knowledge Retrieval & Context Enrichment
    try:
        key_terms_match = re.findall(r'\b\w{4,15}\b', doc_content) # Get 4-15 char words
        key_terms = " ".join(key_terms_match[:7]) # Use first 7 such words
        
        if key_terms:
            knowledge_results = knowledge_agent.retrieve(key_terms, k=3)
            context["retrieved_knowledge"] = [res_doc['text'] for _, res_doc in knowledge_results if isinstance(res_doc, dict) and 'text' in res_doc] if knowledge_results else []
            logger.debug(f"Retrieved {len(context.get('retrieved_knowledge',[]))} knowledge snippets for '{key_terms}' (ID {doc_hash})")
        else:
            context["retrieved_knowledge"] = []
            logger.debug(f"No key terms found in document ID {doc_hash} for knowledge retrieval.")
    except Exception as e:
        logger.error(f"Knowledge retrieval failed for ID {doc_hash}: {str(e)}", exc_info=True)
        context["retrieved_knowledge"] = [] # Ensure it's initialized for next steps
    
    # 3. Knowledge-Enhanced Reasoning
    reasoning_result = None
    try:
        reasoning_input = {
            "document": doc_content,
            "knowledge": context.get("retrieved_knowledge", []),
            "context": context 
        }
        reasoning_output_for_eval = reasoning_agent.perform_task(reasoning_input) 
        
        if isinstance(reasoning_output_for_eval, dict) and reasoning_output_for_eval.get("contradictions_found", 0) > 0:
            logger.warning(f"[FLAGGED ID {doc_hash}] Reasoning agent found {reasoning_output_for_eval.get('contradictions_found')} contradictions.")
            shared_memory.set(f"processing_status:{doc_hash}", {"status": "rejected_reasoning", "reason": "contradictions_found"})
            return None
    except Exception as e:
        logger.error(f"Error during reasoning for ID {doc_hash}: {e}", exc_info=True)
        shared_memory.set(f"processing_status:{doc_hash}", {"status": "error_reasoning", "error": str(e)})
        return None
    
    # 4. Knowledge-Augmented Evaluation
    try:
        evaluation_input = {
            "document": doc_content,
            "knowledge": context.get("retrieved_knowledge", []),
            "reasoning_output": reasoning_output_for_eval if isinstance(reasoning_output_for_eval, dict) else {}
        }
        current_eval_result = evaluation_agent.perform_task(evaluation_input) 
        
        passing_score_val = configs.get("evaluation", {}).get("passing_score", 0.7)
        current_score = current_eval_result.get("score", 0.0) if isinstance(current_eval_result, dict) else 0.0
        
        if current_score < passing_score_val:
            logger.info(f"[SKIPPED ID {doc_hash}] Low evaluation score: {current_score:.2f} < {passing_score_val}")
            shared_memory.set(f"processing_status:{doc_hash}", {"status": "rejected_evaluation", "score": current_score})
            return None
    except Exception as e:
        logger.error(f"Error during evaluation for ID {doc_hash}: {e}", exc_info=True)
        shared_memory.set(f"processing_status:{doc_hash}", {"status": "error_evaluation", "error": str(e)})
        return None
    
    # 5. Adaptive Learning (Observe)
    try:
        embedding = extract_embedding(doc_content)
        if embedding is not None and isinstance(embedding, np.ndarray):
            learning_agent.observe((embedding.tolist(), [1])) 
            obs_count = shared_memory.get("learning_observations", 0)
            shared_memory.set("learning_observations", obs_count + 1)
    except Exception as e:
        logger.error(f"Learning agent observation failed for ID {doc_hash}: {str(e)}", exc_info=True)
    
    # 6. Prepare submission metadata
    submission_metadata = {
        "doc_id": doc_hash,
        "content_preview": doc_content[:100],
        "timestamp": context["timestamp"],
        "eval_score": current_eval_result.get("score", 0.0) if current_eval_result else 0.0,
        "safety_score": current_safety_assessment.get("final_safety_score", 0.0) if current_safety_assessment else 0.0,
        "cyber_risk": current_safety_assessment.get("reports", {}).get("cyber_safety", {}).get("risk_score", 1.0) if current_safety_assessment else 1.0,
        "status": "approved_for_submission"
    }
    
    # 7. Submit Data (Actual submission action)
    try:
        submit_data(doc_content, submission_metadata) 
        submission_metadata["status"] = "submitted_successfully"
        shared_memory.set(f"processing_status:{doc_hash}", submission_metadata)
        return submission_metadata 
    except Exception as e:
        logger.error(f"Submission failed for ID {doc_hash}: {str(e)}", exc_info=True)
        submission_metadata["status"] = "error_submission"
        submission_metadata["error"] = str(e)
        shared_memory.set(f"processing_status:{doc_hash}", submission_metadata)
        return None

def extract_embedding(text):
    """Placeholder for embedding generation"""
    import numpy as np
    # Ensure output matches observation_space shape for LearningAgent
    return np.random.rand(configs.get("learning", {}).get("embedding_dim", 512))

STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(os.path.join(STATIC_FOLDER, 'css'), exist_ok=True)
    os.makedirs(os.path.join(STATIC_FOLDER, 'js'), exist_ok=True)
    os.makedirs(os.path.join(STATIC_FOLDER, 'img'), exist_ok=True)
    # Create a dummy dashboard.html if it doesn't exist
    dummy_html_path = os.path.join(STATIC_FOLDER, 'dashboard.html')
    if not os.path.exists(dummy_html_path):
        with open(dummy_html_path, 'w') as f_html:
            f_html.write("<!DOCTYPE html><html><head><title>Dummy Dashboard</title></head><body><h1>Dashboard Loading...</h1><p>If you see this, the actual dashboard.html was not found.</p></body></html>")
        logger.info(f"Created dummy dashboard.html at {dummy_html_path}")

dashboard_app = Flask("SN22")

os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('static/img', exist_ok=True)

@dashboard_app.route('/')
def serve_dashboard():
    return send_from_directory(STATIC_FOLDER, 'dashboard.html')

@dashboard_app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(STATIC_FOLDER, path)

@dashboard_app.route('/favicon.ico')
def favicon():
    # Ensure favicon.ico exists or handle FileNotFoundError
    favicon_path = os.path.join(STATIC_FOLDER, 'favicon.ico')
    if not os.path.exists(favicon_path):
        # Create a tiny dummy favicon if it doesn't exist
        try:
            from PIL import Image
            img = Image.new('RGB', (16, 16), color = 'blue')
            img.save(favicon_path)
            logger.info(f"Created dummy favicon.ico at {favicon_path}")
        except ImportError:
            logger.warning("Pillow not installed, cannot create dummy favicon. Favicon requests might fail.")
            # Return a 404 if you can't create it and it's missing
            return "Favicon not found", 404
        except Exception as e:
            logger.error(f"Error creating dummy favicon: {e}")
            return "Favicon error", 500
            
    return send_from_directory(STATIC_FOLDER, 'favicon.ico')

@dashboard_app.route('/metrics')
def dashboard_metrics():
    try:
        # Ensure recent_submissions_log is a deque
        recent_submissions_raw = shared_memory.get("recent_submissions_log", deque(maxlen=200))
        if not isinstance(recent_submissions_raw, deque):
            if isinstance(recent_submissions_raw, list):
                recent_submissions = deque(recent_submissions_raw, maxlen=200)
            else:
                recent_submissions = deque(maxlen=200)
                logger.warning(f"recent_submissions_log was not a deque or list, reinitialized. Type was: {type(recent_submissions_raw)}")
        else:
            recent_submissions = recent_submissions_raw

        batch_size = configs.get("collaborative", {}).get("batch_size", 5)
        # Get the latest 'batch_size' items from the deque
        current_cycle_docs = list(recent_submissions)[-batch_size:]
        
        eval_scores = [d.get("eval_score", 0.0) for d in current_cycle_docs if isinstance(d, dict)]
        safety_scores = [d.get("safety_score", 0.0) for d in current_cycle_docs if isinstance(d, dict)]
        cyber_risks = [d.get("cyber_risk", 1.0) for d in current_cycle_docs if isinstance(d, dict)]

        current_cycle_rewards = get_recent_rewards(current_cycle_docs) # Pass the processed docs

        total_rewards_session = shared_memory.get("total_rewards_session", 0.0)
        total_docs_processed_session = shared_memory.get("total_docs_processed_session", 0)

        return jsonify({
            "current_eval_score_avg": np.mean(eval_scores).item() if eval_scores else 0.0,
            "current_reward_avg": np.mean(current_cycle_rewards).item() if current_cycle_rewards else 0.0,
            "current_throughput": len(current_cycle_docs),
            "current_cyber_risk_avg": np.mean(cyber_risks).item() if cyber_risks else 0.0,
            "current_safety_score_avg": np.mean(safety_scores).item() if safety_scores else 0.0,

            "orchestrator_status": shared_memory.get("orchestrator_status", "Initializing"),
            "total_docs_processed_session": total_docs_processed_session,
            "total_rewards_session": round(total_rewards_session or 0.0, 4),
            "knowledge_base_size": len(knowledge_agent.knowledge_agent) if hasattr(knowledge_agent, 'knowledge_agent') else 0,

            "blocked_docs_safety": shared_memory.get("safety_blocked", 0),
            "last_block_reason": shared_memory.get("last_block_reason", "N/A"),
            "obs_count": shared_memory.get("learning_observations", 0),
            "training_runs_learning": shared_memory.get("learning_training_runs", 0), # Specific to learning_agent
            "embedding_buffer_size": len(learning_agent.embedding_buffer) if hasattr(learning_agent, 'embedding_buffer') else 0,

            # These seemed like placeholders or generic, let's keep them for now
            "eval_score_avg": shared_memory.get("overall_eval_score_avg", 0.0), # More specific name
            "risk_incidents": shared_memory.get("total_risk_incidents", 0), # More specific name
            "anomaly_flags": shared_memory.get("total_anomaly_flags", 0), # More specific name
        })
    except Exception as e:
        logger.error(f"Error in /metrics endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def run_dashboard(host='0.0.0.0', port=5001):
    logger.info(f"Starting dashboard server on http://{host}:{port}")
    try:
        dashboard_app.run(host=host, port=port, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Failed to start dashboard server: {e}", exc_info=True)

def miner_loop():
    logger.info("Starting Desearch mining loop with knowledge integration")
    shared_memory.set("orchestrator_status", "Running")
    shared_memory.set("total_docs_processed_session", 0)
    shared_memory.set("total_rewards_session", 0.0)
    shared_memory.set("safety_blocked", 0)
    shared_memory.set("learning_observations", 0)
    shared_memory.set("learning_training_runs", 0)
    shared_memory.set("recent_submissions_log", deque(maxlen=200))

    consecutive_errors = 0
    miner_loop_config = configs.get("collaborative", {}) 
    max_errors = miner_loop_config.get("max_consecutive_miner_errors", 5)
    iteration_count = 0

    while True:
        iteration_count += 1
        logger.info(f"Miner Loop Iteration: {iteration_count}")
        shared_memory.set("orchestrator_status", f"Running - Iteration {iteration_count}")
        processed_docs_info_this_cycle = [] # Store successfully processed doc info
        try:
            batch_size_val = miner_loop_config.get("batch_size", 5)
            new_docs = fetch_data(batch_size=batch_size_val) 
            
            if not isinstance(new_docs, list):
                logger.error(f"fetch_data did not return a list. Got: {type(new_docs)}. Skipping this cycle.")
                time.sleep(60) # Wait longer if data fetching is problematic
                continue

            for doc_content in new_docs:
                if not isinstance(doc_content, str):
                    logger.warning(f"Skipping non-string document: {type(doc_content)}")
                    continue
                
                submission_result_dict = process_document(doc_content) # Renamed
                if submission_result_dict and isinstance(submission_result_dict, dict) and \
                   submission_result_dict.get("status") == "submitted_successfully":
                    processed_docs_info_this_cycle.append(submission_result_dict)
                    current_total_docs = shared_memory.get("total_docs_processed_session", 0)
                    shared_memory.set("total_docs_processed_session", current_total_docs + 1)

            if processed_docs_info_this_cycle:
                reward_values = get_recent_rewards(processed_docs_info_this_cycle)
                if isinstance(reward_values, list):
                    cycle_reward_sum = 0
                    for i, reward_val in enumerate(reward_values):
                        if isinstance(reward_val, (int, float)):
                            doc_id_for_reward = processed_docs_info_this_cycle[i]['doc_id']
                            shared_memory.set(f"doc_reward:{doc_id_for_reward}", reward_val) 
                            cycle_reward_sum += reward_val
                            logger.info(f"Logged reward for Doc ID {doc_id_for_reward}: {reward_val:.4f}")
                        else:
                            logger.warning(f"Invalid reward type received: {type(reward_val)}")
                    current_total_rewards = shared_memory.get("total_rewards_session", 0.0)
                    shared_memory.set("total_rewards_session", current_total_rewards + cycle_reward_sum)
                else:
                    logger.warning(f"get_recent_rewards did not return a list. Got: {type(reward_values)}")
            
            # Update LearningAgent observation/training counts for dashboard
            shared_memory.set("embedding_buffer_size", len(learning_agent.embedding_buffer) if hasattr(learning_agent, 'embedding_buffer') else 0)
            learning_train_interval = configs.get("learning", {}).get("train_interval", 10)
            if learning_agent.observation_count > 0 and (iteration_count % learning_train_interval == 0):
                logger.info(f"Miner loop triggering LearningAgent training (iteration {iteration_count}).")
                learning_agent.train_from_embeddings()
                current_training_runs = shared_memory.get("learning_training_runs", 0)
                shared_memory.set("learning_training_runs", current_training_runs + 1)


            kb_rule_interval = configs.get("knowledge", {}).get("rule_discovery_interval", 50)
            if iteration_count > 0 and iteration_count % kb_rule_interval == 0:
                logger.info(f"Miner loop triggering KnowledgeAgent rule discovery (iteration {iteration_count}).")
                knowledge_agent.discover_rules()
                knowledge_agent.broadcast_knowledge()

            consecutive_errors = 0 
            poll_interval_val = miner_loop_config.get("poll_interval", 15)
            shared_memory.set("orchestrator_status", f"Sleeping ({poll_interval_val}s) - Iteration {iteration_count} complete")
            time.sleep(poll_interval_val)
            
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Unhandled error in mining loop (iteration {iteration_count}): {str(e)}", exc_info=True)
            shared_memory.set("orchestrator_status", f"Error Occurred - Iteration {iteration_count}")
            if consecutive_errors >= max_errors:
                logger.critical(f"Too many consecutive errors ({consecutive_errors}). Exiting miner loop.")
                shared_memory.set("orchestrator_status", "STOPPED - Too many errors")
                break
            time.sleep(60)

if __name__ == "__main__":
    dashboard_thread = Thread(
        target=run_dashboard, 
        daemon=True,
        kwargs={'host': '0.0.0.0', 'port': 5001}
    )
    dashboard_thread.start()
    logger.info("Dashboard thread started.")

    try:
        miner_loop()
    except KeyboardInterrupt:
        logger.info("Miner loop interrupted by user. Shutting down.")
        shared_memory.set("orchestrator_status", "Shutting down")
    except Exception as e:
        logger.critical(f"Critical unhandled exception in orchestrator __main__: {e}", exc_info=True)
    finally:
        logger.info("Miner orchestrator finished.")
