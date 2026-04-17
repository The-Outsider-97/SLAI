sensitive_attributes: ["gender", "age_group", "race", "education_level"]

base_agent:
    defer_initialization: True
    memory_profile: low
    network_compression: True
    max_error_log_size: 50
    error_similarity_threshold: 0.75
    max_task_retries: 0
    task_timeout_seconds: None
    task_similarity_str_threshold: 0.9
    jaccard_threshold: 0.5
    jaccard_min_for_no_shared: 0.7
    final_key_threshold: 0.7
    final_value_threshold: 0.7
    task_similarity_seq_elem_threshold: 0.8

adaptive_agent:
  state_dim: 10
  num_actions: 2
  num_handlers: 3
  max_episode_steps: 100

  base_learning_interval: 10
  skill_max_steps: 10
  skill_batch_size: 32
  manager_batch_size: 32
  tuning_window: 100
  performance_window: 50

  max_task_history: 200
  max_route_history: 200
  max_feedback_history: 200

  random_seed: 42

  explore_skills: true
  explore_actions: true
  learn_after_task: true
  auto_behavior_cloning: true
  auto_dagger_update: true
  auto_meta_optimize: true
  auto_save_state: true
  auto_log_interventions: true
  sync_tuner_to_skills: true
  use_global_rl_engine: true
  share_meta_registry: true

  recovery_reward_threshold: -10.0
  success_reward_threshold: 0.0
  correction_bonus: 1.0
  demonstration_bonus: 2.0
  feedback_bonus: 0.5
  route_similarity_threshold: 0.25

  shared_memory_state_ttl: 2592000
  shared_memory_recovery_ttl: 604800
  shared_memory_report_ttl: 2592000

  default_task_type: generic
  task_embedding_strategy: hash_bow
  report_detail_level: full

  meta_update_frequency: 5
  auto_behavior_cloning_epochs: 1
  checkpoint_protocol: 5

  skills: {}
  handlers: {}
  env: {}

alignment_agent:
    safety_buffer: 0.1
    learning_rate: 0.01
    momentum: 0.9
    risk_score:
    risk_assessment:
        total_risk: []
    operation_limiter:
        max_requests: 10
        interval: 60
        penalty: 'cool_down'
    weight:
        stat_parity: 0.4
        equal_opp: 0.4
        ethics: 0.2
    corrections: {}
    alignment_ttl: 604800
    fail_safe_action_space: []
    risk_threshold: 0.5

browser_agent:
    max_retries: 3
    default_wait: 0.0
    default_search_engine: "https://www.google.com"
    headless: true
    user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    window_size: [1366, 920]

collaborative_agent:
    risk_threshold: 0.7
    max_concurrent_tasks: 100
    load_factor: 0.75
    optimization_weight_capability: 0.5
    optimization_weight_load: 0.3
    optimization_weight_risk: 0.2
    bayes_prior_alpha: 1.0
    bayes_prior_beta: 1.0
    use_collaboration_manager: true

evaluation_agent:
    risk_threshold: 0.5
    memory_warning_threshold: 0.7
    memory_critical_threshold: 0.9
    model_dir: "src/agents/evaluators/models/"
    anomaly_detector: "src/agents/evaluators/models/anomaly_detector.joblib"
    deep_anomaly: "src/agents/evaluators/models/deep_anomaly.pt"
    initial_hazard_rates:
        system_failure: 0.000001
        sensor_failure: 0.00001
        unexpected_behavior: 0.0001
    update_interval: 60.0
    decay_factor: 0.95
    risk_thresholds:
        warning: [0.4, 0.7]
        critical: [0.7, 1.0]
    risk_adaptation: {}
    autonomous_tasks:
        - id: "nav_test"
    type: "navigation"
    path: [[0,0], [1,1], [2,2]]
    optimal_path: [[0,0], [2,2]]
    completion_time: 10.0
    energy_consumed: 100
    collisions: 0
    success: true

issue_database:
    host: 'localhost'
    port: 5432
    database: 'ai_issues'
    user: 'user'
    password: ''

operations_notification:
    email:
        enabled: true
        from: "ai-agent@yourdomain.com"
        to: "ops@yourdomain.com"
        subject: "URGENT: AI System Hazard"
        smtp_host: "smtp.yourdomain.com"
        smtp_port: 587
        username: "ai-agent"
        password: "secure-password"
        use_tls: true

    webhook:
        enabled: true
        url: "https://hooks.slack.com/services/..."

    external_logging:
        enabled: true
        url: "https://logging-service.yourdomain.com/ingest"

execution_agent: {}

handler_agent:
    max_retries: 2
    circuit_breaker_threshold: 5
    cooldown_seconds: 30
    failure_budget_window_seconds: 300
    checkpoint_max_age_seconds: 600
    telemetry_buffer_size: 1000
    evaluator_hooks_enabled: true

knowledge_agent:
    source: "local_files"
    is_query: True
    stopwords: "src/agents/language/templates/stopwords.json"
    knowledge_ontology_path: "src/agents/knowledge/utils/knowledge_ontology.db"
    use_graph_ontology: True  # False for in-memory
    ontology_threshold: 500   # Switch to graph DB at this size
    in_memory_cache: True  # Whether to keep in-memory cache
    cache_size: 1000
    bias_threshold: 0.87
    bias_detection_enabled: True
    embedding_model: "src/agents/knowledge/models/all-MiniLM-L6-v2"
    directory_path: "src/agents/knowledge/templates/"
    similarity_threshold: 0.35
    use_ontology_expansion: True
    first_pass: True
    decay_factor: 0.95
    context_window: 10
    knowledge_tag: "knowledge_snippet"
    ttl: 600
    retrieval_mode: "hybrid"
    safety_check_callback: foodie_security.validate_action
    max_workers: 8

language_agent: {}

learning_agent:
    rl_algorithm: null
    embedding_dim: 512
    strategy_weights: [0.25, 0.25, 0.25, 0.25]
    prediction_weights: [0.25, 0.25, 0.25, 0.25]
    maml_task_pool_size: 100
    rsi_improvement_cycle: 50
    performance_threshold: 0.7
    data_change_threshold: 0.15
    retraining_interval_hours: 24
    novelty_threshold: 0.3
    uncertainty_threshold: 0.25
    embedding_buffer_size: 512
    performance_history_size: 1000
    error_history_size: 100
    state_recency_size: 1000
    architecture_history_size: 10
    maml_adaptation_steps: 10
    task_embedding_dim: 256
    task_ids: ["dqn", "maml", "rsi", "rl"]
    batch_size: 32
    max_episode_steps: 128
    max_eval_episodes: 3
    recovery_trigger_threshold: 3
    deferred_init:
        max_network_size: 256
        max_task_pool: 50
        max_history: 500

network_agent:
    max_delivery_attempts: 3
    retry_sleep_enabled: False
    max_retry_sleep_ms: 1500
    default_receive_timeout_ms: 5000
    health_shared_key:
        network_agent: "health"
    last_result_shared_key:
        network_agent: "last_result"
    history_shared_key:
        network_agent: "history"
    max_history: 200

observability_agent:
    enabled: true
    default_service: "slai"
    default_task_name: "agent_workflow"
    default_trace_operation: "observe"
    default_incident_status: "open"

    auto_start_trace: true
    auto_finalize_trace: true
    enable_trace_span_ingestion: true
    enable_trace_event_ingestion: true
    enable_state_transition_ingestion: true
    enable_log_ingestion: true
    enable_performance_trace_analysis: true
    enable_health_snapshots: true
    enable_kpi_tracking: true
    enable_shared_context_export: true
    allow_degraded_reports: true

    max_recent_reports: 100
    max_recent_errors: 200
    max_signature_history: 512
    max_event_records_per_run: 200
    max_log_records_per_run: 200
    max_alert_records_per_run: 100
    max_error_records_per_run: 100
    max_state_transition_records_per_run: 200
    max_objective_records_per_run: 50
    max_related_agents: 16

    alert_dedupe_window_seconds: 1800.0
    alert_dedupe_repeat_threshold: 3
    recurring_incident_threshold: 2
    alert_precision_decay: 1.0
    degraded_status_levels: ["warning", "critical"]
    required_trace_context_fields: ["task_name", "agent_name", "operation_name"]

    routing:
        handler_on_warning: true
        handler_on_critical: true
        planning_on_capacity: true
        safety_on_critical: true
        evaluation_on_degradation: true
        handler_agent_names: ["handler", "handler_agent", "HandlerAgent"]
        planning_agent_names: ["planning", "planning_agent", "PlanningAgent"]
        safety_agent_names: ["safety", "safety_agent", "SafetyAgent"]
        evaluation_agent_names: ["evaluation", "evaluation_agent", "EvaluationAgent"]

perception_agent:
    device: "cpu"
    embed_dim: 512
    masking_ratio: 0.15
    contrastive_temperature: 0.07
    learning_rate: 0.0001
    weight_decay: 0.01
    adam_betas: [0.9, 0.999]
    adam_eps: 0.0000001
    loss_type: 'hybrid'  # mse, contrastive, or hybrid
    max_scale: 3
    temperature: 0.1
    mse_weight: 1.0     # for hybrid loss
    contrastive_weight: 1.0  # for hybrid loss

planning_agent: {}

privacy_agent: {}

quality_agent:
    enabled: true
    default_window: latest
    stop_on_blocking_structural: true
    fail_closed_on_subsystem_error: true
    auto_route_via_workflow: true
    prefer_workflow_verdict: true
    include_workflow_findings_in_summary: true
    include_record_previews_in_shared_memory: false
    max_shared_record_preview: 5
    pass_threshold: 0.90
    warn_threshold: 0.75

    subsystem_order:
        - structural
        - statistical
        - semantic

    subsystem_weights:
        structural: 0.34
        statistical: 0.33
        semantic: 0.33

    bridge_resolution:
        resolve_handler_from_factory: false
        resolve_safety_from_factory: false
        handler_factory_names:
            - handler_agent
            - handler
            - HandlerAgent
        safety_factory_names:
            - safety_agent
            - safety
            - SafetyAgent

    shared_memory:
        enabled: true
        ttl_seconds: 86400
        publish_notifications: true
        result_key_prefix: quality_agent.result
        summary_key_prefix: quality_agent.summary
        error_key_prefix: quality_agent.error
        event_channel: quality.events

reader_agent:
    default_output_format: "txt"
    output_dir: "output/reader"
    max_concurrency: 4
    enable_cache: true
    recovery_min_quality_score: 0.55

reasoning_agent:
    learning_rate: 0.01
    exploration_rate: 0.1
    tuple_key: ["subject", "predicate", "object"]
    hypothesis_graph:
        nodes: {}
        edges: {}
        confedence: 0.0
    language_config_path: "src/agents/language/configs/language_config.yaml"
    knowledge_db: "src/agents/knowledge/templates/knowledge_db.json"
    glove_path: "data/embeddings/glove.6B.200d.json"
    ner_tag: None
    embedding: None
    decay: 0.8
    max_iterations: 100

safety_agent:
    constitutional_rules_path: "src/agents/safety/templates/constitutional_rules.json"
    audit_level: 2
    collect_feedback: true
    enable_learnable_aggregation: true
    system_models: {}
    known_hazards: []
    global_losses: []
    safety_policies: []
    architecture_map: {}
    formal_specs: {}
    fault_tree_config: {}
    risk_thresholds:
        overall_safety: 0.85
        cyber_risk: 0.7
        compliance_failure_is_blocker: true
    secure_memory:
        default_ttl_seconds: 3600
