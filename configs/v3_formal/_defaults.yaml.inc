schema_version: v3-formal-experiment-config-v1
metadata:
  name: overridden-by-runnable-config
  description: overridden-by-runnable-config
variant: v3_role
ruleset:
  id: legacy
  version: legacy-v1
  hash: 8f43c73a1d07e0da0d5127b958303115701b176f5725223eb4f490b8a56ad3a0
model:
  family: v3_hybrid
  version: v3_hybrid
  config:
    hidden_size: 256
    history_encoder: lstm
    history_layers: 2
    history_heads: 8
    history_dropout: 0.0
    shared_fusion_layers: 2
    landlord_adapter_layers: 2
    farmer_adapter_layers: 4
    farmer_channel_gate: none
    farmer_channel_gate_reduction: 4
    adapter_dropout: 0.0
    attention_type: none
    score_clamp: 32.0
    dmc_target_transform: raw
    dmc_target_clamp: 32.0
    nan_guard: true
    belief_feedback: none
    human_prior_enabled: false
    strategy_features_enabled: false
    strategy_hand_enabled: true
    strategy_structure_enabled: true
    strategy_control_enabled: true
    strategy_cooperation_enabled: true
    strategy_risk_enabled: true
    strategy_aux_enabled: false
    strategy_node_budget: 500
    strategy_time_budget_ms: 0
    style_enabled: false
    style_embedding_dim: 64
    bidding_enabled: false
    bidding_hidden_size: 128
    bidding_uncertainty_enabled: false
features:
  role_model: true
  adaptive_dmc: false
  oracle: false
  belief: false
  cooperation: false
  human_bc: false
  strategy: false
  style: false
  bidding: false
  search: false
losses:
  weights:
    dmc: 1.0
    win: 1.0
    score: 0.5
    oracle: 0.0
    belief: 0.0
    cooperation: 0.0
    bc: 0.0
    strategy: 0.0
    bidding: 0.0
  schedules:
    dmc: {kind: constant, start: 1.0, end: 1.0, updates: 0}
    win: {kind: constant, start: 1.0, end: 1.0, updates: 0}
    score: {kind: constant, start: 1.0, end: 1.0, updates: 0}
    oracle: {kind: constant, start: 1.0, end: 1.0, updates: 0}
    belief: {kind: constant, start: 1.0, end: 1.0, updates: 0}
    cooperation: {kind: constant, start: 1.0, end: 1.0, updates: 0}
    bc: {kind: constant, start: 1.0, end: 1.0, updates: 0}
    strategy: {kind: constant, start: 1.0, end: 1.0, updates: 0}
    bidding: {kind: constant, start: 1.0, end: 1.0, updates: 0}
  role_weights:
    landlord: 1.0
    landlord_up: 1.0
    landlord_down: 1.0
feature_configs:
  adaptive_dmc:
    enabled: false
    mode: disabled
    gamma_start: 0.2
    gamma_end: 0.05
    gamma_schedule_updates: 100000
    epsilon: 0.001
    delta: 0.1
  oracle:
    enabled: false
    warmup_updates: 0
    guided_updates: 0
    finetune_updates: 0
    guidance_weight_start: 0.0
    guidance_weight_end: 0.0
    temperature_start: 2.0
    temperature_end: 1.0
  belief:
    enabled: false
    mode: disabled
    feedback: none
    policy_updates_per_cycle: 1
    belief_updates_per_cycle: 1
    shared_updates_per_cycle: 0
  cooperation:
    enabled: false
    mode: disabled
    mixer: disabled
    trajectory_alignment: sequential-farmer-pair-v1
  human_bc:
    enabled: false
    dataset_identity: null
    license: null
    dataset_version: null
    pseudonymization_contract: null
    hmac_key_id_hash: null
  strategy:
    enabled: false
    feature_layout: public-strategy-v1
  style:
    enabled: false
    history_access: public_only
  bidding:
    enabled: false
    action_semantics: separate_bid_action_v1
  search:
    enabled: false
    training_semantics: deployment_wrapper_shared_checkpoint_v1
runtime:
  topology: single_process
  device: cuda
  batch_size: 32
  replay_capacity: 100000
  checkpoint_cadence_updates: 1000
  policy_lag_limit: 0
  checkpoint_enabled: true
seeds:
  training: [101, 202, 303]
  evaluation: 41001
  deal_set: 51001
  derivation: sha256(root_seed,stream_name,worker_id,episode_id)-v1
budgets:
  pilot:
    training_seed_count: 1
    wall_clock_seconds: 3600
    sample_budget: 1000000
    optimizer_step_budget: 10000
    paired_deals: 5000
  development:
    training_seed_count: 3
    wall_clock_seconds: 14400
    sample_budget: 5000000
    optimizer_step_budget: 50000
    paired_deals: 20000
  promotion:
    training_seed_count: 3
    wall_clock_seconds: 28800
    sample_budget: 20000000
    optimizer_step_budget: 200000
    paired_deals: 100000
initialization:
  kind: seeded_fresh
  source: v3-role-seeded-fresh-v1
  seed: 101
  checkpoint_path: null
  checkpoint_sha256: null
  checkpoint_kind: null
evaluation:
  bootstrap_unit: deal
  confidence: 0.95
  deal_set_strategy: fixed-deal-seat-swap-v1
