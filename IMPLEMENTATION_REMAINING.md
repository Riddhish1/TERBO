# ReCall / TEBO Remaining Work

## Current Status

The repository currently implements the ReCall baseline: multi-turn tool calls, sandbox execution, retriever integration, outcome rewards, PPO/GRPO/RLOO-style optimization, and evaluation.

The TEBO algorithm described in `DA1_RL-_1_.md` is not yet implemented. The remaining work is listed below.

## 1. TEBO Components to Add

### Entropy Monitor

- [ ] Detect the point immediately after a tool response is inserted into the prompt.
- [ ] Compute next-token Shannon entropy at that point.
- [ ] Support measuring the first configurable number of post-tool tokens.
- [ ] Add an entropy threshold `tau`.
- [ ] Store entropy values and decision-point token positions.
- [ ] Make the entropy signal available to the branch manager.
- [ ] Ensure the entropy is measured during rollout, before the continuation is generated.

Relevant code: `src/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

### Branch Manager

- [ ] Fork a trajectory into `k` continuations after a high-entropy tool response.
- [ ] Implement fixed-schedule branching as an intermediate milestone.
- [ ] Track the parent trajectory and all sibling branches.
- [ ] Enforce a maximum number of branches per prompt.
- [ ] Enforce a total branch-token or branch-rollout budget per batch.
- [ ] Reallocate budget from low-entropy or confident rollouts as described in the report.
- [ ] Stop branching when the budget cap is reached.
- [ ] Preserve deterministic ordering across distributed tensor-parallel workers.

### Branch Metadata

Add branch information to the rollout output and preserve it through `DataProto`:

- [ ] `branch_id`
- [ ] `parent_id`
- [ ] `branch_group_id`
- [ ] `shared_prefix_length`
- [ ] `continuation_start`
- [ ] `decision_point_positions`
- [ ] `post_tool_entropy`
- [ ] `branch_budget_used`
- [ ] A mask identifying shared-prefix tokens.
- [ ] A mask identifying branch-continuation tokens.
- [ ] A flag identifying branched versus unbranched samples.

## 2. Advantage Estimation

### Attributed Advantage

- [ ] Add a TEBO attributed-advantage function.
- [ ] Compute one shared-prefix advantage across sibling branches.
- [ ] Compute an individual continuation advantage for each sibling.
- [ ] Compare each continuation against its sibling outcomes.
- [ ] Apply the correct prefix and continuation masks.
- [ ] Fall back to GRPO or RLOO for unbranched trajectories.
- [ ] Handle groups with homogeneous rewards without producing invalid values.
- [ ] Register the estimator in `compute_advantage()`.
- [ ] Add a configuration value such as `algorithm.adv_estimator=tebo`.

Relevant code:

- `src/verl/trainer/ppo/core_algos.py`
- `src/verl/trainer/ppo/ray_trainer.py`

## 3. PPO and Policy Training Changes

### Entropy-Aware PPO Loss

- [ ] Add a high-entropy token mask to the policy loss inputs.
- [ ] Define exactly which part of PPO clipping receives the stop-gradient.
- [ ] Implement the TEBO stop-gradient safeguard.
- [ ] Keep normal PPO clipping for ordinary tokens.
- [ ] Keep KL loss behavior consistent with the baseline.
- [ ] Add equivalent support for the FSDP actor path.
- [ ] Add equivalent support for the Megatron actor path if Megatron training is supported.
- [ ] Log high-entropy token count and clipping statistics.
- [ ] Log normal-token and high-entropy-token losses separately.

Relevant code:

- `src/verl/trainer/ppo/core_algos.py`
- `src/verl/workers/actor/dp_actor.py`
- `src/verl/workers/actor/megatron_actor.py`

## 4. Configuration to Add

Add a TEBO section to `src/verl/trainer/config/ppo_trainer.yaml`:

```yaml
tebo:
  enabled: false
  entropy_threshold: 2.0
  entropy_window: 1
  branch_count: 2
  max_branches_per_prompt: 4
  max_branches_per_batch: 16
  branch_budget_tokens: 4096
  schedule: entropy
  advantage_mode: attributed
  stop_gradient_high_entropy_clip: false
```

Also add command-line overrides to both training scripts for:

- [ ] Enable/disable TEBO.
- [ ] Entropy threshold.
- [ ] Number of branches.
- [ ] Entropy window.
- [ ] Branch budget.
- [ ] Fixed-schedule versus entropy-gated branching.
- [ ] Advantage estimator.
- [ ] High-entropy PPO safeguard.

## 5. Validation and Experiment Work

Implement the four-stage validation plan from the report:

### Stage A: Baseline

- [ ] Run the current GRPO baseline.
- [ ] Record accuracy, format validity, tool success, tokens, runtime, GPU memory, KL, and clip fraction.

### Stage B: Estimator Sanity Checks

- [ ] Run RLOO.
- [ ] Run REINFORCE++.
- [ ] Compare reward normalization and stability against GRPO.

### Stage C: Fixed-Schedule Branching

- [ ] Branch at predefined post-tool decision points.
- [ ] Validate sibling generation and sequence packing.
- [ ] Validate branch metadata.
- [ ] Validate attributed advantage without entropy gating.

### Stage D: Full TEBO

- [ ] Enable entropy-gated branching.
- [ ] Enable branch-budget limits.
- [ ] Enable attributed advantages.
- [ ] Enable the high-entropy PPO safeguard.
- [ ] Compare against all previous stages on held-out tool-use tasks.

### Experiment and Reporting Tools

- [ ] Add reproducible configs for all four stages.
- [ ] Add a head-to-head experiment launcher.
- [ ] Add metrics for branch count and branch budget usage.
- [ ] Add entropy distribution metrics.
- [ ] Add rollout-token accounting.
- [ ] Add results aggregation and comparison output.
- [ ] Save per-trajectory branch metadata for debugging.

## 6. Tests to Add

There is currently no dedicated project test suite for the TEBO behavior.

Create a `tests/` directory and add:

- [ ] `test_tool_rollout.py`
  - Tool-call extraction.
  - Tool-response insertion.
  - Multi-turn stopping behavior.
  - Response and loss masks.

- [ ] `test_entropy_monitor.py`
  - Entropy calculation.
  - Threshold behavior.
  - Post-tool decision-point detection.

- [ ] `test_branch_manager.py`
  - Fixed-schedule branching.
  - Sibling grouping.
  - Branch-budget limits.
  - Distributed ordering.

- [ ] `test_attributed_advantage.py`
  - Shared-prefix advantages.
  - Individual branch advantages.
  - Unbranched fallback.
  - Homogeneous rewards.

- [ ] `test_tebo_loss.py`
  - High-entropy clipping behavior.
  - Stop-gradient behavior.
  - Normal-token PPO behavior.

- [ ] `test_tool_executor.py`
  - Valid tool calls.
  - Invalid JSON.
  - Tool errors.
  - Timeout behavior.

## 7. Existing Bugs to Fix Before Training

### Training Scripts

In:

- `scripts/train/train.sh`
- `scripts/train/train_multi_node.sh`

Fix the following:

- [ ] Use the parsed `N_GPUS_PER_NODE` instead of hard-coding `8`.
- [ ] Initialize `CHECKPOINT_SAVE`; it is currently referenced but never defined.
- [ ] Quote shell paths and variable expansions consistently.
- [ ] Use the configured GPU count for multi-node Ray startup.
- [ ] Validate required arguments before launching training.

### Tool-Call Parsing

In `src/re_call/inference/re_call.py`:

- [ ] Make the parse-error return string match the caller's error check.
- [ ] Apply the same parsing behavior in the training rollout and inference wrapper.
- [ ] Add tests for malformed tool-call JSON.

### Sandbox Safety

In `scripts/serving/sandbox.py`:

- [ ] Enforce the requested timeout. The current `timeout` field is unused.
- [ ] Execute user/model code in a separate process.
- [ ] Kill processes that exceed the timeout.
- [ ] Add memory limits.
- [ ] Restrict imports and filesystem access.
- [ ] Restrict or disable network access where appropriate.
- [ ] Limit request and output sizes.
- [ ] Do not expose the current unrestricted `exec()` service publicly.

### Dataset Validation

In `src/verl/utils/dataset/rl_dataset.py`:

- [ ] Validate that tool-enabled samples contain an `env` field.
- [ ] Produce a clear error when a required tool environment is missing.
- [ ] Support custom tools without assuming only MuSiQue behavior.

### Evaluation Command

In `README.md`:

- [ ] Add the missing continuation slash after `--save_note re-call_qwen7b_ins`.
- [ ] Document TEBO evaluation options.
- [ ] Document the four experiment stages.
- [ ] Document the required metrics and output files.

## 8. Documentation to Add

- [ ] Add a TEBO architecture section to `README.md`.
- [ ] Document the rollout data contract.
- [ ] Document branch IDs and prefix/continuation masks.
- [ ] Document the attributed-advantage formula.
- [ ] Document entropy units, threshold selection, and entropy window behavior.
- [ ] Document branch-budget accounting.
- [ ] Document how to reproduce each validation stage.
- [ ] Document known GPU, CUDA, vLLM, Ray, and Linux/WSL requirements.
- [ ] Update the citation and project description if TEBO becomes the primary contribution.

## Recommended Implementation Order

1. Fix training scripts, parsing, and sandbox timeout behavior.
2. Add the rollout metadata contract.
3. Implement and test fixed-schedule branching.
4. Implement and test attributed advantages.
5. Add entropy monitoring during rollout.
6. Add entropy-gated branching and budget reallocation.
7. Add the PPO stop-gradient safeguard.
8. Add staged experiment configs and comparison tooling.
9. Run held-out benchmarks and update the documentation.

## Validation Command

A syntax-only project check currently passes:

```bash
python -m compileall -q src scripts data
```

It reports existing invalid-escape warnings in several files, which should be cleaned up separately.
