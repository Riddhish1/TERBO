Absolutely. Here is **all the text extracted from the PPT**, organized slide-by-slide. I’ve kept the wording and structure as close to the original as possible. 

---

# SLIDE 1 — DA1 // PROJECT REPORT

## TEBO

**Byte-Level RL Framework for Robust Agentic Tool-Use Reasoning**

A reinforcement-learning framework and algorithm for multi-turn, tool-augmented LLM reasoning entropy-gated branching with attributed credit assignment at every tool-call decision point.

### OPERATOR

Vijval Shah
Jiya Sharma
Riddhish Bonde

### REGISTER NO.

23BRS1431
23BAI1050
23BPS1203

---

# SLIDE 2 — EXECUTIVE SYNTHESIS

## Abstract

### BASELINE — UNOPTIMIZED PATH

* Single scalar advantage applied to an entire multi-turn rollout
* Group-std normalization collapses when rewards are homogeneous (std → 0)
* No signal at the moment right after a tool call — exactly where uncertainty spikes

**One scalar advantage smeared uniformly across an entire trajectory — flat sampling, fragile group-std normalization.**

### TEBO — STRUCTURED BRANCHING

* Entropy Monitor flags high-uncertainty tokens right after each tool response
* Branch Manager forks k continuations exactly at those decision points
* Attributed Advantage splits credit:

  * shared prefix
  * each branch's own outcome

**Entropy-gated forks at every post-tool-call decision point, each with its own attributed advantage signal.**

### SETTING

Multi-turn, tool-augmented rollouts — not single-turn verifiable reasoning.

### PROBLEM

Group optimization is entropy-blind and collapses when group std → 0.

### METHOD

Entropy Monitor + Branch Manager + Attributed Advantage + Safeguards.

### OUTCOME

Higher tool-use accuracy at an equal or lower rollout-token budget.



---

# SLIDE 3 — THE PARADIGM

## Introduction

### 01 — WHY TOOLS MATTER

LLMs can't look up post-cutoff facts, execute reliable arithmetic, or verify live claims — tool-augmented agents systematically outperform closed-book models on multi-hop QA and verification tasks.

### 02 — WHY REINFORCEMENT LEARNING

Supervised traces teach imitation of a fixed pattern. RLVR lets tool-use strategy emerge from trial and error, rewarded only on correctness.

### 03 — THE BASE FRAMEWORK

A novel RLVR loop: sample prompts, roll out with `<tool_call>` access to a sandbox / retriever, splice results into context, score, update policy.

### 04 — THE SPECIFIC GAP

Standard group optimization is blind to internal rollout structure — it can't detect or weight the high-uncertainty tokens right after a tool response returns.

## PROJECT OBJECTIVE

### Objective

Document the baseline pipeline's precise mismatch with tool-augmented rollouts; ground a fix in current agentic-RL literature; and fully specify TEBO — a new algorithm that detects post-tool-call decision points and allocates credit and exploration exactly there.



---

# SLIDE 4 — THEORETICAL FOUNDATION

## Literature Review — 15 Sources

### CORE POLICY OPTIMIZATION

**[1] 2017**
Schulman et al. — PPO

Clipped surrogate objective; the ancestor mechanism every algorithm below modifies.

**[2] 2024**
Shao et al. — DeepSeekMath

Group-relative optimization removes the learned critic entirely.

**[3] 2024**
Ahmadian et al. — RLOO

Leave-one-out baseline avoids group-std blow-ups.

**[6] 2025**
Zheng et al. — GSPO

Sequence-level, not token-level, importance ratios.

### AGENTIC SCALING & CREDIT

**[7] 2025**
Feng et al. — GiGPO

Two-level episode + step groups for long-horizon agents.

**[8] 2025**
Dong et al. — ARPO

Entropy-spike branching right after tool responses.

**[9] 2025**
Dong, Bao et al. — AEPO

Branch-budget cap + stop-gradient on high-entropy tokens.

**[13] 2026**
Progress/Reliability GPO

Progress + reliability signals beyond one scalar reward.

### TOOL-USE & SYSTEMS

**[4] 2025**
Chen et al. — ReSearch

Pure-RL search-call reasoning for multi-hop QA.

**[5] 2025**
Yu et al. — DAPO

Clip-Higher + Dynamic Sampling fix entropy collapse.

**[10] 2026**
Long-Horizon Tool Recipe

Systems-level study of what actually moves the needle.

**[11] 2026**
Tool-Abuse Mitigation

Penalizes unnecessary tool invocation.

**[12] 2025**
Turn-Level IS + Clip Norm

Stabilizes off-policy long-horizon training.

**[14] 2025**
SLM Tool-Use RL

Reward design for small-model syntax errors.

**[15] 2025**
Agentic Reasoning + Tools

Joint reasoning/tool-use objective.



---

# SLIDE 5 — THE STATUS QUO

## Baseline Methodology

### 01 — BASE MODEL & SERVING

Qwen2.5-7B-Instruct fine-tuned via RL; high-throughput engine for batched rollout generation.

### 02 — ROLLOUT LOOP

Generate → detect `<tool_call>` → dispatch to sandbox / retriever → splice result → continue.

### 03 — REWARD FUNCTION

Outcome-only: final-answer correctness + output-format adherence. No step-level signal.

### 04 — BASELINE FORMULA

Aᵢ = (Rᵢ − mean(R)) / std(R) across a group of G sampled rollouts per prompt.

### 05 — POLICY UPDATE

PPO-style clipped surrogate loss + KL penalty vs. a frozen reference policy.

## SYSTEM LIMITATIONS — 3 FAILURE MODES

### 1. FLAT SAMPLING

G independent full rollouts per prompt — no way to concentrate budget at the moments right after a tool call, where uncertainty actually spikes.

### 2. STD INSTABILITY

Tool-use groups are often all-correct or all-wrong; group std → 0 makes the advantage numerically explosive or meaningless.

### 3. NO SUB-TRAJECTORY CREDIT

One scalar advantage is smeared across an entire multi-turn rollout — "good tool call, bad usage" and "bad tool call" look identical to the optimizer.



---

# SLIDE 6 — THE INNOVATION

## Proposed Methodology — TEBO

### NEW 01 — ENTROPY-GATED BRANCHING

Measures next-token Shannon entropy on the tokens immediately after a tool response splices into context. If H exceeds threshold τ, forks the trajectory into k continuations, reallocating budget from confident (low-entropy) rollouts elsewhere in the batch.

### NEW 02 — ATTRIBUTED ADVANTAGE

Splits credit for branched trajectories: a shared advantage for the common prefix (scored across all k siblings as one group) and an individual advantage for each sibling's own continuation vs. its k−1 siblings.

### NEW 03 — STABILITY SAFEGUARDS

A branch-budget cap bounds compute growth if the model is broadly uncertain; a stop-gradient on the clipping term for high-entropy tokens keeps PPO from zeroing exactly the gradients that matter most.

### NEW 04 — STAGED VALIDATION PLAN

(a) baseline benchmark,
(b) RLOO / REINFORCE++ sanity checks,
(c) fixed-schedule branching as an engineering milestone,
(d) full TEBO — compared head-to-head on held-out tool-use tasks.

### Design lineage

Entropy-driven branching and step-level credit assignment build on GiGPO [7], ARPO [8] and AEPO [9] — TEBO's specific mechanism, integration point, and validation plan are original to this project.



---

# SLIDE 7 — THE ENGINE ROOM — SYSTEM LEVEL

## Overall Architecture

### BASELINE COMPONENT

Prompt, Policy Model, Tool Executor, Reward, Estimators, Clipped Loss

### NOVEL TEBO COMPONENT

Entropy Monitor, Branch Manager, Attributed Advantage, Stop-Gradient Safeguard

### Architecture flow

**Stage A — Rollout Generation**

Prompt / Query
↓
Policy Model
↓
Generate Tokens
↓
Tool Call?
↓
Sandbox / Retriever
↓
Generate Tokens

At the relevant post-tool-response decision point:

**Branch Manager → fork k continuations**

↓

**Stage B — Scoring**

Completed Trajectory
↓
Reward Function

↓

**Stage C — Update**

Attributed Advantage
↓
Stop-Gradient Safeguard
↓
Updated Policy Weights



---

# SLIDE 8 — THE ENGINE ROOM — COMPONENT LEVEL

## Component-Wise Deep Dive

### 1. ENTROPY-GATED BRANCHING

**Process:**

Post-tool tokens → Compute entropy H → H > τ → fork k

Shannon entropy is computed on the next-token distribution for the first few tokens generated right after a tool response splices into context. Extra continuations are funded by reallocating budget away from low-entropy (confident) rollouts elsewhere in the batch, so total rollouts per step stay constant.

**Project application:** wraps directly around the Rollout Worker's tool-call loop (section 4.2, step 4) — it watches real Sandbox / Retriever responses in this project's pipeline, not a synthetic signal.

---

### 2. ATTRIBUTED ADVANTAGE ESTIMATION

**Process:**

k sibling branches → Shared + individual advantage → Attributed signal

Replaces the single scalar advantage from baseline group optimization for branched groups only: a shared advantage scores the common prefix across all k siblings; an individual advantage compares each sibling's own continuation to its k−1 siblings.

**Project application:** feeds the existing Advantage Estimator module — triggered only when the Branch Manager has created siblings for that prompt; falls back to standard GRPO / RLOO otherwise, so the current pipeline keeps working unmodified elsewhere.

---

### 3. STABILITY SAFEGUARDS

**Process:**

Branch budget cap → Stop-gradient on high-H tokens → Stable PPO update

A branch-budget cap bounds compute growth if the model is broadly uncertain early in training. A stop-gradient on PPO's clipping term for high-entropy tokens specifically prevents the gradient from being zeroed exactly where branching says it matters most.

**Project application:** plugs into the Policy Trainer alongside the existing PPO-clipped loss + KL penalty; the budget cap is enforced inside the Branch Manager before any extra rollout is dispatched, protecting this project's fixed per-step compute budget.

---

### 4. STAGED VALIDATION PLAN

**Process:**

(a) Baseline → (b) RLOO / REINFORCE++ → (c) Fixed-schedule branch → (d) Full TEBO

A four-stage rollout de-risks the architecture: each stage isolates one variable — sampling, estimator choice, branching mechanics — before the next is layered on top of it.

**Project application:** this is the project's actual experimental roadmap — stages (a)/(b) establish a fair baseline on this project's held-out tool-use benchmarks before entropy-gating (5.1) is engineering-validated in (c) and the complete algorithm ships in (d).



---

# SLIDE 9 — TECH STACK OVERVIEW

## Module Description

| STATUS | MODULE                      | CLASS                          | DESCRIPTION                                                                                                    |
| ------ | --------------------------- | ------------------------------ | -------------------------------------------------------------------------------------------------------------- |
|        | **Rollout Worker**          | Core Engine (Modified)         | Step-wise tool-call loop; now invokes the Entropy Monitor right after a tool response is spliced into context. |
|        | **Tool Executor — Sandbox** | Service Layer                  | Executes model-issued Python code calls in an isolated remote sandbox service.                                 |
|        | **Retriever Service**       | Service Layer                  | Serves Wikipedia search / retrieval for multi-hop QA tasks.                                                    |
|        | **Entropy Monitor**         | Novel Module                   | Reads per-token logprobs post-tool-response; flags high-uncertainty points via threshold τ.                    |
|        | **Branch Manager**          | Novel Module                   | Forks flagged trajectories into k continuations; reallocates budget; enforces the branch-budget cap.           |
|        | **Advantage Estimator**     | Math Core (Modified)           | Evaluates baseline estimators alongside TEBO's attributed advantage (shared + per-branch).                     |
|        | **Policy Trainer**          | Optimization Engine (Modified) | PPO-clipped loss + KL penalty, plus the new stop-gradient safeguard on high-entropy tokens.                    |
|        | **Reward Function**         | Evaluation Logic               | Verifiable outcome reward — final-answer correctness and output-format adherence.                              |



---

# SLIDE 10 — THE DATABASE

## References

**[1]** Schulman, Wolski, Dhariwal, Radford, Klimov (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.

**[2]** Shao, Wang, Zhu, Xu, Song, Bi, Zhang, Zhang, Li, Wu, et al. (2024). *DeepSeekMath.* arXiv:2402.03300.

**[3]** Ahmadian, Cremer, Gallé, Fadaee, Kreuzer, Pietquin, Üstün, Hooker (2024). *Back to Basics: REINFORCE-Style Optimization.* arXiv:2402.14740.

**[4]** Chen, Wan, Wu, Wu, et al. (2025). *ReSearch: Learning to Reason with Search for LLMs via RL.* arXiv:2503.19470.

**[5]** Yu, Zhou, Wu, Cheng, Xu, Zhou, Zhang, Sun, et al. (2025). *DAPO.* arXiv:2503.14476.

**[6]** Zheng, Liu, Li, Chen, Yu, Gao, Dang, Liu, Men, Yang, Zhou, Lin (2025). *Group Sequence Policy Optimization.* arXiv:2507.18071.

**[7]** Feng, Xue, Wang, Xu, Guo, Wang, Fan, Zhao (2025). *Group-in-Group Policy Optimization.* arXiv:2505.10978.

**[8]** Dong, Chen, Cao, Chen, Wan, Cui, Chen, Zheng, Ren, Zhao, Wei, Wen (2025). *Agentic Reinforced Policy Optimization.* arXiv:2507.19849.

**[9]** Dong, Bao, et al. (2025). *Agentic Entropy-Balanced Policy Optimization.* arXiv:2510.14545.

**[10]** *Demystifying RL for Long-Horizon Tool-Using Agents* (2026). arXiv:2603.21972.

**[11]** *Learning When Not to Act: Mitigating Tool Abuse in Agentic RL* (2026). arXiv:2606.02132.

**[12]** *Stabilizing Off-Policy Training via Turn-Level Importance Sampling* (2025). arXiv:2511.20718.

**[13]** *Progress- and Reliability-Oriented Group Policy Optimization* (2026). arXiv:2607.04242.

**[14]** *Advancing SLM Tool-Use Capability using RL* (2025). arXiv:2509.04518.

**[15]** *Agentic Reasoning and Tool Integration for LLMs via RL* (2025). arXiv:2505.01441.

