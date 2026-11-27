# Training-based methods that make LLMs useful for function-level vulnerability detection (2023-2026)

Scope note: covers fine-tuned LLM detectors, reasoning distillation, RL-trained reasoning (GRPO-family), preference/contrastive training. Numbers are the authors' own unless stated. "Random split" = random/stratified split of a mixed dataset; "realistic" = PrimeVul time split and/or paired (vulnerable vs. patched) evaluation. Pairwise metric names vary: PrimeVul calls it P-C (pair-correct), ReVD calls it VP-S, VULPO uses P-Pass@k.

## Q1. Which fine-tuned LLM approaches beat fine-tuned small code models (CodeBERT/UniXcoder/CodeT5)?

### Takeaway
On random or mixed splits, several fine-tuned 1.5B-13B LLMs beat CodeBERT/UniXcoder by about 5-10 F1 points. VulLLM-CL-13B beat UniXcoder on Devign (71.63 vs 66.05) and on ReVeal (64.52 vs 59.57), and R2Vul-1.5B beat CodeBERT (81.03 vs 71.51 macro-F1). On PrimeVul-style time/paired evaluation the gains are small in absolute terms. The best numbers are ReVD at 58% accuracy / VP-S 16 and VULPO-4B at P-Pass@1 22.6. An independent 2026 execution-grounded study found that VulnLLM-R-7B separates matched vulnerable/fixed pairs no better than chance.

### Cited Findings
**VulLLM (Du et al., Findings of ACL 2024): multi-task instruction tuning (detection + localization + GPT-4 "CoT-SV" interpretation)**
- Data: PatchDB for the auxiliary tasks, plus DiverseVul and Devign for detection. A random-identifier-substitution augmentation replaces 10% of identifiers. Base models: Llama-2, StarCoder and CodeLlama at 7B/13B/15B. — [VulLLM, ACL Findings 2024](https://aclanthology.org/2024.findings-acl.625.pdf)
- F1, Table 1. Training sets (Dataset1) are DiverseVul and Devign. BigVul, CVEfixes, ReVeal and Juliet are not used in training. — [VulLLM](https://aclanthology.org/2024.findings-acl.625.pdf)

  | Model | DiverseVul | Devign | BigVul | CVEfixes | ReVeal | Juliet | Avg all |
  |---|---|---|---|---|---|---|---|
  | CodeBERT (125M) | 65.45 | 66.81 | 55.19 | 56.03 | 46.48 | 5.36 | 49.22 |
  | UniXcoder (126M) | 67.27 | 66.05 | 56.06 | 59.35 | 59.57 | 61.36 | 61.61 |
  | VulLLM-CL-7B | 68.23 | 67.84 | 63.51 | 59.26 | 66.45 | 58.84 | 64.02 |
  | VulLLM-CL-13B | 70.99 | 71.63 | 64.42 | 61.92 | 64.52 | 65.77 | 66.54 |
- Robustness: under identifier/dead-code attacks, average F1 was VulLLM 36.64 vs UniXcoder 22.73 (MHM), 21.08 vs 3.30 (WIR) and 44.63 vs 34.86 (DCI). — [VulLLM](https://aclanthology.org/2024.findings-acl.625.pdf)
- Caveat: the splits are random within each dataset (not PrimeVul time or paired splits). Devign is a training dataset; ReVeal is out-of-distribution.
- In R2Vul's later multilingual benchmark, VulLLM-CL-13B scored only 60.48 macro-F1 and lost to a CodeBERT classifier (71.51). — [R2Vul v2](https://arxiv.org/html/2504.04699v2)

**MSIVD (Yang et al., arXiv 2406.05892, 2024): multitask self-instructed fine-tuning (detect + explain + fix), combined with a GNN**
- Reported F1 0.92 on BigVul and 0.48 on PreciseBugs. The paper also notes that LineVul reaches 0.81 F1 on BigVul, i.e. BigVul random-split numbers are saturated. — [MSIVD, arXiv 2406.05892](https://arxiv.org/abs/2406.05892)
- In R2Vul's benchmark, MSIVD-CL-13B scored 66.28 macro-F1, below CodeBERT-CLS at 71.51. — [R2Vul v2](https://arxiv.org/html/2504.04699v2)

**R2Vul (Weyssow et al., arXiv 2504.04699, 2025; v2 revised): structured reasoning distillation + RLAIF via ORPO preference (valid vs. label-flipped reasoning)**
- Macro-F1 on a random stratified 80/10/10 split of an ~18k-function, 5-language dataset (C#, JS, Java, Python, C). — [R2Vul v2](https://arxiv.org/html/2504.04699v2)

  | Model | Macro-F1 |
  |---|---|
  | R2Vul-7B | 81.47 |
  | R2Vul-1.5B | 81.03 |
  | R2Vul-0.5B | 78.35 |
  | SFT-1.5B | 76.00 |
  | CodeBERT classifier | 71.51 |
  | Claude-Opus-4 | 69.19 |
  | Teacher Qwen2.5-Coder-32B CoT | 66.48 |
  | MSIVD-CL-13B | 66.28 |
  | GPT-4.1 | 61.62 |
  | VulLLM-CL-13B | 60.48 |
- The authors report 9.56%-13.93% relative gains over CodeBERT. — [R2Vul v2](https://arxiv.org/html/2504.04699v2)
- Out-of-distribution test on an external, manually labeled Java set (106 functions): R2Vul-1.5B F1 86.96, SFT 81.36, CLS 66.67. — [R2Vul v2](https://arxiv.org/html/2504.04699v2)
- There is no PrimeVul paired or time-split evaluation in R2Vul. — [R2Vul v2](https://arxiv.org/html/2504.04699v2)

**ReVD (Wen et al., Findings of ACL 2025): synthetic forward/backward reasoning + triplet SFT + curriculum online preference optimization (IPO)**
- Table 1 (Accuracy / F1 / VP-S, where VP-S is a pairwise score) on PrimeVul and SVEN. — [ReVD, arXiv 2506.07390](https://arxiv.org/html/2506.07390v1); [ACL Anthology](https://aclanthology.org/2025.findings-acl.467)

  | Method | PrimeVul Acc / F1 / VP-S | SVEN Acc / F1 / VP-S |
  |---|---|---|
  | CodeBERT (fine-tuned) | 51.03 / 16.47 / 2.07 | 51.77 / 49.79 / 3.53 |
  | UniXcoder (fine-tuned) | 49.89 / 16.48 / -0.23 | 51.90 / 43.45 / 3.80 |
  | LineVul | 49.77 / 17.70 / -0.46 | 51.90 / 40.40 / 3.80 |
  | GPT-4 CoT | 51.72 / 43.40 / 3.44 | 51.36 / 56.76 / 2.72 |
  | ReVD (Qwen2.5-Coder-7B) | 58.05 / 63.83 / 16.09 | 63.72 / 69.49 / 27.44 |
  | ReVD (Llama-3.1-8B) | 56.21 / 54.91 / 12.41 | 61.82 / 60.70 / 23.65 |
  | ReVD (StarCoder2-7B) | 55.63 / 68.74 / 11.26 | 57.34 / 53.96 / 14.68 |
- Ablation (Qwen, PrimeVul): without reasoning-SFT 51.03 / 28.28 / 2.07; without the preference stage 55.75 / 56.50 / 11.49. Preference optimization adds about 4.6 VP-S points on top of SFT. — [ReVD](https://arxiv.org/html/2506.07390v1)
- Inconsistency: the abstract claims 12.24%-22.77% accuracy gains, but the introduction and Table 1 give different relative figures. — [ReVD](https://arxiv.org/html/2506.07390v1)

**VulnLLM-R (Nie et al., arXiv 2512.07533, Dec 2025): reasoning distillation from DeepSeek-R1 + QwQ-32B into Qwen2.5-7B-Instruct**
- F1 (Table 3): Python 0.723, C/C++ 0.737, Java 0.870 (Java not in training). The no-reasoning ablation scored 0.797 / 0.784 overall but dropped to 0.497 (Python OOD-CWE) and 0.683 (C OOD-CWE). The base model scored 0.675 / 0.570 / 0.690. — [VulnLLM-R](https://arxiv.org/html/2512.07533v1)
- Table 4 gives different VulnLLM-R numbers (Python 0.730, C/C++ 0.601, Java 0.849), unexplained. Qwen3-8B and Qwen2.5-32B variants reach C/C++ 0.649 and 0.683. — [VulnLLM-R](https://arxiv.org/html/2512.07533v1)
- Comparisons against o3, Claude-3.7-Sonnet, DeepSeek-R1, QwQ, CodeQL and others are shown only in figures, so exact numbers were not extractable. No CodeBERT/UniXcoder baseline and no pairwise metric are reported. — [VulnLLM-R](https://arxiv.org/html/2512.07533v1)
- Test set is "PrimeVul-filtered": samples were filtered for context sufficiency by o3, Claude-3.7 and R1, and merged with Juliet, SVEN, SecCodePLT and ARVO. This is not the PrimeVul paired protocol. — [VulnLLM-R](https://arxiv.org/html/2512.07533v1)

**Independent evaluation (Zhang & Chen, "Right Verdict, Right Reason?", arXiv 2611.28584, Nov 2026, ECOOP 2027)**
- Setup: 481 execution-confirmed matched vulnerable/fixed pairs from BigVul and PrimeVul. — [Right Verdict, Right Reason?](https://arxiv.org/abs/2611.28584)
- Table 3 results (Acc / F1 / AUC / P-C, with chance-level P-C in parentheses):
  - VulnLLM-R-7B: 0.553 / 0.537 / 0.476 / 24.1 (chance 24.9)
  - gpt-oss-20b (high effort): 0.568 / 0.573 / 0.511 / 21.6 (chance 25.0)
  - Foundation-Sec-8B-Reasoning: 0.511 / 0.630 / 0.503 / 16.0
  - Qwen2.5-Coder-7B: 0.509 / 0.160 / 0.511 / 3.1
  - Surface-feature logistic classifier: 0.523 / 0.492 / 0.530 / 5.8
  - Claude Opus 5.5: 0.828 / 0.846 / 0.899 / 67.2
  - GPT-5.6-sol: 0.613 / 0.661 / 0.624 / 26.0

  [Right Verdict](https://arxiv.org/abs/2611.28584)
- "None of [the open-weight reasoning models] exceeds its chance level", because they give both versions the same verdict. On pairs with differing verdicts, VulnLLM-R-7B is correct more often than reversed by 10.6 points [5.0, 15.7], and gpt-oss-20b by 13.5 points. — [Right Verdict](https://arxiv.org/abs/2611.28584)
- On CVEs published after the models' training cutoffs, part of the commercial models' advantage disappears (finding F4). — [Right Verdict](https://arxiv.org/abs/2611.28584)

**PrimeVul baseline reference (Ding et al., ICSE 2025, arXiv 2403.18624)**
- Fine-tuned on PrimeVul (time split), F1 / VD-S / P-C:
  - UniXcoder: 21.43 / 89.21 / 1.60
  - CodeBERT: 20.86 / 88.78 / 1.77
  - CodeT5: 19.70 / 89.93 / 1.06
  - StarCoder2-7B: 18.05 / 89.64 / 2.30
  - CodeGen2.5-7B: 19.61 / 91.51 / 3.01

  [PrimeVul](https://arxiv.org/html/2403.18624)
- Trained on BigVul and tested on PrimeVul, F1 collapses, e.g. UniXcoder 65.46 → 1.94 and StarCoder2 68.26 → 3.09. — [PrimeVul](https://arxiv.org/html/2403.18624)
- Fine-tuned GPT-3.5 (1 epoch, 1:3 vulnerable:benign) reached P-C 1.24 and predicted 90.96% of pairs as both benign, worse than prompting. GPT-4 CoT reached P-C 12.94. Random-guess P-C is 22.70. — [PrimeVul](https://arxiv.org/html/2403.18624)

### Inferences
- Clear wins over fine-tuned UniXcoder exist on Devign and ReVeal, the user's own datasets. VulLLM-CL-13B scores +5.6 F1 on Devign and +5 F1 on ReVeal (OOD), and the 7B variant scores +1.8 and +6.9. These are random-split F1 gains, not AUC. The 7B Devign gain is marginal, and VulLLM failed to replicate as a strong baseline in R2Vul's benchmark.
- Under paired evaluation, the most credible evidence (the independent 2026 study) says a 7B distilled reasoning model (VulnLLM-R) is roughly at chance in pair discrimination. Its logprob-based AUC (0.476) is essentially the same as the user's observed gpt-oss-20b verbalized-confidence AUC (~0.47). Swapping gpt-oss-20b for VulnLLM-R-7B is therefore unlikely, on its own, to fix the "LLM adds zero signal" problem on pair-like data.
- Devign and ReVeal are not paired benchmarks, so VulnLLM-R might still add some signal there. The user's existing VulnLLM-R downloader makes this cheap to test (see Q4 for the scoring method).
- The highest-leverage training ingredient seen across papers is preference/RL on top of reasoning-SFT. ReVD's preference stage adds +4.6 VP-S; R2Vul's ORPO beats SFT by ~5 F1; VULPO's GRPO variant beats SFT/DPO/ORPO. Pure SFT on labels is weak, and fine-tuned GPT-3.5 collapsed on PrimeVul.

### Gaps
- No exact VulnLLM-R vs o3/Claude/CodeQL numbers (figures only in the HTML), and no VulnLLM-R results on Devign or ReVeal.
- LLM4Vuln, SecureFalcon and CleanVul-trained detectors were not investigated in depth within the tool budget. CleanVul appears only as OOD data in the GRPO paper below.
- "Vul-R2" (arXiv 2510.05480) is a vulnerability repair model (EM 24.83% on PrimeVul), not a detector. — [Vul-R2](https://arxiv.org/html/2510.05480v1)
- MSIVD's training details and split were not extracted.

## Q2. Exact results on realistic and paired splits: do gains hold?

### Takeaway
Gains shrink sharply but are not zero when the training method uses paired vulnerable/patched data. ReVD (VP-S 16.09 on PrimeVul) and VULPO-4B (P-Pass@1 22.60 on a time-split paired ContextVul set) are the best reported. VULPO's re-implementations of R2Vul and ReVD on its paired benchmark fell below the untrained base model. That suggests R2Vul and ReVD gains are tied to their own data and evaluation.

### Cited Findings
**VULPO (Li et al., arXiv 2511.11896, 2025/26): GRPO-variant on-policy RL with an LLM-judge reward**
- Rewards: outcome 0.4, localization 0.2 and reasoning quality 0.4 (judged by GPT-OSS-120B), plus difficulty-adaptive weighting. Base model is Qwen3-4B. — [VULPO](https://arxiv.org/html/2511.11896)
- Data: ContextVul, built from PrimeVul + SecVulEval paired functions with up to 2-hop call-graph context (libclang, Joern fallback). Split by commit date 8:1:1 into 15,760 / 1,970 / 1,970 samples. — [VULPO](https://arxiv.org/html/2511.11896)
- Table III results (P-Pass@1 / P-Pass@8 / F1). — [VULPO](https://arxiv.org/html/2511.11896)

  | Model | P-Pass@1 | P-Pass@8 | F1 |
  |---|---|---|---|
  | VULPO-4B | 22.60 | 53.91 | 40.09 |
  | DeepSeek-V3.1 | 22.50 | 42.13 | 39.64 |
  | GPT-OSS-120B | 22.16 | 40.10 | 39.58 |
  | DeepSeek-R1-0528 | 21.55 | 42.03 | 37.32 |
  | Qwen3-4B-DPO | 16.94 | 40.30 | 31.28 |
  | Qwen3-4B-ORPO | 15.13 | 36.24 | 28.79 |
  | Qwen3-4B-SFT | 14.95 | 34.62 | 28.26 |
  | Qwen3-4B-MARCO (GRPO) | 11.52 | 27.51 | 22.24 |
  | Qwen3-4B base | 7.45 | 22.23 | 15.20 |
  | Qwen3-4B-R2Vul | 5.36 | 25.99 | 12.26 |
  | VulnLLM-R-7B (zero-shot) | 5.08 | 21.73 | 12.18 |
  | Qwen3-4B-ReVD | 4.63 | 23.25 | 11.28 |
  | R2Vul-7B (original) | 3.78 | 16.24 | 14.68 |
- The authors attribute the R2Vul/ReVD drop to ground-truth leakage into teacher prompts and noisy trace filtering. — [VULPO](https://arxiv.org/html/2511.11896)
- Ablation: removing the difficulty scaling cuts P-Pass@1 from 22.60 to 19.91. — [VULPO](https://arxiv.org/html/2511.11896)

**GRPO for VD (Simoni et al., arXiv 2507.03051, 2025)**
- Models: Qwen2.5-3B, Llama-3B and Llama-8B. Trained on balanced DiverseVul (random 80/20). CleanVul and BigVul are OOD. — [GRPO-VD](https://arxiv.org/html/2507.03051v1)
- Accuracy / macro-F1:

  | Model | DiverseVul | CleanVul | BigVul |
  |---|---|---|---|
  | Qwen2.5-3B, GRPO | 0.67 / 0.67 | 0.57 / 0.57 | 0.64 / 0.64 |
  | Qwen2.5-3B, SFT | 0.50 / 0.38 | 0.49 / 0.39 | 0.51 / 0.43 |
  | Qwen2.5-3B, zero-shot | 0.52 / 0.52 | 0.57 / 0.57 | 0.59 / 0.54 |
  | Llama-8B, GRPO | 0.62 / 0.60 | 0.58 / 0.57 | 0.62 / 0.62 |
  | Llama-8B, SFT | 0.62 / 0.60 | 0.50 / 0.40 | 0.52 / 0.44 |

  [GRPO-VD](https://arxiv.org/html/2507.03051v1)
- GRPO mainly fixes the not-vulnerable class: Qwen non-vulnerable recall on DiverseVul goes from 0.07 to 0.61. — [GRPO-VD](https://arxiv.org/html/2507.03051v1)
- Code is listed as github.com/winstonsmith1897/GRPO-Vulnerability-Detection. — [alphaXiv resources](https://alphaxiv.org/resources/2507.03051v1)

**Other paired-split reference points (from search snippets, not verified in full text)**
- VulInstruct reports 17.2% correct pairs on PrimeVul paired. — [VulInstruct, arXiv 2511.04014](https://arxiv.org/pdf/2511.04014)
- ReasonVul (multi-agent) reports PairAcc 40.00% and F1 72.52% on PrimeVul. — [snippet via search; source list included SBVUL arXiv 2611.06135](https://arxiv.org/html/2611.06135)

**Negative findings**
- VulnLLM-R tried DPO and saw no gain; on its benchmarks DPO even dropped F1 to 0.601 / 0.631 / 0.607. Online RL (GRPO/DAPO) increased false negatives, so they shipped SFT-only. — [VulnLLM-R](https://arxiv.org/html/2512.07533v1)
- Independent L1 pair test: VulnLLM-R-7B P-C 24.1 vs chance 24.9 (Q1 above). — [Right Verdict](https://arxiv.org/abs/2611.28584)

### Inferences
- The conflicting DPO/RL results (VULPO: RL helps a lot; VulnLLM-R: RL/DPO hurt) suggest reward design matters. VULPO combines a judged reasoning/localization reward with difficulty weighting. A plain correctness reward with GRPO (the MARCO baseline in VULPO) underperformed SFT.
- Training on paired pre/post-patch data with a pairwise objective is the common factor of the methods that show any pairwise gain (ReVD, VULPO). For the user, a pair-aware training set (PrimeVul paired, or Devign/ReVeal commit pairs if available) matters more than model size. This is consistent with the user's finding that 120b ≈ 20b, and with VULPO-4B ≈ GPT-OSS-120B.
- Even the best paired result (P-Pass@1 ~22.6) roughly equals random-guess P-C (22.7 in PrimeVul's table). Realistic absolute performance remains low.

### Gaps
- No method reports results on Devign/ReVeal with AUC, the user's metric. Cross-paper comparisons use different datasets and metrics.
- VULPO's code repository was not found, only the HF collection link. The exact VULPO compute (GPU-hours) is not reported.

## Q3. Training recipes: data, LoRA vs full, compute

### Takeaway
All recipes fit 7B-scale and most report single-node GPUs. R2Vul is the most Kaggle-friendly: LoRA r=8, one H100, 0.5B-7B students. ReVD used full SFT plus LoRA preference optimization on 8×A100-40GB. VULPO needs a 12-GPU RL setup plus a 120B judge, which is out of Kaggle reach. VulnLLM-R did not publish hyperparameters in the extracted text.

### Cited Findings
- **R2Vul** — [R2Vul v2](https://arxiv.org/html/2504.04699v2)
  - Students: Qwen2.5-Coder-Instruct 0.5B/1.5B/7B; teacher: Qwen2.5-Coder-32B-Instruct.
  - Data: ~18k NVD-linked fix-commit functions. GPT-4o re-scored labels, keeping only score-4 vulnerable functions; a manual check found ~93% of labels correct.
  - Training: ORPO with λ=0.3 and LoRA (r=8, α=16, dropout 0.05). LR 3e-4, batch 2, up to 5 epochs; TRL 0.12.1 on one H100-80GB. Training time not reported.
- **ReVD** — [ReVD](https://arxiv.org/html/2506.07390v1)
  - Teacher: Qwen2.5-32B-Coder-Instruct; ~28k reasoning samples (3× with task decomposition).
  - Triplet SFT (full fine-tuning): 3 epochs, LR 1e-5, batch 8, max length 2048.
  - COPO (LoRA, IPO β=0.1): 3 rounds × 1 epoch, LR 5e-6.
  - Hardware: 8×A100-40GB with LLaMA-Factory. Time not reported.
- **VulnLLM-R** — [VulnLLM-R](https://arxiv.org/html/2512.07533v1)
  - Data: ~11.8k function-level training samples (Python 2,562; C/C++ 9,233), plus 1,172 project-level samples.
  - Teachers DeepSeek-R1 + QwQ-32B; 8 samples per teacher, keeping the shortest correct chain.
  - 73 "constitutions" for 19 CWEs add ~30% more correct data.
  - Second-stage training on R1-summarized chains cuts test time ~80%; average output is ~362 tokens.
  - 20-token n-gram decontamination between train and test.
- **VULPO** — [VULPO](https://arxiv.org/html/2511.11896)
  - Cold-start SFT on DeepSeek-R1-0528 traces: 3 epochs, LR 1e-5, 4×A100-80GB.
  - RL: LR 1e-6, 8 rollouts, KL β=0, 8 policy GPUs + 4 rollout GPUs.
  - Judge: GPT-OSS-120B. GPT-OSS-20B made 5 errors per 400 judgments vs 1 for 120B, and GPT-4.1-mini made 59.
- **GRPO-VD** — [GRPO-VD](https://arxiv.org/html/2507.03051v1)
  - 1 epoch; 12 samples per group; LR 5e-6; β=1e-6; paged AdamW 8-bit; max prompt 4,000 / answer 3,000 tokens.
  - Hardware and LoRA use not stated.
- **VulLLM**: data from PatchDB + DiverseVul + Devign, with GPT-4 interpretations, and 10% identifier substitution. — [VulLLM](https://aclanthology.org/2024.findings-acl.625.pdf)

### Inferences
- A Kaggle-feasible recipe (2×T4 or P100, 16GB) is QLoRA SFT, then ORPO/DPO, on Qwen2.5-Coder-1.5B/7B in the R2Vul style. Reasoning traces could be generated by gpt-oss-20b via API, and VULPO's judge table suggests gpt-oss-20b is a reasonably accurate judge.
- Use paired functions and avoid putting the ground-truth label or CVE text into the teacher prompt. VULPO blames that leakage for R2Vul/ReVD collapsing on its benchmark.
- GRPO with vLLM rollouts on a 7B model is likely infeasible on Kaggle T4s. A 1.5B-3B model with 4-8 rollouts might fit, but no paper reports such a low-resource run.

### Gaps
- No paper reports GPU-hours.
- No reported Kaggle-class (T4/P100) reproductions were found.

## Q4. Do fine-tuned LLM verifiers produce calibrated probabilities usable for fusion?

### Takeaway
Only R2Vul explicitly builds a probability: a sigmoid of the length-normalized log-likelihood margin between two class-template prefixes. It uses that score for threshold tuning to control FPR, not for ECE-calibrated fusion. The independent 2026 study read logP(YES)−logP(NO) from VulnLLM-R-7B and got AUC 0.476 on matched pairs, i.e. no ranking signal there.

### Cited Findings
- R2Vul calibration: confidence = sigmoid(margin of length-normalized log-likelihoods of "vulnerable" vs "safe" templates). — [R2Vul v2](https://arxiv.org/html/2504.04699v2)
  - Python test set at 1:1 balance: threshold τ=0.73 cuts FPR from 28% to 20% (F1 79.39 → 78.57).
  - At 1:10 imbalance, τ=0.96 gives FPR 20% at recall 66.03.
  - No ECE or Brier score is reported.
- The TitanCA CVE-discovery pipeline reportedly uses R2Vul as a reasoning-based filter with a calibration step that scores confidence from the model's reasoning likelihoods. — [TitanCA, arXiv 2604.17860](https://arxiv.org/pdf/2604.17860) (from search snippet; not read in full)
- AUC from log P(YES) − log P(NO) at the answer token (open-weight models) on 481 matched pairs. — [Right Verdict](https://arxiv.org/abs/2611.28584)
  - VulnLLM-R-7B 0.476; Qwen3.6-27B 0.525; gpt-oss-20b 0.511 (stated confidence); Qwen2.5-Coder-32B 0.518.
  - Claude Opus 5.5 reached 0.899.
  - The authors note that "the scores of the two versions of a function differ far less than those of different functions, so a single threshold gives both versions the same label."
- VulnLLM-R's model card defines no fixed final-answer label format; the example prompt asks for "reasoning followed by the final answer" (chat template, max_new_tokens=512 in the example). The paper says the output ends with a verdict plus a CWE type. — [HF model card](https://huggingface.co/UCSB-SURFI/VulnLLM-R-7B); [VulnLLM-R](https://arxiv.org/html/2512.07533v1)
- VulnLLM-R found that majority voting over 1-16 samples gave minimal gains. — [VulnLLM-R](https://arxiv.org/html/2512.07533v1)

### Inferences
- For fusion with the UniXcoder stack, the most usable score is the token log-prob margin at the answer position, as in R2Vul and the 2026 study. Verbalized confidence is the less usable option. The user's verbalized-confidence AUC ~0.47 matches the logprob AUC that independent researchers measured for VulnLLM-R on pairs, so expectations should be low unless the model is fine-tuned on the user's distribution.
- A cheap experiment: run VulnLLM-R-7B locally (vLLM or HF) on the Devign/ReVeal validation set and force a final "yes"/"no" token after the reasoning. Take logP(yes)−logP(no), compute its AUC, then add it as a stacking feature. Treat Platt/isotonic calibration on validation as necessary; no paper shows these scores are calibrated out of the box.
- Because gpt-oss-20b is API-only, its token logprobs may not be exposed. A locally fine-tuned 1.5B-7B verifier is the practical route to a real probability.

### Gaps
- No paper reports ECE or Brier for fine-tuned VD LLMs.
- No paper reports fusing an LLM verifier score with a small-model classifier score on Devign/ReVeal.

## Q5. Public checkpoints and licenses

### Takeaway
VulnLLM-R-7B is the best-documented public checkpoint (Apache-2.0, HF). VULPO artifacts are in an HF collection. R2Vul promises checkpoints and data in its replication package. ReVD releases code and data but no weights were mentioned. VulLLM release status was not verified.

### Cited Findings
- **VulnLLM-R-7B** — [HF model card](https://huggingface.co/UCSB-SURFI/VulnLLM-R-7B); [mirror](https://huggingface.co/Virtue-AI-HUB/VulnLLM-R-7B/blob/main/README.md)
  - Location: huggingface.co/UCSB-SURFI/VulnLLM-R-7B (also mirrored at Virtue-AI-HUB/VulnLLM-R-7B).
  - License: Apache-2.0; fine-tuned from Qwen2.5-7B-Instruct.
  - Card stats: ~1,188 downloads/month, 12 quantizations, HF demo Space.
  - Code: github.com/ucsb-mlsec/VulnLLM-R.
- **VULPO**: HF collection huggingface.co/collections/Leopo1d/vulpo. Contents and license were not verified. — [VULPO](https://arxiv.org/html/2511.11896)
- **R2Vul**: code at github.com/martin-wey/R2Vul. The paper says "we share our datasets, model checkpoints, and code in our replication package". HF hosting and license were not confirmed. — [R2Vul v2](https://arxiv.org/html/2504.04699v2)
- **ReVD**: code and data at github.com/Xin-Cheng-Wen/PO4Vul. No license or weights were stated in the paper. — [ReVD](https://arxiv.org/html/2506.07390v1)
- **GRPO-VD**: code at github.com/winstonsmith1897/GRPO-Vulnerability-Detection; no weights confirmed. — [alphaXiv](https://alphaxiv.org/resources/2507.03051v1)
- **Foundation-Sec-8B-Reasoning**: a security-tuned 8B reasoning model, evaluated in the 2026 study with P-C 16.0 and AUC 0.503 on pairs. — [Right Verdict](https://arxiv.org/abs/2611.28584)

### Inferences
- Priority for the user:
  1. Score Devign/ReVeal validation with VulnLLM-R-7B using logprob scoring; it is cheap and already downloadable, but expect weak signal.
  2. Fine-tune a 1.5B-7B Qwen2.5-Coder verifier R2Vul-style (QLoRA SFT + ORPO on paired data, traces from gpt-oss-20b, no label leakage in the teacher prompt).
  3. Consider VULPO-style RL only if multi-GPU compute is available.

### Gaps
- Licenses for the R2Vul, ReVD, VULPO and VulLLM artifacts were not verified.
- Whether VulLLM or MSIVD weights are on HF was not checked.
