# Context-enrichment, data quality, and evaluation practice for LLM / hybrid vulnerability detection (2023-2026)

Scope note: VulGuardVN-v2 currently works at function level only. It uses Joern CPG facts (sinks, backward slices, guard checks V1-V3), a two-pass prompt (hypothesis, then verify), and 1-2 retrieved contrastive vulnerable/patched demos. On Devign/ReVeal the LLM confidence AUC is about 0.47 and fusion adds about 0 AUC. Every number below comes from a source fetched or searched in this session. Claims I could not verify are listed under Gaps.

## Q1. Which context-enrichment techniques (interprocedural context, CPG slicing, CWE/spec retrieval, structured prompting) report gains, and on which datasets and splits?

### Takeaway
The reported gains are real but modest, and they come almost entirely from **selected/profiled** context plus **reasoning-trained or fine-tuned** models, usually evaluated on PrimeVul. Appending raw callers, callees and globals does not help and often hurts. Even the best context-rich systems reach only about 10-37% "both correct" pairs on PrimeVul-style paired data.

### Cited Findings
**Why function-level context is insufficient**
- Risse, Liu & Böhme, "Top Score on the Wrong Exam: On Benchmarking in Machine Learning for Vulnerability Detection" (arXiv 2408.12986, v1 Aug 2024, v2 Apr 2025):
  - For more than 90% of functions in widely used ML4VD datasets, vulnerability cannot be decided without outside context.
  - The authors built a vulnerable calling context for 82 of 90 functions labeled secure.
  - High scores are achievable from word counts alone.
  - The v2 survey covers 81 papers, which mostly treat the task as isolated-function classification.
  - [arXiv 2408.12986](https://arxiv.org/abs/2408.12986v2)

**CORRECT (Li et al., "Everything You Wanted to Know About LLM-based Vulnerability Detection But Were Afraid to Ask", arXiv 2504.13474, Apr 2025)** — [arXiv HTML](https://arxiv.org/html/2504.13474v1)
- Benchmark: 2,000 vulnerable/patched pairs from 364 projects and 99 CWEs, drawn from MoreFixes, PrimeVul and ReposVul. A manual audit of 50 pairs found 98% label accuracy. The main evaluation uses 400 pairs.
- Context size: the shared context averages 13,249 tokens, against 1,574 tokens for the function alone.
- Without context, most models score F1 0.5-0.6, accuracy 0.5-0.55 and precision 0.5-0.55.
- With context in Strict Mode:
  - DeepSeek-R1 reaches about 67% accuracy and F1 about 0.6; smaller models reach about 60%.
  - Precision is about 0.8 for DeepSeek-R1 and DeepSeek-V3.
  - The pair proportion "(1,1)", meaning both functions flagged vulnerable, falls to about 10%.
  - DeepSeek-R1's "(1,0)" proportion, meaning both functions correct, is 37% against a 25% random baseline.
- Model size shows a clear scaling trend only when context is given.
- CWE F1 is about 0.7 for common fixed-pattern CWEs: CWE-682 0.713, CWE-664 0.700. Rare CWEs are worse, e.g. CWE-697 0.400.
- The contributions of callers, callees, types and globals are not ablated separately.
- Prompts include CWE descriptions, but their effect is not ablated.

**CPRVul (Li et al., "Beyond Function-Level Analysis: Context-Aware Reasoning for Inter-Procedural Vulnerability Detection", arXiv 2602.06751, Feb 2026, SMU/Monash/GovTech)** — [arXiv HTML](https://arxiv.org/html/2602.06751v1)
- Method: build a repository CPG, extract callers, callees and globals, have an LLM write security profiles and relevance scores, keep only high-impact context, and fine-tune Qwen2.5-Coder with reasoning traces.
- Raw context hurts fine-tuned encoders (accuracy, 1,024-token limit):

| Dataset | Model | Function only | With raw context | Change |
|---|---|---|---|---|
| PrimeVul | UniXcoder | 56.65% | 54.22% | -2.43 |
| CleanVul | UniXcoder | 58.24% | 55.22% | -3.02 |
| CleanVul | CodeBERT | 55.67% | 52.47% | -3.20 |

- Profiled (Phase I) context alone also does not help encoders. On PrimeVul, UniXcoder goes from 56.65% to 55.26%.
- CPRVul accuracy with Qwen2.5-32B: PrimeVul 67.78% (+11.13 pts over UniXcoder; +12.61 over VulTrial's 55.17%), TitanVul 73.76%, CleanVul 64.94%.
- Ablation (Qwen2.5-32B, PrimeVul):
  - Removing context selection (Phase I) costs only -2.36 (65.42%).
  - Removing reasoning fine-tuning (Phase II) drops accuracy to 52.54% (-15.24).
  - **Most of the gain comes from reasoning-trained fine-tuning, not from the context itself.**
- Zero-shot GPT-4.1 scores 51.90% on PrimeVul, 51.36% on TitanVul and 51.83% on CleanVul, which is near chance on what appear to be balanced sets.
- No pairwise metrics are reported.

**LLMxCPG (Lekssays et al., USENIX Security 2025, arXiv 2507.16585)** — [USENIX](https://www.usenix.org/conference/usenixsecurity25/presentation/lekssays); [arXiv](https://arxiv.org/html/2507.16585v1); [artifact appendix](https://www.usenix.org/system/files/usenixsecurity25-appendix-lekssays.pdf)
- Method: an LLM writes Joern CPG queries to extract a minimal slice, and a second LLM classifies the slice.
- The slices are 67.84-90.93% smaller than the original code.
- The abstract reports +15-40% F1 and +9-27% accuracy over SOTA baselines on function-level real-world detection.
- The artifact reports project-level results: ReposVul F1 0.610 / accuracy 0.634, and the post-cutoff PKCO-25 set F1 0.617 / accuracy 0.600.
- Code: github.com/qcri/llmxcpg.

**VulInstruct (Zhu et al., "Specification-Guided Vulnerability Detection with LLMs", arXiv 2511.04014, Nov 2025)** — [arXiv](https://arxiv.org/html/2511.04014v1); [PDF](https://arxiv.org/pdf/2511.04014)
- Method: retrieve similar historical cases and provide security specifications mined from patches. General specs come from high-quality patches across projects; domain-specific specs come from repeated violations in the target repository.
- PrimeVul results:
  - F1 45.0% (+32.7% relative) and recall 37.7% (+50.8%).
  - Pairwise: +32.3% relative, with 17.2% P-C accuracy. The pairwise evaluation is strict and requires a correct prediction plus valid reasoning.
- It found a new bug, CVE-2025-56538.
- No independent replication was found.

**ReVD (Wen et al., "Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data", Findings of ACL 2025, arXiv 2506.07390)** — [ACL Anthology](https://aclanthology.org/2025.findings-acl.467); [arXiv](https://arxiv.org/html/2506.07390v1)
- Method: synthesize forward/backward reasoning for vulnerable and fixed code, then run triplet SFT and curriculum online preference optimization.
- Reported SOTA on PrimeVul and SVEN, with "12.24%-22.77% improvement in accuracy". The ACL text gives +9.08% accuracy, +10.33% F1 and +18.15% VP-Score against baselines.
- The absolute PrimeVul numbers were not captured.
- Motivation statistic: GPT-4 fails to distinguish 78.62% of PrimeVul vulnerable/fixed pairs.
- Code: github.com/Xin-Cheng-Wen/PO4Vul.

**Repository-level benchmarks**
- JitVul (ACL 2025, 879 CVEs, 91 vulnerability types, pairwise): ReAct agents with interprocedural context outperform plain LLMs at telling vulnerable from benign code. Search snippet only; numbers not captured. — [arXiv 2503.03586](https://arxiv.org/abs/2503.03586)
- VulEval: repository-level frameworks outperform function-level ones, and the data split, retrieval technique and vulnerability type substantially affect results. — [SMU repository](https://ink.library.smu.edu.sg/sis_research/11291)

### Inferences
- The strongest evidence (CPRVul ablations) suggests that adding callers and callees to VulGuardVN's prompt for a frozen, prompted LLM will probably not, alone, lift AUC from about 0.47. Context helped when it was (i) filtered for security relevance and (ii) consumed by a model fine-tuned on reasoning traces.
  - For a prompted model, CORRECT is the best evidence for context. Its gains show up mainly in **precision** and in a lower "both flagged vulnerable" rate, and mainly with strong reasoning models (DeepSeek-R1).
- VulGuardVN's backward slices and guard checks resemble LLMxCPG slicing. LLMxCPG's headline gains are relative to weak baselines and on ReposVul/PKCO-25, not Devign/ReVeal. So the current null result on Devign/ReVeal does not contradict LLMxCPG.
- Risse et al.'s finding (>90% context-dependent; 82/90 "secure" functions can be made vulnerable) means Devign/ReVeal function labels are partly unlearnable from the function alone. That supports moving emphasis to paired, context-available benchmarks.
- The biggest lever in this literature is reasoning-trained fine-tuning (ReVD, CPRVul Phase II), not prompt-side context.

### Gaps
- None of these papers ablates callers vs callees vs globals for a prompted (non-fine-tuned) LLM.
- I found no published LLM context-enrichment result on Devign or ReVeal specifically. Most of this work moved to PrimeVul, CleanVul, TitanVul or ReposVul.
- Not captured:
  - ReVD's absolute PrimeVul accuracy and pairwise numbers.
  - JitVul's numeric results.
  - LLMxCPG's per-dataset function-level table.
  - R2Vul (arXiv 2504.04699) and VulnLLM-R (arXiv 2512.07533) pairwise numbers. A third-party Hugging Face replication claims VulnLLM-R hits a "single-function ceiling" on out-of-distribution PrimeVul pairs; I did not verify its figures. — [Euraika dataset card](https://huggingface.co/datasets/Euraika/VulnLLM-R-Interprocedural-Ceiling)

## Q2. How much do cleaned labels change results, and do LLMs look better on cleaner data?

### Takeaway
Label noise in older datasets is severe: only 24-60% of "vulnerable" labels are correct in CodeXGLUE/Devign, BigVul, CrossVul, CVEFixes and DiverseVul, against 86-92% in PrimeVul. Cleaning mostly **deflates** fine-tuned model scores, which BigVul had inflated. LLMs do not become good on clean data, but evaluations on clean, context-rich pairs make them look **less bad**: precision rises from about 0.5 to 0.8 in CORRECT.

### Cited Findings
**PrimeVul (Ding et al., "Vulnerability Detection with Code Language Models: How Far Are We?", ICSE 2025, arXiv 2403.18624)** — [arXiv HTML v2](https://arxiv.org/html/2403.18624v2)

Manual label accuracy, 50 vulnerable functions each (Table I):

| Dataset | Correct labels |
|---|---|
| SVEN | 94.0% |
| CodeXGLUE (= Devign) | 24.0% |
| VulnPatchPairs | 36.0% |
| BigVul | 25.0%* |
| CrossVul | 47.8%* |
| CVEFixes | 51.7%* |
| DiverseVul | 60.0%* |
| PrimeVul-OneFunc | 86.0% |
| PrimeVul-NVDCheck | 92.0% |

\* taken from Chen et al. (DiverseVul)

- Exact-duplicate rates: BigVul 12.7%, CVEFixes 18.9%, CodeXGLUE 0.6%, DiverseVul 3.3% (de-duplicated), PrimeVul 0.0%.
- Size: 6,968 vulnerable and 228,800 benign functions, 755 projects, 140 CWEs.
  - Train: 5,574 vulnerable / 178,853 benign. Dev: 699 / 24,731. Test: 695 / 25,216.
  - Paired subset: 5,480 pairs (train 4,354 / dev 562 / test 564).
  - Chronological split.

StarCoder2, train set vs test set (Table V):

| Train → Test | F1 | VD-S | P-C |
|---|---|---|---|
| BigVul → BigVul | 68.26 | 69.14 | 35.23 |
| BigVul → PrimeVul | 3.09 | 96.83 | 0.89 |
| PrimeVul → PrimeVul | 18.05 | 89.64 | 2.30 |

**Other label-quality and cleaning studies**
- Croft, Babar & Kholoosi, "Data Quality for Software Vulnerability Datasets" (ICSE 2023), four SOTA datasets:
  - 20-71% of labels inaccurate, and inaccurate labels change performance by up to 65%.
  - 17-99% duplicated data points; 0-47% inconsistent labels.
  - The per-dataset breakdown, e.g. Devign, was not captured.
  - [arXiv 2301.05456](https://arxiv.org/abs/2301.05456v1)
- CleanVul / VulSifter (Li et al., arXiv 2411.17274, several versions up to v7 Sep 2025):
  - Existing datasets contain about 40-75% noise.
  - An LLM plus heuristics scores each commit change 0-4 for relevance; the threshold trades dataset size against cleanliness.
  - Correct vulnerable-function rate rises from 28.7% to 90.6%; identification F1 is 0.82.
  - Dataset size is 8,198 functions in the abstract and 11,632 in a later version, so check the version before citing.
  - CodeBERT fine-tuned on CleanVul and tested on PrimeVul reaches 58.09% accuracy, against 56.61% when trained on PrimeVul.
  - [arXiv 2411.17274 v6](https://arxiv.org/html/2411.17274v6)

**Noise-robust training**
- Croft et al., "Noisy Label Learning for Security Defects" (MSR 2022, just before the 2023 window): a two-stage noise-cleaning method improves baseline AUC by up to 8.9% and recall by up to 23.4%. — [arXiv 2203.04468](https://arxiv.org/abs/2203.04468)
- "An Empirical Study on Noisy Label Learning for Program Understanding" (ICSE 2024, includes Devign-based vulnerability detection):
  - Small models trained from scratch are prone to label noise.
  - Confident Learning and TracIn detect mislabeled samples, with Confident Learning often strongest on the human-verified subset.
  - [arXiv 2307.08990](https://arxiv.org/pdf/2307.08990)
- Gong & Massacci, "When in Doubt Throw It Out" (ICSE-NIER 2025): confident learning assumes a reasonably good base model, which vulnerability detectors are not, so it is less reliable here. — [PDF](https://fabiomassacci.github.io/docs/Confidence_Learning_for_Vulnerabilities_icse.pdf)
- "Automatic Data Labeling for Software Vulnerability Prediction Models: How Far Are We?" (2024) compares Confident Learning, centroid-based removal and domain-specific removal on auto-labeled data. Numbers not captured. — [arXiv 2407.17803](https://arxiv.org/pdf/2407.17803)

**LLMs on clean, context-rich pairs**
- In CORRECT (98% label accuracy, with context), precision rises from 0.5-0.55 to about 0.8 for DeepSeek-R1/V3. — [arXiv 2504.13474](https://arxiv.org/html/2504.13474v1)
- Zero-shot GPT-4.1 is still about 52% accuracy on PrimeVul, TitanVul and CleanVul. — [CPRVul](https://arxiv.org/html/2602.06751v1)

### Inferences
- CodeXGLUE/Devign has about 24% vulnerable-label accuracy in PrimeVul's audit. An LLM that "reasons correctly" would be scored against labels that are wrong about 3/4 of the time on the positive class.
- An LLM confidence AUC of about 0.47 on Devign/ReVeal is therefore weak evidence that the LLM layer is useless. Devign/ReVeal is a poor yardstick for the LLM component.
- Fine-tuned baselines (GNN/encoder) look much worse once labels are clean and splits are chronological (F1 68 → 18). On PrimeVul the gap between VulGuardVN's learned component and the LLM component may narrow. This is a hypothesis, not something any source reports.
- Noise-robust training (CL, co-teaching) only helps the trained component. The cited evidence of gains is limited (MSR 2022: up to +8.9% AUC). Using CleanVul or PrimeVul as training data, or CleanVul-style LLM relabeling, has better-documented effects than CL on Devign.

### Gaps
- No direct Devign or ReVeal noise percentage beyond the 24% CodeXGLUE/Devign audit and Croft's 20-71% range. A ReVeal-specific audit was not found.
- No 2023-2026 vulnerability-detection paper using co-teaching was found.
- No paper was found that compares the same LLM prompt on BigVul vs PrimeVul with pairwise metrics.

## Q3. Which evaluation metrics and protocols are recommended, and what LLM pairwise accuracies are reported?

### Takeaway
PrimeVul's protocol, which most 2025-2026 work now adopts, is:
- a chronological split with de-duplication;
- **VD-S**: FNR at FPR ≤ 0.5%;
- **paired metrics**: P-C (pair correct), P-V (both flagged vulnerable), P-B (both flagged benign), P-R (reversed).

Reported LLM P-C on PrimeVul paired test sets is very low: about 5-13% for GPT-3.5/4 and about 8-22% for current RAG, reasoning or fine-tuned systems. Only oracle knowledge or rich context reaches about 37-73%.

### Cited Findings
**PrimeVul metric definitions and baselines** — [PrimeVul](https://arxiv.org/html/2403.18624v2)

GPT models on PrimeVul paired test pairs (Table VIII):

| Model | Method | P-C | P-V | P-B | P-R |
|---|---|---|---|---|---|
| GPT-3.5 | Two-shot | 5.67 | 13.83 | 77.84 | 2.66 |
| GPT-3.5 | CoT | 6.21 | 4.79 | 83.51 | 5.50 |
| GPT-3.5 | Fine-tuned | 1.24 | 5.32 | 90.96 | 2.48 |
| GPT-4 | Two-shot | 5.14 | 71.63 | 21.45 | 1.77 |
| GPT-4 | CoT | 12.94 | 54.26 | 24.47 | 8.33 |
| Random guess | – | 22.70 | 26.24 | 26.42 | 24.65 |

- Fine-tuned code LMs on PrimeVul paired, P-C: CodeT5 1.06, CodeBERT 1.77, UnixCoder 1.60, StarCoder2 2.30, CodeGen2.5 3.01.
- The paper does not report VD-S for the GPT models.

**Later systems on PrimeVul paired**

| System | Paired result | Source |
|---|---|---|
| VulInstruct | 17.2% P-C | [VulInstruct](https://arxiv.org/pdf/2511.04014) |
| LLM baselines in ReasonVul | about 8.28-10.57% PairAcc | [ReasonVul](https://arxiv.org/pdf/2605.18153) |
| MiST (Qwen3) | about 10.3-16.4 before task fine-tuning, 16.3-21.3 after | [MiST](https://arxiv.org/pdf/2609.18496) |
| VulTriage | 147 correct pairs on the test set; claims to beat VulTrial on P-C | [VulTriage](https://arxiv.org/pdf/2605.09461) |
| DeepSeek-R1 with CORRECT context | 37% "(1,0)", against 25% random (CORRECT's 2,000-pair set, not PrimeVul alone) | [CORRECT](https://arxiv.org/html/2504.13474v1) |

**Unified open-weight benchmark: "Retrieve, Reproduce, Reveal: Dissecting Retrieval-Augmented Software Vulnerability Detection"** (Kaniewski et al., arXiv 2609.37669, Sep 2026, PrimeVul Paired) — [arXiv](https://arxiv.org/html/2609.37669)
- Mean pairwise accuracy across a common set of models: Vul-RAG 0.20, VulTriage 0.12, SVD-Bench 0.08, Llama-VD 0.08, **GRACE 0.03**.
- Best configuration per system: Vul-RAG 0.22, VulTriage 0.21, GRACE 0.17, Llama-VD 0.15, SVD-Bench 0.12.
- **Some GRACE configurations reach F1 0.67 with pairwise accuracy 0.00-0.01.** F1 hides the failure.
- VulTriage reported 0.34 pairwise with GPT-4o in its own paper.
- Llama-VD reproduces at about 0.67 accuracy on PrimeVul but with recall just above 1%, a degenerate "always benign" behavior.

### Inferences
- VulGuardVN-v1 was renamed from "GRACE-improve". The independent benchmark showing GRACE at 0.03 mean pairwise accuracy (with F1 up to 0.67) is a direct warning: F1 or AUC on unpaired data can hide a model that cannot tell a vulnerable function from its patch.
- Adopting PrimeVul paired P-C (plus P-V/P-B/P-R to show the bias direction) and VD-S as primary metrics would make VulGuardVN comparable to the 2025-2026 literature.
- Realistic targets for a prompted pipeline: about 10-22% P-C (SOTA RAG/reasoning). The empirical random baseline is 22.7%, and the theoretical one is 25%. Even "SOTA" numbers are around or below random on P-C. Reporting P-R (reversed) and P-V vs P-B shows whether the LLM is just biased one way, as GPT-4 is toward "vulnerable" (P-V 71.6) and GPT-3.5 toward "benign" (P-B 77.8–91).
- An AUC near 0.5 on Devign/ReVeal and near-random P-C on PrimeVul paired would be consistent with the literature, not an anomaly.

### Gaps
- No VD-S numbers were found for prompted LLMs, since VD-S needs a calibrated score.
- No paper was found that reports LLM confidence AUC on PrimeVul paired. That is a measurement VulGuardVN could contribute.
- The VulTriage test-set denominator is unclear: PrimeVul v2 has 564 test pairs, and other versions differ.

## Q4. Does retrieval of demos / RAG help or hurt on these benchmarks?

### Takeaway
Knowledge-level RAG (CVE-derived cause/fix knowledge, specifications) gives the most consistent but still small gains on paired benchmarks: Vul-RAG at 0.20-0.26 pairwise, VulInstruct at 17.2% P-C. Code-level demo retrieval is weaker. Even oracle retrieval only gets mean pairwise accuracy to about 0.51, so retrieval is not the bottleneck; detection and reasoning are.

### Cited Findings
**Vul-RAG (Du et al., arXiv 2406.11147, v1 Jun 2024 to v3)** — [arXiv v3](https://arxiv.org/html/2406.11147v3); [v1](https://arxiv.org/html/2406.11147v1)
- Method: an LLM extracts multi-dimensional knowledge (cause, fix) from CVEs, retrieves it by functional semantics, and reasons over it.
- PairVul: 4,314 vulnerable/patched pairs, 2,073 CVEs.
- v1 results: accuracy 0.61 and pairwise accuracy 0.21, a relative +12.96% and +110% over the best baseline (LLMAO).
- v3 frames the result as +16-27% pair accuracy over code-level RAG and +22-26% over fine-tuning baselines.

**Independent reproduction and the unified benchmark** — [Kaniewski et al. 2026](https://arxiv.org/html/2609.37669)
- Vul-RAG reproduces at pairwise 0.26 on PairVul with Qwen2.5-Coder-32B, matching the reported 0.26.
- On PrimeVul Paired it is the best of six systems, but only at 0.20 mean and 0.22 best.
- A 75% knowledge-base subset outperformed the full knowledge base. The authors hypothesize that near-duplicate or low-relevance entries dilute retrieval.
- Retrieval effectiveness is more sensitive to **query abstraction** (how the code is summarized before retrieval) than to which generator built the knowledge base.
- Oracle knowledge derived from the test set gives mean pairwise 0.51 on PrimeVul (0.66 for Qwen2.5-Coder-14B; best detector Qwen3.6-27B at 0.73 on PrimeVul and 0.72 on PairVul). Oracle MRR is 0.93-0.95, yet detection stays low. Quote: "oracle retrieval does not solve vulnerability detection."
- Published RAG4SVD results do not consistently transfer to open-weight models, and the best backbone is pipeline-specific.
- The paper reports no no-retrieval baseline, so "RAG vs no RAG" is not quantified there.

**Other evidence**
- VulInstruct retrieves specifications mined from patches and similar historical cases: +32.3% relative pairwise and 17.2% P-C on PrimeVul. — [VulInstruct](https://arxiv.org/pdf/2511.04014)
- PatchCon (arXiv 2611.06583) reports Vul-RAG at 0.1950 pairwise accuracy with GPT-4o on a different PairVul split, and says it is not directly comparable. — [PatchCon](https://arxiv.org/pdf/2611.06583)
- PrimeVul's two-shot prompting gave GPT-3.5 P-C 5.67 and GPT-4 P-C 5.14, both below CoT (6.21 and 12.94). For GPT-4, two-shot demos increased the "both vulnerable" bias (P-V 71.63 vs 54.26 with CoT). The demos in that paper were not contrastive-retrieved pairs. — [PrimeVul](https://arxiv.org/html/2403.18624v2)

### Inferences
- VulGuardVN's 1-2 retrieved contrastive code demos correspond to the "code-level RAG" baseline that Vul-RAG beats by 16-27% in pair accuracy.
- A plausible upgrade is switching from raw-code demos to **abstracted knowledge**: cause, fix pattern and the specification violated (Vul-RAG / VulInstruct style). Retrieval should run on an abstract functional summary rather than raw code, given the finding that query abstraction matters most.
- The expected ceiling is low: about +0.05-0.1 pairwise accuracy, based on Vul-RAG 0.20-0.26 and VulInstruct 17.2%.
- Leakage risk: PrimeVul paired train pairs used as the retrieval store must be from before the test cutoff, and near-duplicates of test functions must be removed. Otherwise RAG gains will be inflated toward the oracle 0.51.
- Demos can shift the LLM's label bias (GPT-4 two-shot toward "vulnerable"). VulGuardVN should check P-V vs P-B with and without demos, not just accuracy.

### Gaps
- No clean head-to-head of "contrastive vulnerable/patched code demos vs zero-shot" on PrimeVul paired with modern models was found.
- The per-system no-retrieval ablation in the 2026 reproduction study was not reported in the portion read.
- Not captured:
  - GRACE's own reported results.
  - The SVD-Bench and Llama-VD papers.
  - VulTriage's ablation of its three context paths.
