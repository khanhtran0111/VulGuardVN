# Root causes: why prompted LLMs add no signal for function-level C/C++ vulnerability detection on Devign / ReVeal (2023-2026 literature)

Local reference point (VulGuardVN-v2): fine-tuned UniXcoder + stacking prefilter AUC ~0.73 (Devign) / ~0.82 (ReVeal); gpt-oss-20b two-pass verifier (hypothesis -> neutral/refute verification, Joern CPG checks, contrastive RAG demo pairs) gives verbalized-confidence AUC 0.46-0.49, ~0 AUC gain under logistic fusion, is right on only 30-40% of prefilter errors, says "vulnerable" 59-71% of the time; gpt-oss-120b and higher reasoning effort no better.

## Q1. Label noise / label correctness in Devign, ReVeal, BigVul and how it caps achievable performance

### Takeaway
Manual audits find that only about 25-80% of "vulnerable" labels in the standard C/C++ datasets are correct. For Devign the figure is 47% (Risse et al.) to 80% (Croft et al.), and for BigVul it is 25-54%. The errors come mostly from fix-commit heuristics: tangled, cleanup and unrelated changes, and wrongly identified fix commits. A verifier that judges real vulnerability therefore disagrees with the labels by construction on a large share of positives. A fine-tuned model can still score well by learning dataset artifacts: word counts alone get to within ~5 F1 points of the state of the art on Devign.

### Cited Findings
- **Croft et al., "Data Quality for Software Vulnerability Datasets" (ICSE 2023).** Across four datasets, 20-71% of vulnerability labels are inaccurate and 17-99% of data points are duplicated — [arXiv 2301.05456](https://arxiv.org/abs/2301.05456)
  - Method: 70 random samples per dataset (90% confidence, +/-10%), two raters, Cohen kappa 0.627. Label accuracy: Devign 0.800, Big-Vul 0.543, D2A 0.286, Juliet 1.000. Uniqueness: Big-Vul 0.830, Devign 0.899, D2A 0.021, Juliet 0.163. Consistency: Big-Vul 0.999, Devign 0.991, D2A 0.531 — [Croft et al. PDF](https://arxiv.org/pdf/2301.05456)
  - Types of label inaccuracy (Table IV): Devign has 42.9% irrelevant code changes, 21.4% cleanup changes, and 35.7% inaccurate fix identification. Big-Vul has 25% / 28.1% / 46.9%. The datasets "largely assume that code touched by a vulnerability fix is vulnerable code"; tangled commits are the main cause. There are also ambiguous cases, e.g. "should the caller of a vulnerable code snippet also be labelled as such?" — [Croft et al. PDF](https://arxiv.org/pdf/2301.05456)
  - Impact: when evaluated against manually corrected labels, precision fell by 29% (Devign), 50% (Big-Vul) and 80% (D2A), significant at p<0.05. "What were previously considered true positives became false positives" — [Croft et al. PDF](https://arxiv.org/pdf/2301.05456)
  - Removing cross-set duplicates on Devign lowered MCC from 0.284 to 0.244 (-13.9%). Precision/recall went from 0.680/0.428 to 0.651/0.399. On Big-Vul, MCC was unchanged (0.830 to 0.829) — [Croft et al. PDF](https://arxiv.org/pdf/2301.05456)
- **Ding et al., PrimeVul, "Vulnerability Detection with Code Language Models: How Far Are We?" (ICSE 2025).** Manually checked vulnerable-label accuracy (Table I): CodeXGLUE (= Devign) 24.0%, VulnPatchPairs 36.0%, SVEN 94.0%, PrimeVul-OneFunc 86.0%, PrimeVul-NVDCheck 92.0%. Figures quoted from Chen et al.: BigVul 25.0%, CrossVul 47.8%, CVEFixes 51.7%, DiverseVul 60.0% — [arXiv 2403.18624 (HTML v2)](https://arxiv.org/html/2403.18624v2)
  - Exact-copy duplication (Table II): BigVul 12.7%, CVEFixes 18.9%, CodeXGLUE/Devign 0.6%, DiverseVul 3.3%, PrimeVul 0%. "Up to 18.9% test samples are leaked from the train set in some benchmarks" — [PrimeVul](https://arxiv.org/html/2403.18624v2)
  - Collapse on realistic data (F1, BigVul vs. trained on BigVul and tested on PrimeVul): StarCoder2 68.26 vs. 3.09, CodeT5 64.93 vs. 5.82, CodeBERT 62.88 vs. 4.49, **UniXcoder 65.46 vs. 1.94**, CodeGen2.5 67.30 vs. 1.91. Trained and tested on PrimeVul, StarCoder2 reaches only 18.05 F1 — [PrimeVul](https://arxiv.org/html/2403.18624v2)
- **Risse, Liu, Böhme, "Top Score on the Wrong Exam: On Benchmarking in Machine Learning for Vulnerability Detection" (ISSTA 2025).** Audit of 100 vulnerable-labeled functions per dataset:
  - Actually vulnerable: Devign 47%, BigVul 39%, DiverseVul 65%, so 53% / 61% / 35% are not vulnerable.
  - Causes of wrong labels: patch-commit identification (BigVul 21%, Devign 54%, DiverseVul 3%), structural changes (48% / 38% / 66%), unrelated changes (31% / 8% / 31%).
  - Agreement: 82% (kappa 0.64).
  - Source: [arXiv 2408.12986v2](https://arxiv.org/html/2408.12986v2); [ISSTA'25 PDF](https://mboehme.github.io/paper/ISSTA25-topscore.pdf)
- **Spurious features (same paper).** Gradient boosting on word counts alone, compared with the state of the art:
  - Devign: F1 0.62 vs. 0.68; accuracy 0.63 vs. 0.68.
  - BigVul: F1 0.86 vs. 0.96.
  - DiverseVul: F1 0.11 vs. 0.48.
  - High benchmark scores are therefore reachable without any vulnerability reasoning — [arXiv 2408.12986v2](https://arxiv.org/html/2408.12986v2)
- **Usage survey (same paper).** 53 of 81 ML4VD papers (2020-2024) use BigVul, Devign or ReVeal (36, 36 and 22 papers respectively). 88% frame the task as function-level binary classification — [arXiv 2408.12986v2](https://arxiv.org/html/2408.12986v2)
- **Conflict on Devign.** Croft et al. find 80% label accuracy, Risse et al. 47%, and PrimeVul's audit of CodeXGLUE/Devign 24%. The differences in criteria are probably the cause: Croft checked only whether the changed lines are "functionally relevant to the reported vulnerability" and "did not verify whether a vulnerability was actually exploitable" — [Croft et al.](https://arxiv.org/pdf/2301.05456); [Risse et al.](https://arxiv.org/html/2408.12986v2); [PrimeVul](https://arxiv.org/html/2403.18624v2)

### Inferences
- With 20-76% of Devign positives mislabeled, a verifier that correctly judges "this function is not actually vulnerable" is scored as a false negative on those items. A verifier that is right on real vulnerability can therefore have near-chance AUC against Devign labels. This fits the observed LLM AUC of 0.46-0.49.
- The UniXcoder prefilter's 0.73 AUC on Devign is probably achieved partly through label-correlated artifacts: commit style, project idiom (FFmpeg/QEMU), and lexical cues. Word counts alone reach 0.62-0.63 on Devign. A semantic LLM verifier has no access to these cues, so its signal is not "complementary". It mostly measures a different target from the one the labels encode.
- Croft's finding that label noise mainly inflates false positives also predicts precision ceilings for any stage that judges real vulnerability.

### Gaps
- I found no ReVeal-specific manual label-accuracy audit in 2023-2026 papers within budget. ReVeal (Chakraborty et al., TSE 2022, Chromium + Debian, fix-commit based) uses the same fix-commit labeling heuristic, so similar noise is plausible but unquantified here.
- No paper found that bounds AUC directly as a function of label-noise rate on these datasets.

## Q2. LLM-for-vulnerability-detection benchmark studies: accuracy, pairwise accuracy, bias toward "vulnerable"

### Takeaway
Across independent studies, prompted LLMs perform at or near chance on realistic C/C++ function-level detection. On paired data they cannot separate vulnerable from patched versions, and they are strongly biased to call both versions vulnerable: GPT-4 does so 54-72% of the time on PrimeVul pairs. Fine-tuned small code encoders, UniXcoder in particular, match or beat LLMs on Devign and ReVeal.

### Cited Findings
- **PrimeVul paired test set (564 pairs; 5,480 pairs in total).**
  - Metrics: P-C = both correct, P-V = both predicted vulnerable, P-B = both predicted benign, P-R = reversed.
  - GPT-4 two-shot: P-C 5.14, P-V 71.63, P-B 21.45, P-R 1.77.
  - GPT-4 with chain of thought: P-C 12.94, P-V 54.26, P-B 24.47, P-R 8.33.
  - GPT-3.5 two-shot: 5.67 / 13.83 / 77.84 / 2.66. GPT-3.5 with chain of thought: 6.21 / 4.79 / 83.51 / 5.50. GPT-3.5 fine-tuned: 1.24 / 5.32 / 90.96 / 2.48.
  - Random guessing gives P-C 22.70. "GPT-4 with the chain-of-thought reasoning could not even outperform the random guessing."
  - Source: [PrimeVul](https://arxiv.org/html/2403.18624v2)
  - Fine-tuned code LMs are no better on pairs: StarCoder2 P-C 2.30 / P-V 8.16 / P-B 88.30, and CodeGen2.5 P-C 3.01 — [PrimeVul](https://arxiv.org/html/2403.18624v2)
- **Steenhoek et al., "To Err is Machine: Vulnerability Detection Challenges LLM Reasoning" (arXiv 2403.17218v2, Jan 2025).** Covers 14 LLMs and 7 prompting methods.
  - Balanced accuracy on SVEN is only 50.9-54.5% (best: StarCoder2 at 54.5%). "No model or prompt exceeded random guessing by more than 5 points."
  - Even simple CWE examples reach only 42-67%.
  - On average, 69.7% of buggy/fixed pairs could not be distinguished (range 55.0-86.1%). Only 16.3% of pairs were both predicted correctly.
  - Source: [arXiv 2403.17218](https://arxiv.org/html/2403.17218)
  - Error analysis (300 responses): misunderstood bounds/NULL checks account for 50% of the semantic errors (80/159); localization/understanding errors 43%; logical errors 9%; hallucination/memorization/repetition 10%. Domain-knowledge chain-of-thought annotations cut bounds/NULL errors by 15-70% but "did not significantly change overall detection performance" — [arXiv 2403.17218](https://arxiv.org/html/2403.17218)
- **Ullah et al., SecLLMHolmes, "LLMs Cannot Reliably Identify and Reason About Security Vulnerabilities (Yet?)" (IEEE S&P 2024).** Covers 228 code scenarios and 8 LLMs. LLMs give "non-deterministic responses, incorrect and unfaithful reasoning, and perform poorly in real-world scenarios." Trivial perturbations (renaming functions or variables, adding library functions) caused wrong answers in 26% of cases for PaLM2 and 17% for GPT-4 — [arXiv 2312.12575](https://arxiv.org/abs/2312.12575)
- **Li et al., "Everything You Wanted to Know About LLM-based Vulnerability Detection But Were Afraid to Ask" (arXiv 2504.13474, Apr 2025).**
  - Setup: 13 LLMs. Without context, most models score F1 0.5-0.6, accuracy 0.5-0.55 and precision ~0.5-0.55.
  - All models predict both versions vulnerable ((1,1)) in about 10% of pairs. With context, DeepSeek-R1 gets the pair right ((1,0)) in 37% of cases, against a 25% random baseline.
  - "Smaller models over-predict vulnerabilities and missing context amplifies this."
  - Source: [arXiv 2504.13474](https://arxiv.org/html/2504.13474v1)
- **Jiang et al., "Investigating Large Language Models for Code Vulnerability Detection: An Experimental Study" (arXiv 2412.18260, Jan 2025).** Models: LoRA-fine-tuned Llama-2/CodeLlama/Llama-3/3.1 (7-8B).
  - Short-sample F1 on Devign: UniXcoder 65.27 is best, ahead of Llama-3.1-8B 61.49, CodeLlama 56.64 and Llama-2 47.86.
  - On ReVeal: UniXcoder 38.89 and CodeBERT 35.92 vs. LLMs 11.19-23.78. ReVeal accuracy of ~92-93% hides this F1 gap.
  - "Medium-sized sequence models are less affected by a low proportion of positive samples than LLMs."
  - Source: [arXiv 2412.18260](https://arxiv.org/html/2412.18260)
- **VulnPatchPairs (Risse & Böhme, USENIX Security 2024).** Built from CodeXGLUE/Devign (26.2k FFmpeg+QEMU functions, half vulnerable and half their patches). ML4VD models reach "up to 70% accuracy" on the standard benchmark but are "worse than random guessing on average" on vulnerable/patch pairs — [USENIX Sec'24](https://www.usenix.org/conference/usenixsecurity24/presentation/risse); [arXiv 2306.17193](https://arxiv.org/abs/2306.17193)
- **Easier synthetic benchmarks look much better, which is a contrast and not a contradiction.**
  - CASTLE (250 micro-programs, 25 CWEs): o3-mini found 126 of 150 known vulnerabilities, but "accuracy declines, and hallucinations increase as the code size grows" — [arXiv 2503.09433](https://arxiv.org/pdf/2503.09433)
  - VulDetectBench (17 models): >80% accuracy on existence and type identification tasks, <30% on detailed analysis tasks — [arXiv 2406.07595](https://arxiv.org/html/2406.07595v4)

### Inferences
- The user's LLM says "vulnerable" 59-71% of the time, which matches GPT-4's 54-72% P-V rate on PrimeVul pairs. A prior toward "vulnerable" is a known, stable failure mode of prompted LLMs on real C code. Refute-mode prompting is unlikely to remove it, because the bias comes from being unable to rule a vulnerability out without context (see Q3).
- On Devign (~46% positive), a "vulnerable"-biased verifier with near-random ranking gives AUC ~0.5 and no fusion gain, which is exactly what was observed.
- The literature reports UniXcoder as the strongest or near-strongest model on both Devign and ReVeal. A prefilter AUC of 0.73/0.82 is already at or above what LLMs reach in published comparisons.

### Gaps
- I found no paper that reports a prompted GPT-4-class or open reasoning-model AUC (as opposed to accuracy/F1) specifically on the Devign or ReVeal test splits.
- I did not retrieve exact SecLLMHolmes figures for false positives on patched versions.

## Q3. Is function-level detection decidable from the function alone?

### Takeaway
Mostly no. Risse et al. found that every one of 151 truly vulnerable functions they audited needed external context, and that for ~91% of "secure" functions a plausible vulnerable calling context exists. PrimeVul's authors also say the function-level task "may be asking an impossible question." Labels are derived from fix-commit diffs, not from properties of the function itself.

### Cited Findings
- **Risse et al. (ISSTA 2025).**
  - All 151 actually-vulnerable functions (from 300 audited) are context-dependent; none is context-independent. Context agreement: 98%, kappa 0.96.
  - Context needed: external function (BigVul 49% / Devign 43% / DiverseVul 52%), function argument (41% / 47% / 40%), globals (8% / 10% / 2%), type declarations (2% / 0% / 3%).
  - For 82 of 90 secure-labeled functions (~91%), the authors could construct a plausible context that makes them vulnerable.
  - Headline: whether a function is vulnerable cannot be decided without further context for >90% of functions.
  - Source: [arXiv 2408.12986v2](https://arxiv.org/html/2408.12986v2)
- **PrimeVul.** "Determining whether code is vulnerable generally depends on information about other components of the system as well"; the function-level task "may be asking an impossible question"; the authors recommend broader context — [PrimeVul](https://arxiv.org/html/2403.18624v2)
- **Croft et al.** Datasets "assume vulnerable code to be all code touched in a vulnerability fix, but commits are messy in practice." Cleanup changes produce false-positive labels that "do not indicate the location of the underlying exploitable code" — [Croft et al.](https://arxiv.org/pdf/2301.05456)
- **Li et al. (2025): context changes the picture.** They built CORRECT: 2,000 vulnerable/patched pairs, 99 CWEs, 98% label accuracy on a 50-pair audit, average shared context 13,249 tokens vs. 1,574 tokens for the function. With this context, DeepSeek-R1 reaches 67% accuracy and the 671B models reach precision close to 0.8. Without context, models sit at 0.5-0.55 accuracy — [arXiv 2504.13474](https://arxiv.org/html/2504.13474v1)

### Inferences
- The user's Pass 1 asks for a "required guard" on a sink. Under function-only input, that guard usually lives in a caller or callee: arguments account for ~47% and external functions ~43% of the context needed on Devign. The LLM therefore cannot verify its absence and defaults to "vulnerable" (unknown guard means possibly unsafe). This is a mechanistic explanation of the 59-71% "vulnerable" rate.
- Joern CPG checks restricted to the single function cannot add the missing interprocedural facts. Contrastive vulnerable/patched demos teach diff-level cues, but the test item comes without its diff.

### Gaps
- No quantified context-dependence audit specifically for ReVeal found.
- No study found that measures how much an interprocedural CPG slice, as opposed to repository-level text context, recovers on Devign or ReVeal.

## Q4. Why verbalized confidence is uninformative, and why LLM errors correlate with code-model errors

### Takeaway
General calibration research finds that verbalized LLM confidence is overconfident, with failure-prediction AUROC only ~0.52-0.61. In vulnerability detection, both LLMs and code models fail on the same context-dependent, mislabeled, or near-identical (vulnerable vs. patched) samples, and both rely on surface or lexical cues. Their errors therefore overlap instead of complementing each other.

### Cited Findings
- **Xiong et al., "Can LLMs Express Their Uncertainty?" (ICLR 2024).** LLMs "tend to be overconfident" when verbalizing confidence; failure-prediction AUROC across methods is only 0.522-0.605; all methods "struggle on tasks requiring professional knowledge" — [arXiv 2306.13063](https://arxiv.org/abs/2306.13063)
- **Verbalized confidence saturation.** Verbalized scores saturate near the top of the scale. With Qwen3-8B, a usable true-positive rate was reached only at a false-positive rate of 0.24 (TriviaQA, not code) — [arXiv 2509.25532](https://arxiv.org/html/2509.25532v1)
- **Surface-feature reliance in LLMs.** Renaming or adding library calls flips 17-26% of answers — [SecLLMHolmes](https://arxiv.org/abs/2312.12575)
- **Surface-feature reliance in code models.** Word-count models nearly match the state of the art on Devign (F1 0.62 vs. 0.68) — [Risse et al.](https://arxiv.org/html/2408.12986v2)
- **Shared failure on near-duplicates.**
  - Fine-tuned models label 88% of PrimeVul pairs both benign, while GPT-4 labels 54-72% both vulnerable. Neither separates the pair — [PrimeVul](https://arxiv.org/html/2403.18624v2)
  - VulnPatchPairs: models are worse than random on pairs — [Risse & Böhme USENIX'24](https://www.usenix.org/conference/usenixsecurity24/presentation/risse)
  - LLMs cannot distinguish 69.7% of buggy/fixed pairs — [Steenhoek et al.](https://arxiv.org/html/2403.17218)
- **Reasoning errors concentrate on guards.** Bounds/NULL-check misunderstandings are 50% of the semantic errors — [Steenhoek et al.](https://arxiv.org/html/2403.17218)

### Inferences
- The prefilter's errors are probably concentrated on: (a) mislabeled positives (20-76% of Devign positives), where the code looks benign; (b) context-dependent items; (c) near-duplicate vulnerable/patched pairs. An LLM fails on all three for the same reasons. In (a) it may even be "right" in reality yet scored wrong. This explains being correct on only 30-40% of prefilter errors, which is below a 50% coin flip and consistent with shared difficulty.
- Verbalized confidence AUC of 0.46-0.49 (slightly below 0.5) fits a verifier that becomes more confident about "vulnerable" on code with security-looking surface features (memcpy, pointer arithmetic, length variables). In fix-commit datasets, such code appears in both the vulnerable and the patched/clean classes.
- Fusion adds ~0 AUC because the LLM score is nearly independent of the label conditional on the prefilter score. Being "right" in reality on mislabeled items yields no measured gain.

### Gaps
- I found no paper that measures verbalized-confidence AUC or calibration specifically for LLM vulnerability detection on Devign or ReVeal. The calibration evidence above is general (QA/reasoning) and has to be extrapolated.
- No paper found that directly measures error correlation (e.g., Cohen's kappa of errors) between LLM verifiers and fine-tuned code models on these datasets.

## Q5. Do larger models or more reasoning help on these datasets?

### Takeaway
Not under function-only input. Model size shows no correlation with performance (R²=0.02), fivefold more reasoning tokens give under 5 points of accuracy, and extended reasoning cuts recall by ~10%. Scaling benefits appear only once real context is supplied. This matches the user's gpt-oss-120b and higher-effort results.

### Cited Findings
- **Size.** There is no significant correlation between model size and performance (R² = 0.02, p = 0.72) across 14 LLMs. Code-specialized training and instruction tuning also give no substantial gain. Fine-tuning StarCoder2-7B on PrimeVul was worse on balanced data — [Steenhoek et al.](https://arxiv.org/html/2403.17218)
- **Reasoning effort.**
  - Without context there is no clear scaling with size; scaling appears only with CORRECT context.
  - o3-mini at high effort (~5x the thinking tokens of medium) improved accuracy by "less than 0.05".
  - Extended reasoning reduced recall by ~10%. Sequential scaling (r1-distilled 14B) lowered recall, while parallel majority voting held or improved it.
  - Source: [Li et al. 2504.13474](https://arxiv.org/html/2504.13474v1)
- **Chain of thought.** GPT-4 with CoT on PrimeVul pairs (P-C 12.94) stays below random (22.70) — [PrimeVul](https://arxiv.org/html/2403.18624v2)
- **Contrast case.** On compact synthetic programs (CASTLE), reasoning models such as o3-mini do well (126/150), but performance degrades as code size grows — [CASTLE](https://arxiv.org/pdf/2503.09433)
- **Other reasoning-model work (2025), not verified in depth.**
  - Reasoning-model studies report gains for DeepSeek-R1 / o3-mini over non-reasoning models on other benchmarks — [MDPI Applied Sciences 2025](https://www.mdpi.com/2076-3417/15/12/6651)
  - Distilled reasoning detectors (VulnLLM-R) use R1/o3 as teachers and filter out wrong reasoning chains — [arXiv 2512.07533](https://arxiv.org/pdf/2512.07533)
  - I did not verify that these gains carry over to Devign, ReVeal, or PrimeVul paired settings.

### Inferences
- The bottleneck is information (missing context) and target validity (noisy labels), not model capacity or reasoning depth. More reasoning, without new facts, tends to build more elaborate "possibly vulnerable" narratives, consistent with the ~10% recall shifts and the stable "vulnerable" bias.
- Any LLM-stage redesign that keeps function-only input and Devign/ReVeal labels is expected to stay at ~0.5 AUC. Realistic levers from the literature:
  - Supply interprocedural context (callers, callees, argument provenance), as in CORRECT.
  - Evaluate on cleaner labels (PrimeVul, CORRECT, SVEN) or on pairs.
  - Use the LLM for explanation and localization rather than as a score-fusion signal.

### Gaps
- No published evaluation found of gpt-oss-20b or gpt-oss-120b on Devign, ReVeal, or PrimeVul.
- Exact PrimeVul-paired numbers for DeepSeek-R1, o3, or o4-mini were not found within budget.
