# Combining an LLM with a fine-tuned code classifier: fusion, LLM confidence, cascades/routing, and VD hybrids

Scope note: these notes cover (a) when a second model adds value, (b) how to get a usable score from an LLM, (c) cascade and router designs and how they fail, and (d) vulnerability-detection (VD) hybrids and ensembles. Items marked **[prior knowledge, not re-fetched]** are well-known results cited from the primary paper's abstract or main text from memory. They were not re-verified in this session, so the report writer should treat their exact digits as needing a check.

Local context for interpretation (from the assignment, not from literature):
- gpt-oss-20b verdict, verbalized confidence, and #hypotheses add about 0 AUC when stacked with the prefilter p_hat.
- Verbalized-confidence AUC is 0.46-0.49.
- The LLM is right on 30-40% of prefilter-wrong functions.
- The LightGBM benefit-router has Spearman 0.09-0.40 with the true benefit.
- The oracle gain is smaller than the oracle gain of a random coin flip.
- Decoding is at T=0.

---

## Q1. Verbalized confidence vs sampling consistency vs logprobs vs P(True): calibration and AUROC

### Takeaway
On standard benchmarks, verbalized confidence from a single response is close to useless for ranking errors. It has AUROC of about 0.51-0.63, and only GPT-4 reaches about 0.63. Sampling about 5 responses and using agreement (vote fraction) raises failure-prediction AUROC to about 0.69-0.75. Combining consistency with verbalized confidence is slightly better still. Token probabilities work well for base models but are miscalibrated after RLHF. Verbalized confidence lowers ECE for RLHF models, but lower ECE does not mean better ranking. For a binary classifier that will be fused with another model, the sampled vote fraction or the P(yes) token logprob is the signal worth trying. The 0.46-0.49 verbalized AUC seen locally is in line with the literature's near-chance results for verbalized confidence on hard, specialist tasks.

### Cited Findings
- **Xiong et al., "Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs" (ICLR 2024):** vanilla verbalized confidence averaged over 8 datasets gives ([arXiv 2306.13063](https://arxiv.org/html/2306.13063)):

  | Model | ECE | AUROC |
  |---|---|---|
  | GPT-3 | 52.0 | 51.3 |
  | Vicuna-13B | 46.1 | 52.5 |
  | LLaMA-2-70B | 43.6 | 56.4 |
  | GPT-3.5 | 37.7 | 55.1 |
  | GPT-4 | 18.0 | 62.7 |
- Same paper: verbalized confidences "primarily range between 80% and 100%", usually in multiples of 5. This is systematic overconfidence, possibly an imitation of human phrasing ([arXiv 2306.13063](https://arxiv.org/html/2306.13063)).
- Same paper: chain-of-thought with one sample (GPT-3.5) averages ECE 25.0 and AUROC 56.4 over 5 datasets. CoT does not fix ranking ([arXiv 2306.13063](https://arxiv.org/html/2306.13063)).
- Same paper, consistency sampling with M=5 (GPT-3.5, CoT), averages over 5 datasets ([arXiv 2306.13063](https://arxiv.org/html/2306.13063)):
  - Self-Random sampling + consistency: ECE 18.7, **AUROC 73.0**.
  - Misleading-prompt sampling + consistency: ECE 17.3, AUROC 69.6.
  - Prompt-based sampling: ECE 24.3, AUROC 69.2.
  - On GSM8K, AUROC goes from **54.8 at M=1 to 92.7 at M=5**, with ECE 6.28.
- Same paper: the hybrid of consistency weighted by verbalized confidence ("Avg-Conf", M=5, GPT-3.5) averages ECE 14.8 and **AUROC 74.5**. That is a small gain of about 1.5 AUROC points over pure consistency ([arXiv 2306.13063](https://arxiv.org/html/2306.13063)).
- Same paper, GPT-4 with Top-K prompting and Self-Random M=5 ([arXiv 2306.13063](https://arxiv.org/html/2306.13063)):

  | Aggregator | ECE | AUROC |
  |---|---|---|
  | Consistency | 12.0 | 66.9 |
  | Avg-Conf | 14.8 | 66.9 |
  | Pair-Rank | 6.90 | 67.6 |
- Same paper: as M grows, performance "improves significantly and then converges", and the gains are marginal beyond about 5 samples (tested up to 13). The abstract concludes that no single technique consistently wins, that all methods struggle on tasks needing professional knowledge, and that white-box (logit) methods are better but only narrowly, at AUROC 0.522-0.605 ([arXiv 2306.13063](https://arxiv.org/abs/2306.13063)).
- **Tian et al., "Just Ask for Calibration" (EMNLP 2023):** for RLHF models (ChatGPT, GPT-4, Claude) on TriviaQA, SciQ and TruthfulQA, verbalized confidences are better calibrated than the model's conditional token probabilities, "often reducing the expected calibration error by a relative 50%". This is an ECE result, not an AUROC result ([arXiv 2305.14975](https://arxiv.org/abs/2305.14975)).
- **GPT-4 Technical Report:** the pre-trained GPT-4 was well calibrated on MMLU, with ECE 0.007, and post-training with PPO/RLHF raised ECE to 0.074 **[prior knowledge, not re-fetched]** ([arXiv 2303.08774](https://arxiv.org/abs/2303.08774)). This is direct evidence that RLHF and chat tuning degrade token-probability calibration. It also motivates recalibrating logprobs, for example with temperature or Platt scaling, rather than using them raw.
- **Kadavath et al., "Language Models (Mostly) Know What They Know" (2022)** ([arXiv 2207.05221](https://arxiv.org/abs/2207.05221)):
  - Larger models are well calibrated on multiple-choice and true/false questions in the right format.
  - P(True) self-evaluation of a proposed answer works on open-ended tasks.
  - Self-evaluation **improves when the model first sees many of its own samples** before judging one answer ("brainstorming").
  - P(IK), trained to predict whether the model knows the answer, "struggle[s] with calibration" on new tasks. A probe trained in-domain does not transfer well.
- **Wang et al., "Self-Consistency Improves Chain of Thought Reasoning" (ICLR 2023):** majority vote over sampled CoT paths raised PaLM-540B GSM8K accuracy by +17.9 points over greedy CoT **[prior knowledge, not re-fetched]** ([arXiv 2203.11171](https://arxiv.org/abs/2203.11171)). The vote fraction gives a graded confidence score for free.
- **Secondary and benchmark evidence (2025-2026), mixed:**
  - One hallucination benchmark found verbalized self-evaluation had the highest pooled AUROC (0.688), ahead of self-consistency (0.638). Self-consistency's strong result in one domain rested on only 5 positive samples ([HalluScan, arXiv 2605.02443](https://arxiv.org/pdf/2605.02443)).
  - In a security-adjacent task (automotive CVE-to-ATM mapping), self-consistency alone gave AUROC 0.705, and adding evidence-grounded judge signals raised it to 0.868 ([arXiv 2607.18684](https://arxiv.org/pdf/2607.18684)).
  - Removing self-consistency (K=1) worsened AURC by 6-10% in a risk-controlled refusal framework ([arXiv 2509.01455](https://arxiv.org/html/2509.01455v1)).
  - A systematic review notes that AUROC can be indifferent to calibration, so both should be reported ([arXiv 2504.18346](https://arxiv.org/pdf/2504.18346)).

### Inferences
- The locally observed verbalized AUC of 0.46-0.49 matches the literature's floor of about 0.51-0.56 for non-GPT-4 models. On specialist tasks, where Xiong et al. say all methods struggle, it can fall to chance. The cheapest next step is **K=5-10 samples at T≈0.7-1.0, with the vote fraction P(vuln) as the score**. Xiong's average uplift from M=1 to M=5 is about +17 AUROC points. On a hard binary task such as VD, the uplift will likely be smaller.
- With gpt-oss-20b open weights, the white-box option is also available: read the logprob of the final verdict token, "yes" vs "no", after reasoning. Then recalibrate it with Platt or isotonic regression on the validation split. White-box AUROC was only narrowly better than black-box in Xiong et al., but it costs one forward pass instead of K.
- A combined score, vote fraction × mean verbalized confidence (Avg-Conf), is cheap once samples exist. Expect it to add only about 1-2 AUROC points over the vote fraction.
- What matters for stacking is the LLM score's **conditional** discriminative power given p_hat. That is the AUC within p_hat bins, or partial correlation with the label residual. Its marginal AUC is not the right measure. A sampled score can have decent marginal AUC and still add nothing if it simply re-encodes p_hat.

### Gaps
- I found no paper that measures verbalized vs sampled vs logprob confidence specifically for LLM **vulnerability detection**. The search returned none, so transfer from QA and math benchmarks is assumed.
- Kadavath et al.'s exact P(True) AUROC numbers were not retrieved; only the abstract was fetched.
- I have no numbers for gpt-oss-style "reasoning" models, whose final-token logprob after long chain-of-thought may be near-deterministic (saturated). That is something to check empirically.

---

## Q2. When does stacking or ensembling a weak, correlated second model add value? Diversity metrics and conditions

### Takeaway
An ensemble helps only if the second model is (i) more accurate than chance on the cases where the first is wrong, and (ii) its errors are not concentrated on the same examples. A coin flip is right on 50% of a binary classifier's errors. A second model that is right on only 30-40% of the prefilter's errors is therefore **worse than random on exactly the subset that matters**. That is strong evidence of positively correlated errors, and no stacker can extract gain from it. Literature on LLMs shows errors are highly correlated across models, and more so for stronger models. VD-specific ensembles of code LLMs give gains on balanced Devign but often fail on imbalanced ReVeal.

### Cited Findings
- **Kuncheva & Whitaker, "Measures of diversity in classifier ensembles and their relationship with the ensemble accuracy" (Machine Learning, 2003):** this paper defines the pairwise oracle-output measures **[prior knowledge, not re-fetched]** ([Springer](https://link.springer.com/article/10.1023/A:1022859003006)):
  - Q-statistic: Q = (N11·N00 − N01·N10)/(N11·N00 + N01·N10), where N11 means both models are right, N00 both wrong, and N10 or N01 exactly one right. Q is in [−1, 1], and Q=0 means the errors are independent.
  - Disagreement measure: (N01+N10)/N.
  - Double-fault measure: N00/N.
  - Correlation ρ.
  - The authors found that the relationship between these measures and ensemble accuracy is **ambiguous in practice**. Diversity is necessary but not sufficient.
- **Kim, Garg, Peng & Garg, "Correlated Errors in Large Language Models" (ICML 2025):** across more than 350 LLMs, models agree on the same wrong answer **60% of the time when both are wrong** on HELM. The mean error agreement is 0.423 on the HuggingFace leaderboard and 0.6 on HELM, roughly double or more the chance baseline. Shared architecture or provider increases correlation, and **more accurate models have more correlated errors**, even across distinct architectures ([arXiv 2506.07962](https://arxiv.org/pdf/2506.07962); [ICML poster](https://icml.cc/virtual/2025/poster/44225)).
- **Bradley, "LLMs and the Madness of Crowds" (2024):** incorrect answers are systematically correlated across models rather than random. The paper argues that ensembling LLMs may be much less effective than ensembling other kinds of models for this reason ([arXiv 2411.01539](https://www.arxiv.org/pdf/2411.01539)).
- **Sun et al., "Ensembling Large Language Models for Code Vulnerability Detection: An Empirical Evaluation" (2025):**
  - Setup: 5 fine-tuned code LLMs (DeepSeek-Coder-6.7B, CodeLlama-7B/13B, CodeQwen1.5-7B, StarCoder2-15B), with Devign, ReVeal and BigVul as datasets ([arXiv 2509.12629](https://arxiv.org/pdf/2509.12629)).
  - Venn diagrams show that each model, and even each epoch of the same model, correctly catches a unique subset of vulnerable samples.
  - Boosting has the best F1 average rank (1.60).
  - **On imbalanced ReVeal, Bagging and Stacking had lower F1 than the baseline LLMs in all 5 instances. Only Boosting improved, with a +34.38% relative gain in Recall.**
  - Diversity was scarce on ReVeal: across 5 epochs of CodeLlama-13B there were 1,683/2,732 divergent test samples on Devign, but only **243/2,274 on ReVeal** and 807/1,726 on BigVul. The authors attribute this to models collapsing toward the majority class, so ensemble gains on divergent samples barely move the aggregate metrics.
- **Jitkrittum et al., "When Does Confidence-Based Cascade Deferral Suffice?" (NeurIPS 2023):** the optimal two-model deferral rule depends on both models' confidences. Simple confidence-based deferral fails when model 2's error probability is highly non-uniform across samples, when models are specialists, under label noise, or under distribution shift. In those cases post-hoc learned deferral can do significantly better ([arXiv 2307.02764](https://arxiv.org/abs/2307.02764); [NeurIPS 2023](https://neurips.cc/virtual/2023/poster/72864)).

### Inferences
- **Minimum condition for gain from model B on top of model A (binary task).** P(B correct | A wrong) must be clearly above P(B correct | A right) − ε. In practice:
  1. Q-statistic well below about 0.5.
  2. P(B correct | A wrong) well above 0.5.
  3. B's score must carry information **within strata of A's score**. A partial-AUC or within-bin AUC above 0.5 is a necessary check before stacking.

  Locally, P(B correct | A wrong) is 0.30-0.40, which is below 0.5. Most likely B is right mainly on easy cases A already gets right, so Q is near +1 on hard cases. This explains why logistic-regression stacking adds about 0 AUC. This is an inference from the numbers given, not a literature result.
- Recommended local diagnostics:
  - Q-statistic, disagreement, and double-fault between thresholded p_hat and the LLM verdict.
  - P(LLM correct | prefilter wrong) vs 0.5.
  - AUC of the LLM score within p_hat deciles.
  - Conditional mutual information I(LLM score; y | p_hat), estimated by binning.
- Sun et al.'s ReVeal result is a warning for the user's imbalanced setting. Stacking can **lower** F1 on imbalanced VD data. Boosting-style reweighting toward hard and minority examples was the only method that helped there.
- Because stronger and same-family LLMs have more correlated errors (Kim et al.), diversity should come from a different *signal type* rather than a bigger LLM. Examples are CPG/static-analysis facts, retrieval of similar CVE patches, or sampled reasoning paths. This is consistent with the user's memory note that 120b was no better than 20b on Devign/ReVeal.

### Gaps
- I did not retrieve exact per-dataset F1 tables from Sun et al. (only ranks and relative recall), so the absolute F1 gains of stacking are unknown here.
- I found no paper reporting Q-statistic or double-fault values between a fine-tuned VD classifier and a prompted LLM.

---

## Q3. Cascade and router designs, results, failure modes, and better upper bounds than a label-peeking oracle

### Takeaway
Published cascades and routers report large *cost* savings at matched quality. Examples are FrugalGPT (up to 98% cost reduction), RouteLLM (more than 2x and up to 3.66x cost savings vs a random router), AutoMix (more than 50%), and Hybrid LLM (about 40% fewer large-model calls). Almost all of them route from a weak model to a **strictly stronger** model. In the user's case the second model is not stronger on the routed subset, so these results do not transfer. The known failure modes are:
- Confidence-based deferral fails under specialization, non-uniform errors, label noise, or shift.
- Sequence-level LM uncertainty has length bias.
- Learned routers generalize poorly and are often matched by kNN.
- Oracle "pick-whichever-is-right" upper bounds are inflated by chance agreement, so they must be compared against a random-router baseline. RouteLLM's metric already does this.

### Cited Findings
- **FrugalGPT (Chen, Zaharia & Zou, 2023):** an LLM cascade with a learned scorer that decides whether to accept a cheaper model's answer. It can match the best individual LLM (GPT-4) with up to 98% cost reduction, or improve accuracy over GPT-4 by 4% at the same cost **[prior knowledge of the abstract, not re-fetched]** ([arXiv 2305.05176](https://arxiv.org/abs/2305.05176)). A secondary review stresses that these savings are task- and configuration-specific and not comparable across papers ([dev.to review](https://dev.to/aridiosilva/llm-routers-for-ai-systems-architecture-governance-multi-agent-systems-swarms-and-security-3g7f)).
- **RouteLLM (Ong et al., 2024):**
  - Routers are trained on preference data.
  - The metric is APGR (average performance gap recovered), where **a random router scores 0.500** by construction.
  - Best routers achieve more than 2x cost savings without hurting quality, and up to a 3.66x cost saving relative to the random baseline.
  - Augmenting with LLM-judge-labeled data is key to performance ([arXiv 2406.18665](https://arxiv.org/html/2406.18665v3)).
- **AutoMix (Aggarwal, Madaan et al., NeurIPS 2024):** the small model few-shot self-verifies its answer, and a POMDP meta-verifier decides whether to escalate. It beats FrugalGPT and HybridLLM across cost regions. Cost reduction is reported as "more than 2x" at matched performance in one summary, and as "over 50%" in secondary sources ([arXiv 2310.12963](https://arxiv.org/pdf/2310.12963)). The two framings are roughly consistent; the exact numbers were not checked.
- **Hybrid LLM (Ding et al., ICLR 2024):** a router predicts the query-difficulty quality gap between a small and a large model. Up to 40% fewer calls to the large model with no drop in response quality **[prior knowledge, not re-fetched]** ([arXiv 2404.14618](https://arxiv.org/abs/2404.14618)).
- **Gupta et al., "Language Model Cascades: Token-Level Uncertainty and Beyond" (ICLR 2024):**
  - Sequence-level uncertainty for deferral "suffers from the length bias problem".
  - Learned post-hoc deferral rules over token-level uncertainty features "significantly outperform" simple aggregation on FLAN-T5.
  - Adding embeddings from the small model and intermediate layers of the large model gives "an additional boost" ([arXiv 2404.10136](https://arxiv.org/abs/2404.10136)).
- **Jitkrittum et al. (NeurIPS 2023):** the Bayes-optimal 2-model deferral rule defers when the expected error of model 1 exceeds that of model 2 plus the cost. This needs **both** posteriors. Confidence-only deferral is near-optimal when model 2 is a consistent estimator of the posterior, and breaks under specialist models, label noise, or shift ([arXiv 2307.02764](https://arxiv.org/abs/2307.02764)).
- **Learning to defer:** Mozannar & Sontag (ICML 2020) give a consistent surrogate loss for jointly learning a classifier and a deferral rule to an expert. The rule defers when the expert's correctness probability exceeds the classifier's confidence **[prior knowledge, not re-fetched]** ([arXiv 2006.01862](https://arxiv.org/abs/2006.01862)).
- **Router generalization failures:**
  - "Rethinking Predictive Modeling for LLM Routing: When Simple kNN Beats Complex Learned Routers" (2025) finds kNN matches or exceeds complex learned routers at a fraction of the compute ([arXiv 2505.12601](https://arxiv.org/abs/2505.12601); [OpenReview](https://openreview.net/forum?id=Chn50flK4X)).
  - The LLMRouter benchmark finds no single router dominates, and the best router varies across tasks and budgets. It reports learned routers 14.6% better than the strongest fixed model, and it ships a RandomRouter baseline ([arXiv 2608.06867](https://arxiv.org/pdf/2608.06867v1); [GitHub](https://github.com/ulab-uiuc/LLMRouter)).
  - Router-R1 notes limited or inconsistent gains of learned routers on unseen models. That is a self-interested comparison ([arXiv 2506.09033](https://arxiv.org/pdf/2506.09033)).
- **Confidence-gated cascades in VD:** VulTriage uses a three-stage cascade (single agent → RAG agent → multi-agent) gated by confidence signals, and claims SOTA on the PrimeVul pair test set. Gate-level numbers were not extracted ([arXiv 2605.09461](https://arxiv.org/html/2605.09461v2)).

### Inferences
- **Why the label-peeking oracle is uninformative locally.** For a binary task, the oracle router "use B when B is right" has accuracy acc_A + P(A wrong, B right). For a random-coin B this equals acc_A + 0.5·(1 − acc_A). When B is right on only 30-40% of A's errors, the oracle gain is *by construction* smaller than a coin's, which is exactly what was observed. The oracle measures the luck available in B's errors, not B's skill. Better reference points:
  1. **Random-normalized gain**, as in RouteLLM's APGR. Report the router's gain minus a random router's gain at the same routing rate, and the oracle gain of B minus the oracle gain of a Bernoulli(acc_B) or Bernoulli(0.5) coin.
  2. **Bayes-optimal deferral with calibrated posteriors (Jitkrittum).** Fit calibrated P(A correct | x) and P(B correct | x, LLM features) on validation data, then route when the latter is larger. This bound is achievable and uses no test labels.
  3. **Stacked upper bound with cross-fitted features.** Use the AUC of a flexible model, such as GBM on p_hat plus all LLM features, with out-of-fold predictions. If this is about AUC(p_hat), no router can help.
  4. **Routing-rate curves.** Plot accuracy or AUC vs the fraction routed, compared with the random-routing curve at each rate.
- A benefit-router with Spearman 0.09-0.40 is in line with the kNN-beats-learned-routers literature. Routers trained on small, noisy "benefit" labels are fragile. A kNN router over code embeddings is a cheap sanity baseline.
- Published cascades work because model 2 is *stronger* on the hard tail. The local precondition, an LLM stronger than the prefilter on hard cases, is unmet (30-40% < 50%). The cascade design is therefore secondary to fixing the LLM's signal, for example with sampling, retrieval or CPG facts, or a fine-tuned LLM.
- Gupta et al. suggest that if routing is kept, router features should include prefilter-side uncertainty: margin, MC-dropout or ensemble variance, and embeddings. They should not rely on aggregate LLM text-level confidence, which carries length and verbosity bias.

### Gaps
- Exact APGR or oracle numbers for RouteLLM, and the FrugalGPT and AutoMix figures, were not verified against full papers in this session.
- I found no paper that explicitly analyzes the "random router beats oracle" artifact. The explanation above is derived arithmetic, not a citation.

---

## Q4. Vulnerability-detection or code-analysis papers that ensemble or fuse LLMs with code models (2023-2026)

### Takeaway
Evidence for gains from LLM plus code-model hybrids in VD is thin and mostly modest:
- Stacking fine-tuned encoders (EnStack) gains about 3-4 accuracy points on Draper VDISC.
- Ensembles of fine-tuned code LLMs help on balanced Devign but stacking hurts on imbalanced ReVeal.
- PLM+GNN hybrids help AUPRC more consistently than F1.
- Distilling LLM explanations into small models improves detection somewhat.

I found **no** paper showing that a *prompted* general LLM's verdict, stacked on a fine-tuned classifier, yields a robust AUC gain on realistic data such as PrimeVul. Prompted LLMs are near chance on realistic VD benchmarks, which is consistent with the local result.

### Cited Findings
- **EnStack (2024, IEEE BigData):** stacks fine-tuned CodeBERT, GraphCodeBERT and UniXcoder with meta-classifiers (LR, SVM, RF, XGBoost) on Draper VDISC. The best combination, GraphCodeBERT+UniXcoder with LR, reaches about **82% accuracy vs about 78.5% for the best single CodeBERT**, a gain of about 3.5 points. This is a single paper and not replicated ([arXiv 2411.16561](https://arxiv.org/abs/2411.16561); [IEEE](https://ieeexplore.ieee.org/document/10825609/)).
- **Sun et al. 2025, LLM ensembles for VD** (details in Q2; [arXiv 2509.12629](https://arxiv.org/pdf/2509.12629)):
  - All ensemble methods have better average ranks than single LLMs.
  - Boosting has the best F1 rank (1.60).
  - On imbalanced ReVeal, Bagging and Stacking F1 fell below baselines in all 5 instances, and only Boosting helped (+34.38% relative recall).
  - Dynamic Gated Stacking, an MoE-style gate that uses code features, beats plain Stacking on imbalanced and multi-class data.
- **PLMGH, "What Matters in PLM-GNN Hybrids for Code Classification and Vulnerability Detection" (2026):** a controlled study. PLM+GNN hybrids do not always win on macro-F1 on Devign but **more consistently improve AUPRC**. The choice of PLM matters more than GNN design, and GNN-only models trail badly ([arXiv 2604.25599](https://arxiv.org/html/2604.25599)).
- **Multimodal fusion baselines on Devign** ([Fisher-guided fusion, arXiv 2601.02438](https://arxiv.org/pdf/2601.02438); [FusionVul, arXiv 2606.08553](https://arxiv.org/html/2606.08553)):
  - Fusion of CodeBERT-based features gives F1 58.11 with gated fusion, 53.87 with concat+MLP, and 53.53 with cross-attention. A unimodal CodeT5-base scores 60.48 and the Devign GNN 56.60, so naive fusion can underperform a single strong encoder.
  - FusionVul reports Devign F1 58.42%, behind CodeBERT-CPP at 61.73%.
- **Explanations or LLM reasoning as training signal** ([R2Vul, arXiv 2504.04699](https://arxiv.org/html/2504.04699); [arXiv 2406.09701](https://arxiv.org/html/2406.09701v4)):
  - R2Vul distills structured reasoning from a teacher LLM into small students via RLAIF/ORPO, contrasting valid and flawed explanations. It reports improved detection, with a 1.5B student rivaling larger models.
  - "Towards Explainable Vulnerability Detection with LLMs" fine-tunes students on teacher explanations and reports improved detection. The teacher matters: average Macro-F1 on PrimeVul was 20.5% with DeepSeek-V3 explanations vs 16.4% with CodeLlama-13B. Absolute performance on PrimeVul stays low.
- **Multi-perspective LLM reasoning** ([ReasonVul, arXiv 2605.18153](https://arxiv.org/html/2605.18153v1); [VulAgent, ACL Findings 2026](https://aclanthology.org/2026.findings-acl.928.pdf)):
  - ReasonVul reports PrimeVul PairAcc 40.00% and F1 72.52%, an 81.24% relative PairAcc gain over the best baseline. These are self-reported preprint numbers.
  - VulAgent's hypothesis-validation multi-agent design compares against fine-tuned CodeBERT on PrimeVul and SVEN. Its numbers were not extracted.
- **Prompting vs fine-tuning Llama on PrimeVul:** the best fine-tuned Llama approach only matches the UniXcoder baseline (AUC 0.842) ([arXiv 2512.09006](https://arxiv.org/html/2512.09006)).
- **Realistic-benchmark ceiling:**
  - PrimeVul (Ding et al., 2024): a 7B code model with 68.26% F1 on BigVul drops to 3.09% F1 on PrimeVul, and GPT-3.5/GPT-4 are near random on paired evaluation **[prior knowledge of the abstract, not re-fetched]** ([arXiv 2403.18624](https://arxiv.org/abs/2403.18624)).
  - GNNs reach at most 17.8% F1 on PrimeVul, and simple code metrics did better ([arXiv 2509.19117](https://arxiv.org/pdf/2509.19117)).
  - Some Devign-trained models predict nearly everything as vulnerable, which inflates F1 ([arXiv 2512.10485](https://arxiv.org/html/2512.10485v2)).
  - Steenhoek et al., "To Err is Machine" (ICSE 2025), found that 14 SOTA LLMs had near-chance balanced accuracy distinguishing vulnerable from patched code, failing at the required semantic reasoning **[prior knowledge, not re-fetched]** ([arXiv 2403.17218](https://arxiv.org/abs/2403.17218)).

### Inferences
- The VD papers that report gains from combining models mostly combine **fine-tuned** models (EnStack, Sun et al., PLM+GNN). They do not combine a *prompted* general LLM with a fine-tuned classifier. The most promising literature-backed ways to inject LLM knowledge into a strong classifier are:
  - Fine-tune the LLM, or distill its explanations into the classifier as in R2Vul and arXiv 2406.09701, rather than stacking its zero-shot verdict.
  - Use boosting or hard-example reweighting rather than LR stacking on imbalanced data.
  - Use gated, MoE-style fusion that conditions on code features.
- Expect gains to show up first in AUPRC rather than F1 or AUC (PLMGH), and to be small, a few points.
- Given near-chance zero-shot LLM performance on realistic paired VD and the local 30-40% hard-case accuracy, the null stacking result is the expected outcome, not a bug.

### Gaps
- No paper found reports AUC gains from stacking a prompted-LLM verdict or confidence onto a fine-tuned VD classifier. This is an open gap and possibly a publishable negative result.
- Vul-LMGNN, GRACE (graph + in-context LLM) and MSIVD (LLM + GNN) are known LLM+GNN hybrids but were not fetched in this session, so their numbers are not reported here.
- I found no VD paper evaluating LLM self-consistency or sampling-based confidence as a feature.
