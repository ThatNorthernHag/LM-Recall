# Mathematical Foundations of the Structure Function Experiment

**Theory Document — Companion to the MVP Report**
April 2026

---

## Overview

The MVP experiment measures something specific: how factual recall degrades as context is compressed, across content types the model knows well, doesn't know at all, and partially knows. The results — flat curves for known content, steep curves for novel, intermediate for mixed — are not just an empirical curiosity. They are predicted by two mathematical frameworks developed decades before LLMs existed.

This document explains those frameworks, defines the formal objects, and shows how the experiment approximates them.

---

## 1. The Problem, Stated Formally

A frozen language model *f* accepts at most *C* tokens as input and produces an output. You have information *I* that exceeds *C*. You need to select a subset — a compressed context — such that *f* applied to the subset approximates what *f* would produce given all of *I*.

Two questions arise:

1. **The outer problem:** Which pieces of information should be in the context window? (Which experiments are sufficient?)
2. **The inner problem:** For each piece of content, can the model reconstruct it from a sparse cue, or must it be preserved verbatim? (What is each item's compression curve?)

The first question belongs to **Blackwell and Le Cam** (comparison of experiments, deficiency). The second belongs to **Vereshchagin and Vitányi** (the algorithmic structure function). The MVP experiment tests predictions from both.

---

## 2. Conditional Kolmogorov Complexity

### Definition

For a string *x* and a fixed model *f*, the **conditional Kolmogorov complexity** *K*(*x* | *f*) is the length of the shortest program that produces *x* when given *f* on an auxiliary tape.

This is the irreducible information content of *x* that *f* cannot supply.

- If *K*(*x* | *f*) is **small**, then *f* already "knows" *x* — a short program plus the model suffices to reconstruct the content. A sparse pointer is enough.
- If *K*(*x* | *f*) is **large**, then *x* contains substantial information that *f* cannot provide. The content must be preserved.

### Algorithmic Mutual Information

The **algorithmic mutual information** between the model and the content is:

> *I*(*f* : *x*) = *K*(*x*) − *K*(*x* | *f*)

High mutual information means the model captures most of the content's complexity. Low mutual information means the content is largely independent of the model. This quantity was formalized by Gács, Tromp, and Vitányi (2001), building on Kolmogorov's 1968 foundations.

### What This Means for Context Selection

Every piece of content in a context window sits somewhere on the spectrum from *K*(*x* | *f*) ≈ 0 (the model knows it completely) to *K*(*x* | *f*) ≈ *K*(*x*) (the model knows nothing about it). But *K*(*x* | *f*) is a single number — it tells you the *total* novelty. It does not tell you the *structure* of that novelty: how much compression budget you need before information loss begins, or where the transition from lossless to lossy occurs.

For that, you need the structure function.

---

## 3. The Vereshchagin-Vitányi Structure Function

### Definition

Vereshchagin and Vitányi (2004) define the **conditional structure function**:

> *h*<sub>*x*|*f*</sub>(*α*) = min{log |*S*| : *x* ∈ *S*, *K*(*S* | *f*) ≤ *α*}

In words: given a complexity budget *α* (how many bits you can spend describing a set), what is the smallest set *S* containing *x* that can be described in at most *α* bits, using the model *f* as side information?

The function traces a curve:

- At *α* = 0, no description budget is available. *S* must be the set of all strings. *h* is maximal.
- As *α* increases, you can specify tighter sets containing *x*, and *h* decreases.
- The curve records exactly how compressible *x* is at each level of descriptive investment.

### The Sufficiency Line

The **sufficiency line** is:

> *L*(*α*) = *K*(*x* | *f*) − *α*

Where *h*<sub>*x*|*f*</sub> first touches *L* locates the ***f*-relative algorithmic minimal sufficient statistic**: the point at which the model-relative description has captured all the structure in *x* that *can* be captured. Below this point, further compression is provably lossy — you are discarding signal, not noise.

The selection-relevant distinction is the **level** of the curve, not where it first touches *L*:

- **Content whose curve sits low** — small *K*(*x* | *f*) — is pointer-compressible: a brief description, combined with the model, recovers everything.
- **Content whose curve rides the full line *K*(*x* | *f*) ≈ |*x*|** is irreducibly novel: you need to preserve almost the entire string.
- **The touch point is orthogonal:** where *h* first meets *L* measures non-stochasticity, not novelty. A maximally novel (random-given-model) string is maximally *stochastic* — its curve *is* the line, touched at *α* ≈ 0 and ridden at slope −1; a *late* touch marks an exotic, structured-but-non-stochastic string.

### The Staircase Theorem

Vereshchagin and Vitányi's main theorem proves that **every non-increasing integer-valued staircase shape within the admissible region — bounded above by |*x*| and below by the sufficiency line *K*(*x* | *f*) − *α* — is realized, to logarithmic precision, as the structure function of some string**.

The structure function is not smooth. It is not sigmoidal. It does not transition cleanly from "pointer-compressible" to "preserve-worthy." Different data objects produce different staircase shapes — different step locations, different step heights, different numbers of steps.

This has a direct consequence: any method that collapses the structure function to a single number (an importance score, a keep/discard threshold) is discarding decision-relevant information. The full curve tells you how much budget each piece of content needs. A single threshold treats all content as if it had the same curve shape. The staircase theorem says this is wrong in the strongest possible sense.

---

## 4. Blackwell's Comparison of Experiments

### Definition

In 1951, David Blackwell asked: when is one statistical experiment better than another?

An **experiment** in Blackwell's sense is a channel — a way of generating data about something you are trying to decide. Blackwell's theorem gives the answer:

> Experiment *E*₁ is at least as good as experiment *E*₂ for **every possible decision problem** if and only if *E*₂ can be obtained from *E*₁ by post-processing — by applying a stochastic transformation (a "garbling") to *E*₁'s output.

If you can get *E*₂'s data by taking *E*₁'s data and adding noise, then *E*₁ is at least as informative. The converse holds: if *E*₁ is better for every decision, then *E*₂ must be a garbling of *E*₁.

### Application to Context Selection

Map this onto the problem:

- The **full information stream** is the reference experiment — the best data you could hand the model.
- The **compressed context** you actually select is a candidate experiment — a reduced view.
- Every compression method (token pruning, retrieval, summarization) is a **garbling** — it takes one experiment and produces a less informative one via post-processing.

Blackwell's theorem gives a partial order over compression methods: method A dominates method B if and only if B's output can always be obtained by garbling A's output. This is a structural comparison, not an empirical one.

---

## 5. Le Cam Deficiency

### Definition

Le Cam (1964) asked the follow-up: what if you don't care about every decision problem — what if you care about one specific decision?

**Le Cam deficiency** *δ*(*E*₁, *E*₂; *D*) measures how much worse experiment *E*₂ is than experiment *E*₁ for a particular decision problem *D*. Where Blackwell gives a total order (one experiment dominates for all decisions or it doesn't), Le Cam gives a targeted measure: this experiment is *this much* worse than that one, *for this specific purpose*.

### Application to Context Selection

The single decision problem we care about is: **emulate the output of *f***. Le Cam deficiency becomes:

> *δ*(selected context, full stream; emulate *f*)

This is literally the quantity to minimize. It is not an analogy or a metaphor. The objects are the same objects. The optimization target is the same optimization target. The selected context is a statistical experiment. The full stream is the reference experiment. The model's output quality is the decision criterion.

### Sufficient Statistics and Attention

A **sufficient statistic** is one that preserves everything decision-relevant about the data. In a transformer, each attention head computes a weighted sum: softmax attention scores applied to value vectors. This attention output is a statistic of the context.

Whether this statistic is sufficient — whether the attention output captures all the decision-relevant information — is exactly Le Cam's question. If attention loses something that matters for the output, the context is deficient, and *δ* measures how deficient.

---

## 6. How the Two Frameworks Connect

The two frameworks address different aspects of the same problem:

| | Blackwell / Le Cam | Vereshchagin-Vitányi |
|---|---|---|
| **Question** | Which pieces of information belong in the context window? | For each piece, how much compression can it tolerate? |
| **Formal object** | Le Cam deficiency *δ* | Structure function *h*<sub>*x*\|*f*</sub>(*α*) |
| **What it measures** | How much worse is the compressed context than the full stream, for the specific task of emulating *f*? | At each compression budget *α*, how much can *x* be compressed given *f*? |
| **Scope** | Outer optimization — selection across items | Inner optimization — representation of each item |
| **Key result** | Blackwell's theorem: dominance = garbling | Staircase theorem: every staircase shape occurs |

Together they define the full problem: Blackwell/Le Cam says which pieces belong in the window; the structure function says how to represent each piece once it's in.

---

## 7. From Non-Computable Theory to Measurable Experiment

### The Computability Problem

Kolmogorov complexity *K* is not computable. This is not a technicality. Vereshchagin and Vitányi (2004) proved — in the appendices of the structure-function paper — that locating the minimal sufficient statistic on the structure function curve is not even approximable; you cannot get arbitrarily close by running longer computations.

So why build an experiment around it?

Because the framework tells you **what you should be approximating**, even when you cannot compute the exact answer.

### The Approximation Chain

The MVP experiment implements a chain of approximations, each step trading theoretical purity for measurability:

**Step 1: K(*x* | *f*) → cross-entropy**

The cross-entropy of *x* under model *f* — the quantity −log *f*(*x*) — approximates *K*(*x* | *f*) up to a constant, under standard assumptions. Delétang et al. (2023) established this connection rigorously in "Language Modeling Is Compression," showing that language models are compressors with empirical rates that track Kolmogorov-complexity-based bounds.

Low cross-entropy = the model predicts this content well = pointer-compressible.
High cross-entropy = the content is surprising = novel information the model doesn't have.

**Step 2: Structure function curve → compression-retention curve**

The structure function *h*<sub>*x*|*f*</sub>(*α*) traces how compressible *x* is at each budget *α*. We approximate this by physically compressing the context (removing sentences) and measuring what the model retains. Each compression level is a point on the curve. The full set of points traces the approximate shape.

This is not computing the structure function. It is measuring the *empirical signature* the structure function predicts: different content types should produce different curve shapes.

**Step 3: Kolmogorov-optimal compression → sentence-level random sampling**

True algorithmic compression would remove the least informative bits first. We use a crude proxy: random sentence removal. This is noisier but still reveals the curve shape, because:

- For known content, *any* compression level still allows the model to reconstruct from weights, so the curve stays flat regardless of which sentences are removed.
- For novel content, *every* sentence carries irreplaceable information, so removal at any level causes degradation.
- The *shape difference* between curve types survives even crude compression.

**Step 4: Exact information retention → keyword matching**

We score retention by checking whether expected keywords appear in the model's response. This is a coarse measure of information preservation — but sufficient to distinguish flat curves from steep ones, which is the qualitative prediction we are testing.

### What the Chain Preserves and What It Loses

| Theoretical Object | Experimental Proxy | What's Preserved | What's Lost |
|---|---|---|---|
| *K*(*x* \| *f*) | Cross-entropy (implicit) | Relative ordering: low-K content ≈ high-retention at zero context | Exact bit counts |
| *h*<sub>*x*\|*f*</sub>(*α*) | Retention at 5 compression levels | Curve shape (flat vs. steep vs. intermediate) | Fine-grained staircase structure |
| Algorithmic minimal sufficient statistic | Zero-context baseline | Whether the model can reconstruct without context | Exact location on the sufficiency line |
| Staircase theorem (all shapes occur) | Three observed category shapes | Different shapes exist and are measurable | We only observed 3 of infinitely many |

The experiment does not compute the structure function. It tests the structure function's qualitative predictions. The predictions hold.

---

## 8. What the Theory Predicts and What We Observed

### Prediction 1: Known content → flat curves

If the model's weights already encode *x* (high algorithmic mutual information *I*(*f* : *x*)), then *K*(*x* | *f*) is small. The structure function hits the sufficiency line at small *α*. Compressing the context should cost little, because the model can reconstruct from weights.

**Observed:** Known-content curves are flat. Average drop from full to quarter context: 25%. Zero-context retention: 75%. Marie Curie scores 100% at both full and zero context, and knitting basics scores 100% full and 80% at zero context — the model's unconstrained weight knowledge matches or exceeds what the context passage prompts.

### Prediction 2: Novel content → steep curves

If the model has never seen *x* (low *I*(*f* : *x*)), then *K*(*x* | *f*) ≈ *K*(*x*). The structure function stays far above the sufficiency line until *α* approaches |*x*|. Every bit of compression is lossy.

**Observed:** Novel-content curves are steep and largely monotonic. Average drop: 70%. Zero-context retention: 15%. The residual 15% comes from coincidental keyword hits (generic words like "cooperative" or "accessibility" appearing in the model's uncontextualised response); the fabricated specifics themselves are absent.

### Prediction 3: Mixed content → steep curves when queried facts are the novel specifics

Content combining known concepts with novel specifics produces a curve whose steepness depends on *what is being queried*. When the queried facts are the novel specifics (fabricated names, numbers, invented mechanisms), mixed content drops as steeply as — or more steeply than — pure novel content, because the known conceptual anchors do not help retrieve the fictional specifics.

**Observed:** Mixed-content curves drop from 90% to 13% to 0%. Both mixed items hit exactly 0% at zero context. The queried facts (EchoScore, SeedPool, specific percentages) are the fabricated specifics; the known concepts that anchor them (GraphSAGE, Cortex-M, proc macros) do not survive as substrings in the model's zero-context response when the question asks for the fictional system by name.

### Prediction 4: Different shapes → different optimal allocations

The staircase theorem says every staircase shape occurs, so no single threshold is correct across all content. A selector that reads curve shapes should outperform one that applies a uniform threshold.

**Observed:** The curve-aware selector outperforms uniform allocation by 35% on the same budget. The mechanism is exactly what the theory predicts: flat curves tolerate aggressive compression, freeing budget for steep curves that need full context.

### Prediction 5 (Le Cam): Compression = garbling = measurable deficiency

Each compression level defines a garbled experiment. Le Cam deficiency measures how much worse the garbled experiment is than the original. Known content has low deficiency under compression (the model's weights compensate). Novel content has high deficiency (nothing compensates for the removed information).

**Observed:** The retention scores *are* empirical deficiency measures. Known-content compression from 100% to 25% barely changes the experiment's informativeness (25% deficiency). Novel-content compression creates massive deficiency (70%), and mixed-content even more (77%) when the queried facts are the novel parts.

---

## 9. The Non-Computability Story

The honest version: we cannot compute the structure function exactly, and we never will. Kolmogorov complexity is non-computable. The minimal sufficient statistic is not even approximable in the strict sense (Vereshchagin–Vitányi 2004).

But this does not make the framework useless. It makes it a *map* rather than a *GPS*.

**What the non-computable framework gives you:**

- **The right shape of the object.** The thing you are trying to approximate is a curve, not a number. Current importance-scoring methods collapse the curve to a single score. Knowing the target is a curve tells you to measure at multiple points.
- **Qualitative predictions.** Different content produces different curve shapes. This is testable without computing *K*.
- **Optimality criteria.** A selector that respects curve shapes beats one that ignores them. This is demonstrable.
- **A named target for approximation.** Perplexity-based scoring is a one-point estimate of the structure function at a single *α*. Knowing the target tells you the estimate is incomplete.

**Computable paths forward:**

- **MDL (Minimum Description Length):** The computable-in-practice version of algorithmic statistics. Uses shortest descriptions within a restricted model class rather than a universal Turing machine. Grünwald (2007) gives the full treatment. The caveat: under model misspecification (guaranteed when applying a frozen LLM to arbitrary user data), MDL can be asymptotically inconsistent (Grünwald and Langford 2007).
- **Cross-entropy as proxy for K(*x* | *f*):** Delétang et al. (2023) showed LLM compression rates track Kolmogorov-complexity-based bounds. Per-token cross-entropy gives an empirical handle on conditional complexity.
- **Multi-point compression curves:** What the MVP experiment does — measuring retention at multiple compression levels — gives empirical approximations of the curve shape, bypassing the need to compute *K* exactly.

---

## 10. What This Framework Does Not Claim

**It does not claim the pointer-vs-preserve problem is solved.** The framework *defines* the problem precisely. It identifies the formal object (the structure function) and gives qualitative predictions (different shapes, staircase structure). It does not hand you an algorithm.

**It does not claim existing methods are wrong.** Perplexity-based scoring, learned rerankers, and retrieval systems all approximate aspects of the structure function without knowing what they are approximating. The claim is that knowing the target improves the approximations — not that the approximations are useless.

**It does not claim the MVP experiment computed the structure function.** We measured an empirical proxy. The proxy is coarse (sentence-level compression, keyword scoring, 10 items, 3 runs). The qualitative predictions of the theory hold in this proxy. A larger, finer-grained experiment would test the predictions more rigorously.

**It does not claim individual-string analysis replaces ensemble benchmarks.** Vereshchagin and Vitányi (2010) proved that individual strings can have rate-distortion profiles richer than any ensemble average. This matters: a compression method that scores well on average can fail on specific content. But ensemble benchmarks remain useful for aggregate evaluation. The structure function adds per-item resolution, it does not replace aggregate measures.

---

## References

### Algorithmic Information Theory

Kolmogorov, A.N. (1968). "Three Approaches to the Quantitative Definition of Information." *International Journal of Computer Mathematics* 2(1-4), 157-168.

Gács, P., Tromp, J., and Vitányi, P.M.B. (2001). "Algorithmic Statistics." *IEEE Transactions on Information Theory* 47(6), 2443-2463.

Li, M. and Vitányi, P.M.B. (2008). *An Introduction to Kolmogorov Complexity and Its Applications*, 3rd ed. Springer.

### Structure Function

Vereshchagin, N.K. and Vitányi, P.M.B. (2004). "Kolmogorov's Structure Functions and Model Selection." *IEEE Transactions on Information Theory* 50(12), 3265-3290.

Vereshchagin, N.K. and Vitányi, P.M.B. (2010). "Rate Distortion and Denoising of Individual Data Using Kolmogorov Complexity." *IEEE Transactions on Information Theory* 56(7), 3438-3454.

Vereshchagin, N.K. and Shen, A. (2017). "Algorithmic Statistics: Forty Years Later." In *Computability and Complexity*, Springer, 669-737.

### Comparison of Experiments and Deficiency

Blackwell, D. (1951). "Comparison of Experiments." *Proceedings of the Second Berkeley Symposium on Mathematical Statistics and Probability*, 93-102.

Blackwell, D. (1953). "Equivalent Comparisons of Experiments." *Annals of Mathematical Statistics* 24(2), 265-272.

Le Cam, L. (1964). "Sufficiency and Approximate Sufficiency." *Annals of Mathematical Statistics* 35(4), 1419-1455.

Le Cam, L. (1986). *Asymptotic Methods in Statistical Decision Theory*. Springer.

Torgersen, E. (1991). *Comparison of Statistical Experiments*. Cambridge University Press.

Mu, X., Pomatto, L., Strack, P., and Tamuz, O. (2021). "From Blackwell Dominance in Large Samples to Rényi Divergences and Back Again." *Econometrica* 89(1), 475-506.

### Compression and Language Models

Delétang, G., Ruoss, A., Duquenne, P.-A., et al. (2023). "Language Modeling Is Compression." arXiv:2309.10668.

### Minimum Description Length

Grünwald, P.D. (2007). *The Minimum Description Length Principle*. MIT Press.

Grünwald, P.D. and Langford, J. (2007). "Suboptimal Behavior of Bayes and MDL in Classification Under Misspecification." *Machine Learning* 66(2-3), 119-149.

Grünwald, P.D. and van Ommen, T. (2017). "Inconsistency of Bayesian Inference for Misspecified Linear Models, and a Proposal for Repairing It." *Bayesian Analysis* 12(4), 1069-1103.

Wallace, C.S. (2005). *Statistical and Inductive Inference by Minimum Message Length*. Springer.

### Adjacent Frameworks (Referenced for Contrast)

Tishby, N., Pereira, F.C., and Bialek, W. (1999). "The Information Bottleneck Method." *Proceedings of the 37th Annual Allerton Conference*, 368-377.

Kolchinsky, A., Tracey, B.D., and Van Kuyk, S. (2019). "Caveats for Information Bottleneck in Deterministic Scenarios." *Proceedings of ICLR*.

Bernardo, J.M. and Smith, A.F.M. (1994). *Bayesian Theory*. Wiley. Chapter 4.
