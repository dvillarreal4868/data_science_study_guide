# Data Science Foundations Study Plan

> A 30-day self-study curriculum for transitioning into clinical multimodal AI, designed for a first-principles learner with a physics/biochemistry background.

**Format:** One topic per day · 2–3 hours per session · Each day includes focus areas, a hands-on exercise, and resources

```
Days 1–3 (Math / Stats / Info Theory)
    │
Day 4 (Optimization) ◄── everything depends on this
    │
Day 5 (Bias-Variance) ─── Day 6 (Classification)
    │                          │
Day 7 (Trees/Ensembles)   Days 8–10 (Neural Net Foundations)
                               │
                    ┌──────────┴──────────┐
                    │                     │
              Days 11–14            Days 15–21
            (Vision Track)        (NLP Track)
                    │                     │
                    └──────────┬──────────┘
                               │
                         Days 22–26
                     (Multimodal Fusion)
                               │
                         Days 27–30
                     (Research Practice)
```

---

## Phase 1: Mathematical & Statistical Foundations (Days 1–7)

Much of this will be review given a physics background, but the goal is to reframe what you know through an ML lens.

- [ ] **Day 1 — Linear Algebra for ML**
  - **Focus:** Matrix multiplication as linear transformation, eigendecomposition, SVD, and PCA as finding directions of maximum variance.
  - **Exercise:** Implement PCA from scratch in NumPy on the Iris dataset, verify against sklearn.
  - **Resource:** 3Blue1Brown's *Essence of Linear Algebra*, then Gilbert Strang's MIT lectures.

- [ ] **Day 2 — Probability, Likelihood, and Bayesian Thinking**
  - **Focus:** MLE as choosing parameters that maximize the probability of observed data. MAP estimation and how adding a prior is mathematically identical to regularization.
  - **Exercise:** Derive the MLE for Gaussian parameters, then add a Gaussian prior on the mean and derive the MAP estimate — observe the shrinkage (this is ridge regression).
  - **Resource:** Bishop's *PRML* chapter 2.

- [ ] **Day 3 — Information Theory Essentials**
  - **Focus:** Entropy as expected surprise (same concept as stat mech). Cross-entropy loss as minimizing KL divergence. Mutual information.
  - **Exercise:** Compute entropy and KL divergence by hand for discrete distributions, show KL is asymmetric and understand the mode-seeking vs. mode-covering distinction.
  - **Resource:** David MacKay's textbook (free online), chapters 2 and 4.

- [ ] **Day 4 — Optimization and Gradient Descent**
  - **Focus:** Convex vs. non-convex. SGD as an unbiased gradient estimator whose variance helps escape shallow minima. Momentum as accumulated velocity (physics analogy). Adam as adaptive per-parameter learning rates.
  - **Exercise:** Implement vanilla GD, SGD, and Adam from scratch to minimize the Rosenbrock function, visualize trajectories on a contour plot.
  - **Resource:** Sebastian Ruder's [optimization overview blog post](https://ruder.io/optimizing-gradient-descent/).

- [ ] **Day 5 — Bias-Variance Tradeoff and Regularization**
  - **Focus:** Derive the bias²-variance-noise decomposition. L2 as Gaussian prior, L1 as Laplace prior.
  - **Exercise:** Fit polynomial regressions of increasing degree to a noisy sine wave, plot the U-shaped train/test error curve, then show L2 regularization recovers good performance at high degree.
  - **Resource:** *ESL* (Hastie et al.) chapters 2–3.

- [ ] **Day 6 — Classification Fundamentals**
  - **Focus:** Logistic regression from first principles — model log-odds as linear, derive sigmoid, show MLE gives cross-entropy loss. Softmax as Boltzmann distribution (same partition function). AUROC, AUPRC, and why accuracy fails for imbalanced clinical data.
  - **Exercise:** Implement logistic regression from scratch, plot the ROC curve by sweeping thresholds.
  - **Resource:** *ESL* chapter 4.

- [ ] **Day 7 — Ensemble Methods and Tree-Based Models**
  - **Focus:** Decision trees, random forests as variance reduction via bagging, gradient boosting as functional gradient descent. These are still often the best models for tabular clinical data.
  - **Exercise:** Train random forest and XGBoost on the UCI Heart Disease dataset, compare to logistic regression, use SHAP for interpretation.
  - **Resource:** *ESL* chapters 9–10, the XGBoost paper.

---

## Phase 2: Deep Learning Foundations (Days 8–14)

- [ ] **Day 8 — Neural Networks from Scratch**
  - **Focus:** The perceptron, MLPs, universal approximation theorem, ReLU and vanishing gradients.
  - **Exercise:** Implement a 2-layer MLP from scratch in NumPy with hand-coded backprop, train on MNIST. **This is the single most valuable exercise in the plan.**
  - **Resource:** Nielsen's [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/), chapters 1–2.

- [ ] **Day 9 — Backpropagation and Computational Graphs**
  - **Focus:** Backprop as reverse-mode autodiff. Gradient flow, vanishing/exploding gradients.
  - **Exercise:** Draw a computational graph by hand, compute forward/backward passes manually, then verify in PyTorch against `tensor.grad`.
  - **Resource:** Stanford CS231n backprop notes.

- [ ] **Day 10 — Training Deep Networks in Practice**
  - **Focus:** Xavier/He initialization (preserving variance across layers). Batch norm, learning rate schedules, dropout, early stopping.
  - **Exercise:** Port your MNIST MLP to PyTorch and systematically ablate — remove batch norm, break initialization, use a huge learning rate. Document the failure modes.
  - **Resource:** Goodfellow et al. *Deep Learning* chapters 7–8.

- [ ] **Day 11 — Convolutional Neural Networks**
  - **Focus:** Convolution as translation-equivariant feature extraction. ResNet and skip connections.
  - **Exercise:** Build a CNN for CIFAR-10, then fine-tune a pretrained ResNet, compare. Visualize first-layer filters.
  - **Resource:** CS231n CNN notes, the [ResNet paper](https://arxiv.org/abs/1512.03385).

- [ ] **Day 12 — Transfer Learning and Feature Extraction**
  - **Focus:** Why ImageNet features transfer to medical imaging. Fine-tuning strategies.
  - **Exercise:** Take a pretrained ResNet and fine-tune on PneumoniaMNIST or ChestX-ray8. Compare training from scratch vs. frozen features vs. full fine-tuning.
  - **Resource:** Yosinski et al. [*How transferable are features in deep networks?*](https://arxiv.org/abs/1411.1792) (2014).

- [ ] **Day 13 — Medical Image Segmentation (U-Net)**
  - **Focus:** Encoder-decoder with skip connections. Dice loss for class imbalance. Medical-specific augmentation.
  - **Exercise:** Implement U-Net for a lung segmentation task (LUNA16 or similar).
  - **Resource:** The original [U-Net paper](https://arxiv.org/abs/1505.04597).

- [ ] **Day 14 — Evaluation and Uncertainty in Medical AI**
  - **Focus:** Calibration curves, temperature scaling. Epistemic vs. aleatoric uncertainty. MC dropout.
  - **Exercise:** Plot calibration for your Day 12 model before/after temperature scaling, implement MC dropout (50 forward passes) for uncertainty estimation.
  - **Resource:** Guo et al. [*On Calibration of Modern Neural Networks*](https://arxiv.org/abs/1706.04599) (2017).

---

## Phase 3: NLP and Language Models (Days 15–21)

- [ ] **Day 15 — Word Embeddings**
  - **Focus:** Word2Vec, distributional hypothesis, negative sampling.
  - **Exercise:** Train Word2Vec on a corpus, explore the embedding space with nearest neighbors and t-SNE.
  - **Resource:** Mikolov et al.; Jay Alammar's [*Illustrated Word2Vec*](https://jalammar.github.io/illustrated-word2vec/).

- [ ] **Day 16 — RNNs and LSTMs**
  - **Focus:** RNNs as networks with memory. Derive why gradients vanish with sequence length. LSTMs: gating as controlling information flow (cell state as information highway, analogous to ResNet skip connections).
  - **Exercise:** Implement RNN and LSTM for text classification, observe the difference on long sequences.
  - **Resource:** Colah's [*Understanding LSTM Networks*](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).

- [ ] **Day 17 — Transformer Deep Dive**
  - **Focus:** Multi-head attention, positional encodings, the residual stream view (each layer makes additive updates). Feed-forward layers as the "memory" storing facts.
  - **Exercise:** Implement a transformer encoder block from scratch in PyTorch, visualize attention patterns.
  - **Resource:** [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762); Jay Alammar's [*Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/).

- [ ] **Day 18 — BERT and Masked Language Modeling**
  - **Focus:** Masked LM pretraining, fine-tuning with [CLS] token, BPE tokenization. Clinical variants: ClinicalBERT, BioBERT, PubMedBERT.
  - **Exercise:** Fine-tune ClinicalBERT on a clinical NLP task using HuggingFace.
  - **Resource:** [BERT paper](https://arxiv.org/abs/1810.04805); [HuggingFace free course](https://huggingface.co/learn/nlp-course).

- [ ] **Day 19 — Autoregressive LMs (GPT-style)**
  - **Focus:** Causal attention masking, scaling laws, in-context learning, temperature/top-k/top-p sampling.
  - **Exercise:** Use a GPT-style model for few-shot clinical information extraction from notes. Compare zero-shot vs. few-shot.
  - **Resource:** GPT-3 paper (methodology sections).

- [ ] **Day 20 — Clinical NLP Specifics**
  - **Focus:** Abbreviations, negation detection, temporal reasoning, medical ontologies (UMLS, SNOMED, ICD).
  - **Exercise:** Build an extraction pipeline with medspaCy, then compare to an LLM-based approach.
  - **Resource:** [medspaCy documentation](https://github.com/medspacy/medspacy).

- [ ] **Day 21 — RAG and Prompt Engineering**
  - **Focus:** Retrieval-augmented generation, vector databases, chunking strategies, grounding and hallucination reduction.
  - **Exercise:** Build a simple RAG system over clinical guidelines, evaluate answer faithfulness.
  - **Resource:** Lewis et al. (2020); [Anthropic's prompting docs](https://docs.anthropic.com).

---

## Phase 4: Multimodal Learning (Days 22–26)

*This is where everything converges on the lab's work.*

- [ ] **Day 22 — Contrastive Learning and CLIP**
  - **Focus:** SimCLR, InfoNCE loss, CLIP's shared embedding space.
  - **Exercise:** Use pretrained CLIP for zero-shot classification on chest X-rays with text descriptions of pathologies.
  - **Resource:** [CLIP paper](https://arxiv.org/abs/2103.00020); [SimCLR paper](https://arxiv.org/abs/2002.05709).

- [ ] **Day 23 — Vision Transformers**
  - **Focus:** ViT as patches-as-tokens. Tradeoffs vs. CNNs (less inductive bias, needs more data, scales better).
  - **Exercise:** Fine-tune ViT on medical imaging, compare to your Day 12 ResNet, visualize attention maps.
  - **Resource:** [*An Image is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929).

- [ ] **Day 24 — Multimodal Fusion Architectures**
  - **Focus:** Early fusion, late fusion, cross-attention fusion. When to use each.
  - **Exercise:** Build a classifier that takes both an image and tabular clinical features, try late fusion and cross-attention, compare to unimodal baselines.
  - **Resource:** Baltrušaitis et al. *Multimodal ML: A Survey and Taxonomy* (2019).

- [ ] **Day 25 — LLMs for EHR Extraction**
  - **Focus:** Prompt-based extraction, fine-tuning, hybrid pipelines, output validation against medical ontologies.
  - **Exercise:** Design and test an extraction pipeline that pulls structured fields from clinical notes, with a validation layer. Measure precision/recall.
  - **Resource:** Recent papers on GPT-4 for clinical IE; MADE 1.0 challenge.

- [ ] **Day 26 — End-to-End Multimodal Clinical Prediction**
  - **Focus:** Combine lung imaging + clinical text + structured data. Handle missing modalities. Deployment considerations: FDA, explainability, prospective validation.
  - **Exercise:** Prototype a full pipeline for a lung-related diagnostic task, even with small/synthetic data.
  - **Resource:** Med-PaLM M paper; recent multimodal medical imaging surveys.

---

## Phase 5: Research Practices (Days 27–30)

- [ ] **Day 27 — Experiment Tracking**
  - **Focus:** wandb/MLflow, reproducibility, Docker, Hydra configs.
  - **Exercise:** Re-run an earlier experiment with full wandb tracking and a hyperparameter sweep.
  - **Resource:** [Weights & Biases docs](https://docs.wandb.ai/).

- [ ] **Day 28 — Clinical Data Engineering**
  - **Focus:** FHIR, HL7, DICOM. Handling missing values, temporal alignment, de-identification. **Critical:** never split patients across train/test, use temporal splits.
  - **Exercise:** Build a preprocessing pipeline on MIMIC or synthetic data.
  - **Resource:** [MIMIC-III documentation](https://mimic.mit.edu/); MIT 6.S897 course materials.

- [ ] **Day 29 — Interpretability**
  - **Focus:** SHAP, LIME, Grad-CAM, attention visualization. What clinicians actually need to trust a model.
  - **Exercise:** Apply Grad-CAM to your imaging model and SHAP to your tabular model, show results to your PI.
  - **Resource:** Molnar's [*Interpretable Machine Learning*](https://christophm.github.io/interpretable-ml-book/) (free online).

- [ ] **Day 30 — Reading Papers**
  - **Focus:** Abstract → figures → conclusion → methods. Key venues: NeurIPS, MICCAI, ML4H, Nature Medicine.
  - **Exercise:** Read 3 papers relevant to your lab, write a one-paragraph summary and critical analysis for each.
  - **Resource:** Set up an arXiv feed; follow Papers with Code newsletter.

---

## Pacing Notes

- **Don't skip the exercises.** Hands-on implementation is where real understanding forms.
- **Days 1–6 may feel fast** given a physics background — focus on the ML-specific reframing, not the raw math.
- **Days 8–9 are the hardest and most important.** Implementing backprop from scratch pays the highest dividends. Budget extra time.
- **Rest days are fine.** This is 30 sessions, not necessarily 30 consecutive days. Spacing over 6–8 weeks is reasonable.
- **After Day 21**, Phases 4 and 5 can be interleaved based on lab priorities.

---

## Repo Structure

```
.
├── README.md
├── phase-1-foundations/
│   ├── day-01-linear-algebra/
│   ├── day-02-probability/
│   ├── ...
│   └── day-07-ensembles/
├── phase-2-deep-learning/
│   ├── day-08-nn-from-scratch/
│   ├── ...
│   └── day-14-uncertainty/
├── phase-3-nlp/
│   ├── day-15-word-embeddings/
│   ├── ...
│   └── day-21-rag/
├── phase-4-multimodal/
│   ├── day-22-clip/
│   ├── ...
│   └── day-26-end-to-end/
└── phase-5-research-practices/
    ├── day-27-experiment-tracking/
    ├── ...
    └── day-30-reading-papers/
```
