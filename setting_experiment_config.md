# Complete Experiment Specification & Benchmark Guide

> **Paper Reference:**  
> *"An Illusion of Unlearning? Assessing Machine Unlearning Through Internal Representations"*  
> **Authors:** Yichen Gao, Altay Unal, Akshay Rangamani, Zhihui Zhu (*AISTATS 2026*, arXiv:2604.08271v1)  
> **Official Codebase:** `https://github.com/ycgao1/CMF_Unlearning`

---

## 1. Executive Protocol & Metric Architecture

The paper investigates the **"Illusion of Unlearning"**: where an unlearned model achieves $\approx 0\%$ accuracy on the forget class at the output level, but its deep representations still encode the forgotten class and can be recovered via feature probing.

```
                  ┌───────────────────────────────────────────────────────────────┐
                  │                        Input Image x                          │
                  └──────────────────────────────┬────────────────────────────────┘
                                                 │
                                                 ▼
                  ┌───────────────────────────────────────────────────────────────┐
                  │                  Backbone Encoder φ_θ(x)                      │
                  │             (Penultimate Layer / avgpool: D=512)              │
                  └──────────────┬────────────────────────────────┬───────────────┘
                                 │                                │
                [Trained / Unlearned Head (W, b)]      [Frozen Feature Extraction]
                                 │                                │
                                 ▼                                ▼
                    ┌─────────────────────────┐      ┌─────────────────────────┐
                    │     Output Accuracy     │      │ Feature Representations │
                    │ (Shallow / Model Preds) │      └───────┬─────────┬───────┘
                    └─────────────────────────┘              │         │
                                                             ▼         ▼
                                                ┌────────────────┐ ┌────────────────┐
                                                │  Linear Probe  │ │      NCC       │
                                                │ (Trained Head) │ │(Training-free) │
                                                └────────────────┘ └────────────────┘
```

### The 3 Evaluation Metrics (§3.1, §3.2)
1. **Output Accuracy (`output_retain_acc`, `output_forget_acc`)**:
   - Direct forward pass through full model ($f(x) = W \phi_\theta(x) + b$) on held-out test set ($D_r^{\text{test}}$ and $D_f^{\text{test}}$).
2. **Linear Probe Accuracy (`probe_retain_acc`, `probe_forget_acc`)**:
   - Freeze the encoder $\phi_\theta$.
   - Extract raw features from **all training samples** $D = D_r^{\text{train}} \cup D_f^{\text{train}}$.
   - Train a fresh single linear layer $\hat{y} = W_{\text{probe}} z + b_{\text{probe}}$ with SGD (lr=0.01, momentum=0.9).
   - Evaluate the trained linear probe on test-set features ($D_r^{\text{test}}$ and $D_f^{\text{test}}$).
3. **Nearest Class Center (NCC) Accuracy (`ncc_retain_acc`, `ncc_forget_acc`)**:
   - Training-free feature classification (Eq. 5):
     $$\text{NCC} := \mathbb{P}\left[ y = \arg\min_k \|\phi_\theta(x) - \mu_k\|^2 \right]$$
   - Class centroids $\mu_k = \frac{1}{|D_k|} \sum_{x \in D_k} \phi_\theta(x)$ are computed from the full training set $D$.

---

## 2. Dataset & Forget/Retain Split Configurations (§A.2, §A.3)

| Dataset | Total Classes | Image Size | Train / Test Split | Single-Class Forgetting | Multi-Class Forgetting |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **CIFAR-10** | 10 | $32 \times 32$ | 50k / 10k | Sweep all 10 classes: `{0}`, `{1}`, ..., `{9}` | 5 representative 3-class groups:<br>• `{0, 1, 2}`<br>• `{3, 4, 5}`<br>• `{6, 7, 8}`<br>• `{0, 5, 9}`<br>• `{2, 4, 8}` |
| **CIFAR-100** | 100 (20 coarse $\times$ 5 fine) | $32 \times 32$ | 50k / 10k | 5 superclass-disjoint classes:<br>`{0}`, `{1}`, `{2}`, `{3}`, `{5}`<br>*(Class 4 omitted as it shares coarse class with 1)* | 5 groups of 10 classes (union of 2 coarse classes):<br>• `{3, 15, 19, 21, 31, 38, 42, 43, 88, 97}`<br>• `{47, 52, 54, 56, 59, 62, 70, 82, 92, 96}`<br>• `{5, 20, 22, 25, 39, 40, 84, 86, 87, 94}`<br>• `{8, 13, 41, 48, 59, 69, 81, 85, 89, 90}`<br>• `{1, 4, 30, 32, 55, 67, 72, 73, 91, 95}` |
| **Tiny-ImageNet** | 200 | $64 \times 64$ | 100k / 10k | 5 representative classes:<br>`{2}`, `{3}`, `{5}`, `{7}`, `{9}` | 5 contiguous groups of 20 classes:<br>• `{0..19}`<br>• `{20..39}`<br>• `{40..59}`<br>• `{60..79}`<br>• `{80..99}` |

*Data Augmentation for Training:* RandomCrop(padding=4 for CIFAR, padding=8 for Tiny-ImageNet) + RandomHorizontalFlip. (For ViT: Resize to $224 \times 224$).

---

## 3. Pretraining & Gold Standard Retrain (§A.4, §A.5)

### Original Model Training ($\Theta_o$)
- **ResNet-18 (CIFAR-10 / CIFAR-100)**: Train from scratch, 300 epochs, early stopping patience 50, batch size 128, SGD (momentum 0.9, weight decay $5 \times 10^{-4}$), 5-epoch linear warmup + cosine decay to min LR $1 \times 10^{-5}$. Initial LR = $0.01$ (Table 4) / $0.05$ (§A.4).
- **ResNet-50 (Tiny-ImageNet)**: Train from scratch, 300 epochs, batch size 128, SGD (LR = 0.05, momentum 0.9, WD = $5 \times 10^{-4}$).
- **ViT-S/16 (All Datasets)**: ImageNet-pretrained backbone fine-tuned for 10 epochs, batch size 128, SGD (momentum 0.9), LR = $3 \times 10^{-4}$ (CIFAR-10/100) or $1 \times 10^{-4}$ (Tiny-ImageNet).

### Retain-only Retrain Baseline (Gold Standard $\Theta_r$)
- Fresh model trained from scratch using **only** $D_r^{\text{train}}$:
  - **CIFAR-10 ResNet-18**: 200 epochs, batch 128, LR = 0.01, WD = $5 \times 10^{-4}$, val-ratio = 0.1.
  - **CIFAR-100 ResNet-18**: 200 epochs, batch 128, LR = 0.01, WD = $5 \times 10^{-4}$.
  - **Tiny-ImageNet ResNet-50**: 150 epochs, batch 256, LR = 0.05, WD = $5 \times 10^{-4}$.
  - **ViT-S/16**: 10 epochs, batch 128 on retain set with same optimizer.

---

## 4. Unlearning Algorithms Formulation

1. **Random Label (RL)**: Relabels forget samples $D_f$ with uniform random retain labels $\tilde{y} \in C_r$.
2. **SalUn**: Saliency-thresholded parameter updates (`threshold = 0.5`) on random-labeled forget samples.
3. **NegGrad+**: Gradient ascent on $D_f$ combined with gradient descent on $D_r$ with gradient norm clipping (`grad-clip = 1.0`).
4. **SCRUB**: Teacher-student selective knowledge distillation (`sgda-bsz = 64`, `msteps = 2`).
5. **UNSIR**: 3 epochs of noise-based impair step on $D_f$ followed by repair step on retain subset.
6. **SVD (Training-Free)**: Subspace projection suppressing top singular components discriminative of $D_f$ ($\alpha_r, \alpha_f$).
7. **CMF Framework (+ CMF, Algorithms 1 & 2)**:
   - At the beginning of each epoch $e$:
     $$\mu_k = \frac{1}{|D_k|} \sum_{x \in D_k} \phi_\theta(x), \quad \bar{\mu} = \frac{1}{K}\sum_{k=1}^K \mu_k, \quad w_k = \frac{\mu_k - \bar{\mu}}{\|\mu_k - \bar{\mu}\|}$$
   - Construct $W_{\text{CMF}} = [w_1, \dots, w_K]^\top$.
   - **Freeze $W_{\text{CMF}}$** during backward pass; update encoder $\theta$ only using the respective unlearning loss $\mathcal{L}_U$.

---

## 5. Hyperparameter Reference Matrix

### Table 4: ResNet Experiments (ResNet-18 / ResNet-50)

| Dataset | Model | Method | Epochs | Batch | LR | Mom. | Key Flags |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **CIFAR-10** | ResNet18 | Original | 300 | 128 | 0.01 | 0.9 | cosine LR; WD=5e-4 |
| CIFAR-10 | ResNet18 | Retain-only Retrain | 200 | 128 | 0.01 | 0.9 | WD=5e-4; val-ratio=0.1 |
| CIFAR-10 | ResNet18 | Retain-only FT | 3 | 128 | 1e-3 | 0.9 | — |
| CIFAR-10 | ResNet18 | Random Label | 3 | 128 | 1e-4 | 0.9 | — |
| CIFAR-10 | ResNet18 | SalUN | 3 | 128 | 1e-4 | — | `threshold=0.5` |
| CIFAR-10 | ResNet18 | NegGrad+ | 3 | 128 | 1e-4 | — | `grad-clip=1.0` |
| CIFAR-10 | ResNet18 | SCRUB | 3 | 64 | 1e-4 | — | `sgda-bsz=64`, `msteps=2` |
| CIFAR-10 | ResNet18 | UNSIR | 3 | 128 | 5e-5 | — | 3 epochs impair/repair |
| CIFAR-10 | ResNet18 | SVD (TF) | — | 900 | — | — | $\alpha_r = 1000, \alpha_f = 30$ |
| CIFAR-10 | ResNet18 | Random Label + CMF | 4 | 128 | 2e-3 *(1e-4 1-class)* | — | — |
| CIFAR-10 | ResNet18 | SalUN + CMF | 4 | 128 | 2e-3 *(2e-4 1-class)* | — | `threshold=0.5` |
| CIFAR-10 | ResNet18 | NegGrad+ + CMF | 3 | 128 | 1e-4 | — | `grad-clip=1.0` |
| CIFAR-10 | ResNet18 | SCRUB + CMF | 3 | 64 | 5e-3 | — | `sgda-bsz=64`, `msteps=2` |
| CIFAR-10 | ResNet18 | UNSIR + CMF | 3 | 128 | 5e-5 | — | 3 epochs impair/repair |
| **CIFAR-100** | ResNet18 | Original | 300 | 128 | 0.01 | 0.9 | cosine LR; WD=5e-4 |
| CIFAR-100 | ResNet18 | Retain-only Retrain | 200 | 128 | 0.01 | 0.9 | WD=5e-4 |
| CIFAR-100 | ResNet18 | Retain-only FT | 3 | 128 | 1e-3 | 0.9 | — |
| CIFAR-100 | ResNet18 | Random Label | 3 | 128 | 3e-3 | — | — |
| CIFAR-100 | ResNet18 | SalUN | 3 | 128 | 1e-3 | — | `threshold=0.5` |
| CIFAR-100 | ResNet18 | NegGrad+ | 3 | 128 | 5e-3 | — | `grad-clip=1.0` |
| CIFAR-100 | ResNet18 | SCRUB | 3 | 64 | 1e-3 | — | `sgda-bsz=64`, `msteps=2` |
| CIFAR-100 | ResNet18 | UNSIR | 3 | 128 | 3e-5 | — | 3 epochs impair/repair |
| CIFAR-100 | ResNet18 | SVD (TF) | — | 990 | — | — | $\alpha_r = 1000, \alpha_f = 30$ |
| CIFAR-100 | ResNet18 | Random Label + CMF | 4 | 128 | 2e-3 | — | — |
| CIFAR-100 | ResNet18 | SalUN + CMF | 4 | 128 | 2e-3 | — | `threshold=0.5` |
| CIFAR-100 | ResNet18 | NegGrad+ + CMF | 3 | 128 | 1e-4 | — | `grad-clip=1.0` |
| CIFAR-100 | ResNet18 | SCRUB + CMF | 3 | 64 | 5e-3 | — | `sgda-bsz=64`, `msteps=2` |
| CIFAR-100 | ResNet18 | UNSIR + CMF | 3 | 128 | 5e-5 | — | 3 epochs impair/repair |
| **Tiny-ImageNet** | ResNet50 | Original | 300 | 128 | 0.05 | 0.9 | cosine LR; WD=5e-4 |
| Tiny-ImageNet | ResNet50 | Retain-only Retrain | 150 | 256 | 0.05 | 0.9 | WD=5e-4 |
| Tiny-ImageNet | ResNet50 | Retain-only FT | 3 | 128 | 1e-3 | 0.9 | — |
| Tiny-ImageNet | ResNet50 | Random Label | 3 | 128 | 5e-4 | — | — |
| Tiny-ImageNet | ResNet50 | SalUN | 3 | 128 | 5e-4 | — | `threshold=0.5` |
| Tiny-ImageNet | ResNet50 | NegGrad+ | 3 | 128 | 5e-4 | — | `grad-clip=1.0` |
| Tiny-ImageNet | ResNet50 | SCRUB | 3 | 64 | 5e-3 | — | `sgda-bsz=64`, `msteps=2` |
| Tiny-ImageNet | ResNet50 | UNSIR | 3 | 128 | 2e-5 | — | 3 epochs impair/repair |
| Tiny-ImageNet | ResNet50 | SVD (TF) | — | 999 | — | — | $\alpha_r = 30, \alpha_f = 10$ |
| Tiny-ImageNet | ResNet50 | Random Label + CMF | 4 | 128 | 1e-2 | — | — |
| Tiny-ImageNet | ResNet50 | SalUN + CMF | 4 | 128 | 1e-2 | — | `threshold=0.5` |
| Tiny-ImageNet | ResNet50 | NegGrad+ + CMF | 3 | 128 | 3e-5 | — | `grad-clip=1.0` |
| Tiny-ImageNet | ResNet50 | SCRUB + CMF | 3 | 64 | 1e-3 | — | `sgda-bsz=64`, `msteps=2` |
| Tiny-ImageNet | ResNet50 | UNSIR + CMF | 3 | 128 | 2e-5 | — | 3 epochs impair/repair |

---

### Table 5: ViT Experiments (ViT-S/16 Backbone)

| Dataset | Model | Method | Epochs | Batch | LR | Key Flags |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **CIFAR-10** | ViT-S/16 | Original | 10 | 128 | 3e-4 | Pretrained backbone |
| CIFAR-10 | ViT-S/16 | Retrain | 10 | 128 | 3e-4 | Pretrained backbone |
| CIFAR-10 | ViT-S/16 | Random Label | 3 | 128 | 3e-4 | — |
| CIFAR-10 | ViT-S/16 | SalUN | 3 | 128 | 3e-4 | `threshold=0.5` |
| CIFAR-10 | ViT-S/16 | NegGrad+ | 3 | 128 | 3e-4 | `grad-clip=1.0` |
| CIFAR-10 | ViT-S/16 | Random Label + CMF | 4 | 128 | 1e-3 | — |
| CIFAR-10 | ViT-S/16 | SalUN + CMF | 4 | 128 | 3e-3 / 2e-3 | `threshold=0.5` |
| CIFAR-10 | ViT-S/16 | NegGrad+ + CMF | 3 | 128 | 5e-5 / 5e-4 | `grad-clip=1.0` |
| **CIFAR-100** | ViT-S/16 | Original | 10 | 128 | 3e-4 | Pretrained backbone |
| CIFAR-100 | ViT-S/16 | Retrain | 10 | 128 | 3e-4 | Pretrained backbone |
| CIFAR-100 | ViT-S/16 | Random Label | 3 | 128 | 3e-4 | — |
| CIFAR-100 | ViT-S/16 | SalUN | 3 | 128 | 3e-4 | `threshold=0.5` |
| CIFAR-100 | ViT-S/16 | NegGrad+ | 3 | 128 | 3e-4 | `grad-clip=1.0` |
| CIFAR-100 | ViT-S/16 | Random Label + CMF | 4 | 128 | 2e-2 | — |
| CIFAR-100 | ViT-S/16 | SalUN + CMF | 4 | 128 | 2e-2 / 1e-2 | `threshold=0.5` |
| CIFAR-100 | ViT-S/16 | NegGrad+ + CMF | 3 | 128 | 5e-5 / 5e-4 | `grad-clip=1.0` |
| **Tiny-ImageNet** | ViT-S/16 | Original | 10 | 128 | 1e-4 | Pretrained backbone |
| Tiny-ImageNet | ViT-S/16 | Retrain | 10 | 128 | 1e-4 | Pretrained backbone |
| Tiny-ImageNet | ViT-S/16 | Random Label | 10 | 128 | 1e-4 | — |
| Tiny-ImageNet | ViT-S/16 | SalUN | 10 | 128 | 1e-4 | `threshold=0.5` |
| Tiny-ImageNet | ViT-S/16 | NegGrad+ | 10 | 128 | 1e-4 | `grad-clip=1.0` |
| Tiny-ImageNet | ViT-S/16 | Random Label + CMF | 4 | 128 | 2e-3 | — |
| Tiny-ImageNet | ViT-S/16 | SalUN + CMF | 4 | 128 | 1e-2 | `threshold=0.5` |
| Tiny-ImageNet | ViT-S/16 | NegGrad+ + CMF | 3 | 128 | 1e-5 / 1e-2 | `grad-clip=1.0` |

---

## 6. Implementation & Evaluation Checklist (Avoiding Common Pitfalls)

1. **Linear Probe Convergence**:
   - For $K=10$ (CIFAR-10): `probe_epochs = 50` is sufficient.
   - For $K=100$ (CIFAR-100): `probe_epochs` should scale to **200–300 epochs** (learning rate 0.01, SGD momentum 0.9) to guarantee full convergence across 100 classes.
2. **Raw Feature Extraction for Metric Evaluation**:
   - In NCC (Eq. 5) and Linear Probe evaluation (§3.2), extract **raw** penultimate features $\phi_\theta(x)$ directly from `model.extract_features(x)`.
   - Do **not** apply `_preprocess_feats_for_cmf()` (mean-centering and double normalization) during evaluation, as global mean subtraction over $K=100$ classes distorts the relative Euclidean distances of the class centroids.
3. **SCRUB Retain Distillation on Large Class Spaces**:
   - On CIFAR-100, ensure the teacher-student SGDA loop has sufficient retain steps / learning rate decay schedule to recover accuracy across 99 retain classes.
4. **CMF Head Update Frequency**:
   - Follow Algorithm 2: Recompute $W_{\text{CMF}}$ from training features at the start of **each epoch**, freeze $W_{\text{CMF}}$, and update only the encoder $\theta$.
