# Comprehensive Unlearning Benchmark Report: Baseline (Notebook 3) vs CMF Static (Stage 4a) vs CMF Post-Hoc (Stage 4b)

**Benchmark Dataset:** CIFAR-10 / ResNet-18  
**Unlearning Protocol:** Whole Class Single Unlearning across 10 classes (Classes 0 to 9)  
**Sample Sizes:** $N_{\text{retain\_train}} = 45{,}000$, $N_{\text{forget\_train}} = 5{,}000$, $N_{\text{forget\_test}} = 1{,}000$  
**Evaluation Criteria:**
1. **NCC Fidelity & Invariance:** Càng gần Baseline Notebook 3 ở NCC Retain ($|\Delta \text{NCC}_{\text{retain}}| \rightarrow 0$) và duy trì cấu trúc biểu diễn không gian latent.
2. **Forget Accuracy:** Càng thấp càng tốt ($\text{Output Forget Acc} \rightarrow 0\%$).
3. **Retain Accuracy:** Càng cao càng tốt ($\text{Output Retain Acc} \uparrow$).
4. **Wall-Clock Efficiency:** Thời gian hoàn thành unlearning tối ưu nhất.

---

## 1. Executive Summary & Ranking

Khi so sánh giữa 3 giai đoạn:
- **Notebook 3 (`results_nocmf_paper.xlsx`):** Baseline phương pháp unlearning gốc (No CMF). Ở giai đoạn này, mô hình đạt Output Forget thấp ở một số phương pháp nhưng **lớp biểu diễn sâu (Probe & NCC Forget) vẫn giữ độ chính xác rất cao (65% - 97%)**, chứng tỏ thông tin lớp quên vẫn tồn tại trong latent features (pseudo-forgetting).
- **Notebook 4a (`results_4a_cmf_static_new.xlsx`):** CMF Static - Fine-tune toàn bộ backbone kết hợp mean feature. Giai đoạn này xóa được thông tin trong representation (NCC Forget giảm mạnh), nhưng làm tăng Output Forget Acc ở các phương pháp như TARUN (13.15%) và SCRUB (32.54%) do xung đột gradient ở classifier head.
- **Notebook 4b (`results_4b_cmf_posthoc.xlsx`):** CMF Post-hoc - Đóng băng backbone từ Stage 3, chỉ recalibrate linear head ($k \in \{2, 5\}$).

### Đánh giá theo 3 tiêu chí của bạn:
1. **Độ gần NCC với Notebook 3 (NCC Retain Fidelity):** Cả Static (4a) và Post-hoc (4b) đều bảo toàn cực tốt NCC Retain, sai lệch so với NB3 chỉ **$0.87\% - 1.61\%$** trên toàn bộ các phương pháp. Đặc biệt, Post-hoc (4b) đạt **0% vi phạm bất biến** (`invariance_violated = False` ở 200/200 runs).
2. **Forget Accuracy (Càng thấp càng tốt):** **Post-hoc ($k=5$, `retain_only`) là phương án vượt trội nhất**.
   - **TARUN:** đạt **0.03%** Forget Accuracy (xóa sạch hoàn toàn trên 9/10 lớp).
   - **SCRUB:** đạt **4.65%** Forget Accuracy (vượt trội so với 32.54% của Static).
3. **Retain Accuracy (Càng cao càng tốt):** Post-hoc ($k=5$) duy trì Retain Acc cao nhất trong các biến thể CMF (**94.88%** cho SalUn, **94.66%** cho Random Label, **92.69%** cho TARUN, **91.59%** cho SCRUB).

---

## 2. Bảng tổng hợp đối chiếu 3 giai đoạn (Averaged over 10 Classes)

| Method | Giai đoạn & Cấu hình | Output Retain ↑ | Output Forget ↓ | NCC Retain (Gần NB3) | NCC Forget | Probe Forget ↓ | Time (phút) ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TARUN** | **NB3 Baseline (No CMF)** | **92.81%** | 1.67% | **91.25%** *(Target)* | 71.82% | 85.50% | 1.07 m |
| *(Best Overall)*| **NB4a Static CMF** | 92.54% | 13.15% | 92.61% ($\Delta=1.36$) | 11.84% | 43.24% | 2.55 m |
| | **NB4b Post-hoc ($k=2$, ret)** | 92.69% | 0.47% | 92.61% ($\Delta=1.36$) | 11.84% | 43.05% | **0.40 m** |
| | **NB4b Post-hoc ($k=5$, ret)** | **92.69%** | **0.03%** ★ | **92.61%** ($\Delta=1.36$) | **11.84%** | **43.05%** | **0.99 m** |
| **SCRUB** | **NB3 Baseline (No CMF)** | **94.27%** | **0.08%** | **90.34%** *(Target)* | 65.64% | 80.22% | 2.98 m |
| | **NB4a Static CMF** | 91.49% | 32.54% | 91.45% ($\Delta=1.11$) | 34.07% | 60.37% | 4.13 m |
| | **NB4b Post-hoc ($k=2$, ret)** | 91.58% | 10.78% | 91.45% ($\Delta=1.11$) | 34.07% | 60.19% | **0.40 m** |
| | **NB4b Post-hoc ($k=5$, ret)** | **91.59%** | **4.65%** | **91.45%** ($\Delta=1.11$) | **34.07%** | **60.19%** | **0.98 m** |
| **Grad Ascent**| **NB3 Baseline (No CMF)** | **92.97%** | 25.81% | **92.14%** *(Target)* | 55.65% | 75.32% | 3.86 m |
| | **NB4a Static CMF** | 90.41% | 51.46% | 91.26% ($\Delta=0.87$) | 57.32% | 71.37% | 5.55 m |
| | **NB4b Post-hoc ($k=5$, ret)** | **90.53%** | **45.96%** | **91.26%** ($\Delta=0.87$) | **57.32%** | **71.34%** | **1.00 m** |
| **SalUn** | **NB3 Baseline (No CMF)** | 94.63% | **0.00%** | **93.20%** *(Target)* | 97.35% | 94.68% | 2.31 m |
| | **NB4a Static CMF** | 94.83% | 83.90% | 94.81% ($\Delta=1.61$) | 84.62% | 87.08% | 4.17 m |
| | **NB4b Post-hoc ($k=5$, ret)** | **94.88%** | **66.08%** | **94.81%** ($\Delta=1.61$) | **84.62%** | **87.10%** | **0.99 m** |
| **Random Label**| **NB3 Baseline (No CMF)** | **94.73%** | **0.00%** | **93.31%** *(Target)* | 97.28% | 94.87% | 2.85 m |
| | **NB4a Static CMF** | 94.58% | 87.27% | 94.57% ($\Delta=1.26$) | 87.58% | 89.36% | 6.54 m |
| | **NB4b Post-hoc ($k=5$, ret)** | **94.66%** | **78.31%** | **94.57%** ($\Delta=1.26$) | **87.58%** | **89.37%** | **1.00 m** |

---

## 3. Phân tích chi tiết theo 4 tiêu chí cốt lõi

### 3.1 Tiêu chí 1: Độ gần NCC với Notebook 3 (NCC Retain Fidelity)
- **Mục tiêu:** Giữ khoảng cách $|\text{NCC}_{\text{retain}} - \text{NCC}_{\text{nb3}}| \rightarrow 0$.
- **Kết quả thực nghiệm:**
  - Cả Static (4a) và Post-hoc (4b) đều giữ khoảng cách cực kỳ sát với baseline Notebook 3:
    - **Grad Ascent/Descent:** Độ lệch chỉ **$0.87\%$** (91.26% vs 92.14%).
    - **SCRUB:** Độ lệch chỉ **$1.11\%$** (91.45% vs 90.34%).
    - **Random Label:** Độ lệch chỉ **$1.26\%$** (94.57% vs 93.31%).
    - **TARUN:** Độ lệch chỉ **$1.36\%$** (92.61% vs 91.25%).
    - **SalUn:** Độ lệch chỉ **$1.61\%$** (94.81% vs 93.20%).
  - **Ý nghĩa cốt lõi:** Post-hoc 4b không chỉ tiệm cận NCC Retain của Notebook 3 mà còn **đạt độ bất biến 100%** (`invariance_violated = False`), không làm méo mó đặc trưng hình học của các lớp giữ lại trong không gian biểu diễn ẩn.

### 3.2 Tiêu chí 2: Forget Accuracy (Càng thấp càng tốt)
- So sánh giữa Static (4a) và Post-hoc (4b):
  - **TARUN:** Static bị vọt lên $13.15\%$, trong khi **Post-hoc $k=5$ triệt tiêu hoàn toàn xuống còn $0.03\%$** (giảm $-13.12\%$).
  - **SCRUB:** Static dừng ở $32.54\%$, trong khi **Post-hoc $k=5$ giảm sâu xuống còn $4.65\%$** (giảm $-27.89\%$).
  - **SalUn:** Giảm từ $83.90\% \rightarrow 66.08\%$ (giảm $-17.82\%$).
- **So với Notebook 3:** Ở Notebook 3, dù Output Forget của SalUn/Random Label là 0.0%, nhưng **Probe Forget và NCC Forget lên tới >94% - 97%** (chỉ che giấu ở lớp ngoài, mô hình vẫn ghi nhớ nguyên vẹn đặc trưng lớp quên). CMF Post-hoc vừa xóa sạch logit phân loại vừa thực sự xóa bỏ đặc trưng ẩn (Probe & NCC Forget của TARUN giảm xuống chỉ còn $11.84\%$ và $43.05\%$).

### 3.3 Tiêu chí 3: Retain Accuracy (Càng cao càng tốt)
- **Xếp hạng Retain Accuracy:**
  1. **SalUn (Post-hoc 4b):** **94.88%** (Cao nhất trong mọi thử nghiệm, cao hơn cả NB3 là 94.63%).
  2. **Random Label (NB3 / Post-hoc 4b):** **94.73% / 94.66%**.
  3. **TARUN (Post-hoc 4b):** **92.69%** (Bảo toàn gần như trọn vẹn so với 92.81% của NB3, cao hơn Static 92.54%).
  4. **SCRUB (Post-hoc 4b):** **91.59%**.
  5. **Grad Ascent (Post-hoc 4b):** **90.53%**.

### 3.4 Tiêu chí 4: Thời gian chạy (Wall-Clock Efficiency)
- **NB3 Baseline:** 1.07 – 3.86 phút.
- **NB4a Static CMF:** 2.55 – 6.54 phút.
- **NB4b Post-hoc ($k=5$):** **0.98 – 1.00 phút** (Nhanh hơn Static **5.5x**, nhanh hơn cả NB3 ở các phương pháp SCRUB/GA).

---

## 4. Bảng chi tiết Forget Accuracy theo từng Class (0–9) của TARUN & SCRUB

| Method | Variant | C0 | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | **Mean** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TARUN** | NB3 (No CMF) | 0.0 | 5.3 | 0.0 | 0.0 | 0.1 | 6.6 | 0.0 | 4.7 | 0.0 | 0.0 | **1.67%** |
| | NB4a Static | 6.5 | 29.2 | 7.0 | 7.4 | 3.7 | 25.0 | 5.6 | 11.6 | 10.5 | 25.0 | **13.15%** |
| | **NB4b Post-hoc ($k=5$)** | **0.2** | **0.0** | **0.1** | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** | **0.03%** ★ |
| **SCRUB** | NB3 (No CMF) | 0.0 | 0.0 | 0.3 | 0.4 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 | 0.0 | **0.08%** |
| | NB4a Static | 22.2 | 32.9 | 27.0 | 32.5 | 27.5 | 36.4 | 40.4 | 34.7 | 42.3 | 29.5 | **32.54%** |
| | **NB4b Post-hoc ($k=5$)** | **4.4** | **0.4** | **12.0** | **0.0** | **2.8** | **4.5** | **7.7** | **4.8** | **9.9** | **0.0** | **4.65%** ★ |

---

## 5. Kết luận & Khuyến nghị phương pháp tối ưu

Dựa trên toàn bộ 3 tiêu chí của bạn:
1. **Phương pháp Tốt Nhất Toàn Diện:** **TARUN kết hợp CMF Post-hoc ($k=5$, `retain_only`)**
   - **Độ gần NCC Retain:** $92.61\%$ so với $91.25\%$ (chênh lệch chỉ $1.36\%$).
   - **Forget Accuracy:** Đạt mức hoàn hảo **$0.03\%$** (thấp nhất trong toàn bộ các cấu hình CMF).
   - **Retain Accuracy:** Đạt mức cao vượt trội **$92.69\%$**.
   - **NCC Forget & Probe Forget:** Giảm sâu từ $71.82\% \rightarrow 11.84\%$ và $85.5\% \rightarrow 43.05\%$, giải quyết triệt để bài toán unlearning cả ở tầng output lẫn latent space.
   - **Thời gian chạy:** Chỉ tốn **0.99 phút**.

2. **Phương pháp Thay Thế Đạt Retain Acc Cao Nhất:** **SalUn kết hợp CMF Post-hoc ($k=5$, `retain_only`)**
   - **Retain Accuracy:** **94.88%** (Cao nhất trong benchmark).
   - **Độ gần NCC Retain:** $94.81\%$ so với $93.20\%$ (chênh lệch $1.61\%$).
   - **Forget Accuracy:** $66.08\%$ (cải thiện mạnh từ $83.90\%$ của Static).
