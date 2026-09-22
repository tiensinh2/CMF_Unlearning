# Từng phương pháp Unlearning: Cơ chế thuật toán và Lý do (Không) Tương thích với CMF

Tài liệu này đi sâu vào **từng phương pháp một**, mô tả chính xác cơ chế thuật toán (không phải chỉ tên gọi), rồi từ chính cơ chế đó suy ra lý do cụ thể vì sao CMF hoạt động tốt hoặc phá huỷ nó. Đây là bản mở rộng, thay thế cách tiếp cận "so sánh chéo bằng 1 chỉ số duy nhất" (illusion-gap) ở tài liệu trước — ở đây mỗi phương pháp được phân tích từ gốc thuật toán của chính nó.

---

## 1. grad_ascent_descent (NegGrad+)

### 1.1 Mô tả thuật toán

NegGrad+ là phương pháp đơn giản nhất về mặt ý tưởng: **đảo ngược mục tiêu tối ưu trên forget-set**. Thay vì tối thiểu hoá cross-entropy như lúc train ban đầu, nó tối đa hoá cross-entropy trên forget-sample, đồng thời vẫn tối thiểu hoá trên retain-sample để không phá huỷ toàn bộ model.

Theo đúng implementation đã trace: mỗi bước cập nhật gồm **2 lệnh backward riêng biệt cộng dồn gradient**:

```
W_fixed = CMFweights.weight.detach()      # classifier bị "đóng băng" khi tính loss

# (a) Ascent trên forget-batch
z_f = preprocess(extract_features(x_f))
logits_f = z_f @ W_fixed.T
loss_ascent = -0.5 * CrossEntropy(logits_f, y_f)     # DẤU ÂM — tối đa hoá loss
loss_ascent.backward()

# (b) Descent trên retain-batch (batch riêng, cùng vòng lặp)
z_r = preprocess(extract_features(x_r))
logits_r = z_r @ W_fixed.T
loss_descent = CrossEntropy(logits_r, y_r)
loss_descent.backward()

optimizer.step()    # gradient của (a) và (b) đã cộng dồn trong .grad
```

Không có cơ chế giới hạn nào cho `loss_ascent` — về mặt lý thuyết, cross-entropy âm không có giới hạn dưới (nó tiến tới −∞ khi logit của lớp đúng tiến tới −∞), nên **gradient ascent này có thể tiếp tục đẩy feature ra xa vô hạn định**, chỉ dừng lại vì số epoch hữu hạn (3 epoch), không phải vì bản thân mục tiêu tự bão hoà.

### 1.2 Vì sao KHÔNG tương thích với CMF

Cơ chế không tương thích ở đây không phải "over-erase runaway feedback loop" thuần tuý như SCRUB/TARUN — mà là **xung đột thứ tự thời gian giữa 2 quá trình có tốc độ khác nhau**:

- Bản thân ascent (không CMF) đã đủ mạnh để đẩy NCC_forget từ ~94% (model gốc) xuống 55.65% chỉ trong 3 epoch — **vượt qua điểm Oracle (67.30%) mà không cần CMF**. Đây là bằng chứng: **cross-entropy ascent không giới hạn (unbounded loss) có tốc độ hội tụ nhanh hơn tốc độ CMF cần để "bắt kịp" và tái cấu trúc W mỗi epoch.**

- Khi thêm CMF: W bị buộc phải đo lại class-mean **sau khi** ascent đã "vượt điểm" — tức là CMF đang cố gắng "diễn giải trung thực" một representation vốn dĩ **đã bị phá quá tay ngay từ epoch đầu**, không phải một representation đang trong quá trình "quên đúng mức". Vì CMF chỉ phản ánh chứ không sửa hướng, và bản thân representation đã sai hướng (quá xa) trước khi CMF kịp làm gì, nên CMF-Static không "khuếch đại thêm" được nhiều (55.65 → 57.33, gần như không đổi) — không phải vì cơ chế feedback yếu, mà vì **representation đã đạt tới điểm bão hoà tự nhiên của riêng ascent-loss trước khi vòng lặp CMF kịp phát huy tác dụng qua nhiều epoch.**

**Điểm mấu chốt cần nhớ:** đây là phương pháp DUY NHẤT trong 5 phương pháp mà **vấn đề tồn tại ngay cả khi không có CMF** (illusion_gap âm, −11.65) — CMF không phải nguyên nhân chính ở đây, chỉ là một yếu tố phụ không giúp gì thêm cho một vấn đề đã tồn tại sẵn từ chính thiết kế loss của NegGrad+ (thiếu cơ chế early-stopping hoặc chặn trên cho ascent-loss).

### 1.3 Vì sao Post-hoc lại hoạt động tốt ở đúng phương pháp này

Vì vấn đề gốc là **classifier bị buộc dùng công thức CMF cứng nhắc trong khi representation ĐÃ tốt** (chỉ hơi quá tay chút ở mức 55.65, không đến nỗi tệ), Post-hoc (đóng băng encoder, cho W học lại bằng gradient thật trên retain) cho phép classifier tìm lại đúng decision boundary phù hợp với representation hiện có — kéo forget-accuracy từ 55.65 lên 76.23 (gần Oracle hơn hẳn). Post-hoc ở đây **không sửa representation** (nó vẫn giữ nguyên representation đã hình thành) — nó chỉ tháo bỏ ràng buộc cứng nhắc của công thức CMF, để lộ ra rằng bản thân representation, khi được đọc bởi một classifier "khôn ngoan" hơn, đã khá gần Oracle.

---

## 2. random_label

### 2.1 Mô tả thuật toán

random_label thay label thật của forget-sample bằng **một label sai được chọn ngẫu nhiên** trong số 9 class còn lại, rồi train model bằng cross-entropy thông thường (không phải ascent) trên hỗn hợp: forget-sample (label giả) + retain-sample (label thật):

```
mixed_loader = ConcatDataset([forget_subset, retain_subset])   # cố định 1 lần

for epoch in 1..4:
  for (x, y_true) in mixed_loader:
    y_train = y_true.clone()
    forget_mask = (y_true thuộc forget_classes)
    y_train[forget_mask] = random_choice(non_forget_classes)   # NGẪU NHIÊN mỗi epoch
    
    W_fixed = CMFweights.weight.detach()
    z = preprocess(extract_features(x))
    logits = z @ W_fixed.T
    loss = CrossEntropy(logits, y_train)      # DESCENT thông thường, không phải ascent
    loss.backward(); optimizer.step()
```

**Điểm quan trọng nhất về mặt cơ chế**: đây **không phải gradient ascent** — nó là descent thông thường, chỉ khác là target label bị thay đổi. Về bản chất toán học, model đang được dạy: "hãy tự tin dự đoán forget-sample là một class SAI cụ thể nào đó" — chứ không phải "hãy trở nên không chắc chắn về forget-sample". Vì `rand_targets` được lấy mẫu **lại từ đầu mỗi epoch** (không cố định seed cho việc chọn target, chỉ cố định seed cho việc chọn subset), **class sai được gán cho cùng 1 forget-sample có thể khác nhau giữa các epoch**.

### 2.2 Vì sao chỉ tương thích MỘT PHẦN với CMF (cải thiện nhưng không đạt Oracle)

Cơ chế không tương thích ở đây khác hẳn NegGrad+ — nó nằm ở **tính không nhất quán của mục tiêu học qua các epoch**:

- **Epoch 1**: model học "forget-sample X trông giống class B" → feature của X bị kéo về phía cụm class B.
- **Epoch 2**: (nếu random assignment đổi khác) model học "forget-sample X trông giống class D" → feature của X bị kéo về phía cụm class D, **ngược hoặc lệch hướng so với epoch 1**.

Đây tạo ra một dạng "random walk" trong feature space thay vì một quỹ đạo dịch chuyển có hướng nhất quán. Khi CMF tính lại class-mean mỗi epoch dựa trên vị trí hiện tại (đã bị kéo qua lại ngẫu nhiên), **không có "hướng chủ đạo" nào để khuếch đại liên tục** — mỗi epoch, hiệu ứng của epoch trước một phần bị "xoá" bởi hướng kéo khác của epoch sau.

**Hệ quả**: CMF vẫn có tác dụng (vì nó buộc classifier phải phản ánh đúng representation, thay vì dựa vào label giả để "diễn" một cách rẻ tiền — đây chính là illusion mà No-CMF thể hiện: Output=0% dù NCC=97.28%) — nhưng vì representation chỉ bị "khuấy động" chứ không bị "đẩy nhất quán", CMF chỉ kéo được nó tới một trạng thái trung gian (87.62%) chứ không đạt được mức độ tách biệt thật sự (Oracle=67.30%).

### 2.3 Vì sao retain quality được bảo toàn tốt (94.55% cho CMF-Static)

Vì target label giả luôn nằm trong **các class KHÔNG phải forget** (`valid = [c for c in range(10) if c not in forget_classes]`), quá trình học này không hề tạo tín hiệu gradient nào cố tình phá retain-class khác — nó chỉ tác động qua lại lên chính forget-class và các class được chọn làm "label giả" ngẫu nhiên (trải đều qua 9 class, nên tác động lên mỗi class riêng lẻ là nhỏ và trung bình hoá).

---

## 3. SalUn

### 3.1 Mô tả thuật toán

SalUn có **hoàn toàn cùng cơ chế relabel ngẫu nhiên như random_label**, nhưng thêm một bước tiền xử lý: xây dựng một **saliency mask nhị phân** trước khi bắt đầu train, dùng để giới hạn tham số nào được phép nhận gradient update:

```
# BƯỚC 1 — Xây mask (chỉ chạy 1 lần, TRƯỚC khi train, trên model θ_0 CHƯA unlearn)
mask = {name: zeros_like(p) for p in model.parameters()}
for (x, y) in forget_loader:
    model.zero_grad()
    loss = CrossEntropy(model(x), y_TRUE)      # dùng NHÃN THẬT, không phải nhãn giả
    loss.backward()
    mask[name] += |gradient|                    # cộng dồn độ lớn gradient tuyệt đối

# Giữ top 50% tham số có |gradient| lớn nhất (tính theo toàn cục, không theo từng layer)
hard_mask[top 50%] = 1.0   # còn lại = 0.0

# BƯỚC 2 — Train giống hệt random_label, NHƯNG:
for epoch in 1..4:
  for (x, y_true) in mixed_loader:
    y_train = relabel_randomly(y_true)   # giống random_label
    loss = CrossEntropy(model(x), y_train)
    loss.backward()
    for p in model.parameters():
        p.grad *= hard_mask[p]           # CHẶN gradient ở 50% tham số "kém quan trọng"
    optimizer.step()
```

### 3.2 Vì sao còn kém tương thích với CMF hơn cả random_label

Cơ chế mask tạo ra thêm một lớp giới hạn: **chỉ 50% tham số của encoder được phép thay đổi**, và 50% đó được chọn dựa trên gradient của forget-loader **trên model θ_0 GỐC, TRƯỚC khi bất kỳ unlearning nào diễn ra**. Điều này có 2 hệ quả:

1. **Giảm thêm biên độ dịch chuyển representation**: ngay cả khi tín hiệu relabel giống hệt random_label, chỉ một nửa số tham số được phép phản hồi lại tín hiệu đó — biên độ dịch chuyển tổng thể nhỏ hơn.

2. **Mask "đông cứng" theo trạng thái ban đầu, không cập nhật theo tiến trình unlearn**: mask được tính **một lần duy nhất** trên θ_0, không bao giờ tính lại dù θ đã thay đổi qua các epoch. Nếu những tham số "quan trọng nhất với forget-class lúc ban đầu" không còn là những tham số quan trọng nhất sau vài epoch (vì representation đã dịch chuyển), cơ chế mask vẫn tiếp tục ưu tiên sai chỗ.

Kết hợp cả 2 yếu tố (tín hiệu không nhất quán hướng của random_label + giới hạn biên độ bởi mask cố định), SalUn's encoder có **ít không gian tự do hơn cả** để dịch chuyển, giải thích tại sao nó cho kết quả tương tự nhưng nhỉnh hơn kém một chút so với random_label ở một số biến thể CMF (84.46% CMF-Static so với 87.62% của random_label — thực ra thấp hơn một chút, nằm trong nhiễu thống kê giữa 2 phương pháp có cùng bản chất cốt lõi).

---

## 4. SCRUB

### 4.1 Mô tả thuật toán

SCRUB có cơ chế phức tạp và "có mục tiêu" nhất trong 5 phương pháp: dùng khung **teacher-student distillation**, với teacher là bản sao cố định của model gốc Θ_o (không bao giờ cập nhật), student là model đang được unlearn:

```
teacher = deepcopy(Θ_o); teacher.freeze()   # KHÔNG BAO GIỜ thay đổi trong suốt quá trình
student = deepcopy(Θ_o)                      # sẽ được cập nhật

for epoch in 1..3:
    if epoch <= msteps(=2):
        # MAX-STEP: tối đa hoá KL-divergence giữa student và teacher, TRÊN FORGET-SET
        for (x_f, y_f) in forget_loader:
            logits_teacher = teacher(x_f)        # CMF logits, W của teacher cố định
            logits_student = student(x_f)        # CMF logits, W của student (đang học)
            loss_max = -KL(softmax(logits_student/T), softmax(logits_teacher/T))
            loss_max.backward(); optimizer.step()   # đẩy student CÀNG XA teacher càng tốt

    # MIN-STEP: tối thiểu hoá KL-divergence + CE, TRÊN RETAIN-SET (luôn chạy mỗi epoch)
    for (x_r, y_r) in retain_loader:
        loss_min = KL(...) + alpha * CrossEntropy(logits_student, y_r)
        loss_min.backward(); optimizer.step()      # giữ student GẦN teacher trên retain

    if epoch >= sstart(=1):
        swa_model.update_parameters(student)        # EMA, chỉ để theo dõi, không dùng để eval

    student.recompute_cmf(train_loader)              # CMF reconstruction cuối mỗi epoch
```

Điểm khác biệt cốt lõi so với NegGrad+ và random_label: **mục tiêu ascent ở đây có một "đích" rõ ràng và cố định** (rời xa hành vi của TEACHER — một model cụ thể, bất biến), không phải rời xa nhãn thật (như NegGrad+) hay tiến gần một nhãn giả ngẫu nhiên (như random_label). KL-divergence cũng là một đại lượng **có xu hướng bão hoà tự nhiên** khi 2 phân phối đã đủ khác biệt (không giống cross-entropy âm của NegGrad+, vốn không có giới hạn).

### 4.2 Vì sao đây là phương pháp KHÔNG TƯƠNG THÍCH NGHIÊM TRỌNG NHẤT với cơ chế lặp của CMF (dù bản thân SCRUB rất tốt)

Đây chính xác là nghịch lý trung tâm: **SCRUB một mình đã gần như hoàn hảo** (NCC=65.64%, chỉ lệch Oracle 1.66 điểm) — vì mục tiêu KL-divergence-tới-teacher-cố-định là một tín hiệu ascent **có chất lượng cao nhất** trong 5 phương pháp (có đích rõ ràng, ổn định, không đổi hướng như random_label, không unbounded như NegGrad+). Nhưng chính "chất lượng cao" này lại là điều khiến nó **dễ bị vòng lặp CMF khuếch đại nhất**:

- Mỗi epoch, max-step đẩy student ra xa teacher theo đúng 1 hướng nhất quán (luôn là "rời xa teacher", không đổi mục tiêu như random_label).
- CMF tính lại class-mean ngay sau đó, dựa trên feature đã bị đẩy — **class-mean mới này lại trở thành một phần của "hành vi hiện tại" mà epoch tiếp theo tiếp tục coi là điểm neo để tính KL-divergence qua CMF logit**.
- Vì SCRUB chạy đích xác 2 epoch max-step liên tiếp (không phải 1 epoch rồi dừng), vòng lặp có đủ số lần lặp để cộng dồn hiệu ứng.

**Đây là bằng chứng mạnh nhất cho giả thuyết vòng lặp phản hồi dương ở toàn bộ nghiên cứu**: chính phương pháp có tín hiệu ascent "tốt nhất" (rõ ràng, nhất quán, có đích) lại là phương pháp bị tổn hại nặng nhất khi kết hợp với CMF (65.64 → 34.40, giảm 31.24 điểm) — vì "tốt" theo nghĩa ascent hiệu quả lại đồng nghĩa với "nguy hiểm" khi kết hợp với một cơ chế khuếch đại không biết điểm dừng.

### 4.3 Vì sao Post-hoc/Budget-Shared không cứu được SCRUB (khác với NegGrad+)

Khác với NegGrad+ (nơi representation vẫn tốt, chỉ classifier bị gò cứng), ở SCRUB **chính representation đã bị phá huỷ thật** trong Stage 1 (do vòng lặp khuếch đại làm feature dịch chuyển thật sự quá xa, không chỉ là vấn đề classifier). Post-hoc chỉ có thể tối ưu classifier trên representation **đã hỏng sẵn** — nó không có cách nào "lấy lại" thông tin representation đã mất. Đây là lý do Post-hoc cho SCRUB vẫn cho kết quả tệ (25.31%, thậm chí tệ hơn cả CMF-Static) — không phải vì post-hoc "làm gì sai", mà vì nó đang cố sửa chữa ở đúng khâu (classifier) trong khi vấn đề thật nằm ở khâu khác (representation) đã không thể đảo ngược.

---

## 5. TARUN (UNSIR)

### 5.1 Mô tả thuật toán

TARUN có cơ chế cực đoan nhất: thay vì dùng dữ liệu thật để tạo tín hiệu quên, nó **tạo ra dữ liệu nhiễu tổng hợp được tối ưu hoá riêng** để đạt hiệu ứng phá huỷ tối đa, qua 2 pha:

**Pha chuẩn bị — tạo nhiễu (Noise generation):**
```
for mỗi forget_class:
    noise = Noise(batch_size, 3, 32, 32)     # tensor ngẫu nhiên, CÓ THỂ HỌC (nn.Parameter)
    optimizer_noise = Adam(noise.parameters(), lr=0.1)
    
    for 5 epochs × 8 steps = 40 bước:
        inputs = noise()                       # sinh ảnh nhiễu hiện tại
        labels = [forget_class] * batch_size
        outputs = model(inputs)                # forward qua CHÍNH model đang unlearn
        loss = -CrossEntropy(outputs, labels) + 0.1 * ||inputs||²     
        # (a) tối đa hoá confidence model dành cho forget_class trên ảnh nhiễu
        # (b) phạt nhẹ để nhiễu không "nổ" quá lớn
        loss.backward(); optimizer_noise.step()   # CẬP NHẬT NHIỄU, không cập nhật model
```

Đây là một dạng **tấn công đối kháng ngược** (adversarial-style optimization): thay vì tìm input đánh lừa model, nó tìm input khiến model "tự tin sai" theo đúng ý muốn — một ảnh nhiễu không có cấu trúc ngữ nghĩa thật, nhưng được tối ưu để kích hoạt mạnh nhất các "đường dẫn" trong mạng liên quan tới forget-class.

**Pha Impair (làm hỏng có chủ đích):**
```
noisy_loader = mix(noise_images_optimized, retain_samples)
for epoch in 1..3:
    for (x, y) in noisy_loader:
        W_fixed = CMFweights.weight.detach()
        logits = preprocess(extract_features(x)) @ W_fixed.T
        loss = CrossEntropy(logits, y)     # y = forget_class cho ảnh nhiễu, y thật cho retain
        loss.backward(); optimizer.step()
    model.recompute_cmf(train_loader)       # CMF reconstruct SAU MỖI EPOCH IMPAIR
```

**Pha Repair (khôi phục retain):**
```
heal_loader = retain_samples_only   # KHÔNG còn ảnh nhiễu
for epoch in 1..3:
    # y hệt vòng lặp trên nhưng chỉ với retain-data
    ...
    model.recompute_cmf(train_loader)
```

Tổng cộng: 3 epoch impair + 3 epoch repair = **6 epoch cập nhật encoder**, gấp đôi số epoch "danh nghĩa" (`epochs_or_steps=3`) mà cấu hình ghi nhận.

### 5.2 Vì sao đây là phương pháp KHÔNG TƯƠNG THÍCH NGHIÊM TRỌNG NHẤT trong toàn bộ benchmark

TARUN hội tụ đủ **cả 3 yếu tố gây bất tương thích đã xác định** ở các phương pháp trước, cộng dồn vào một phương pháp duy nhất:

1. **Tín hiệu ascent cực mạnh và có chủ đích tuyệt đối** (giống SCRUB, nhưng còn mạnh hơn): ảnh nhiễu được thiết kế RIÊNG để tối đa hoá phản ứng của model đối với forget-class — đây không phải tín hiệu "tự nhiên" từ dữ liệu thật, mà là một **input được thiết kế đối kháng, tối ưu hoá chuyên biệt** để gây nhiễu loạn representation mạnh nhất có thể trong phạm vi ràng buộc năng lượng (`0.1 * ||inputs||²`).

2. **Chạy đủ nhiều epoch liên tiếp theo cùng 1 hướng** (giống SCRUB, nhưng gấp đôi thời lượng): 3 epoch impair liên tục cùng một mục tiêu (đẩy model về phía nhận diện forget-class từ nhiễu), đủ để vòng lặp phản hồi dương của CMF cộng dồn nhiều lần.

3. **Dữ liệu đầu vào (nhiễu) không mang thông tin thật của forget-class**: đây là điểm khác biệt quan trọng nhất so với SCRUB. Khi CMF tính lại class-mean của forget-class dựa trên feature của **ảnh nhiễu** (không phải ảnh thật), class-mean mới này **không còn đại diện cho bất kỳ khái niệm hình ảnh thật nào** — nó trở thành class-mean của "thứ khiến model bối rối nhất", một điểm hoàn toàn nhân tạo trong feature space. Vòng lặp CMF sau đó tiếp tục neo vào điểm nhân tạo này, khuếch đại sự trôi dạt càng lúc càng xa khỏi bất kỳ điều gì có ý nghĩa thống kê thật.

Đây giải thích tại sao TARUN có mức sụp đổ **tệ nhất trong toàn bộ benchmark** (71.82 → 13.05, giảm 58.77 điểm) — nó không chỉ bị vòng lặp phản hồi dương như SCRUB, mà còn bị khuếch đại trên một nền tảng dữ liệu (nhiễu tổng hợp) vốn dĩ đã không ổn định về mặt thống kê ngay từ đầu.

### 5.3 Vì sao Pha Repair không "cứu" được vấn đề

Dù có pha Repair chạy sau đó (chỉ dùng retain-data thật), pha này **cũng chạy dưới đúng cơ chế CMF-per-epoch-reconstruction** — nghĩa là nó cố "sửa" bằng cách tiếp tục vòng lặp đo-rồi-cập nhật, trên một class-mean của forget-class **đã bị lệch hoàn toàn khỏi thực tế** từ pha Impair. Retain-accuracy được bảo toàn tốt (92.42-92.64% — thậm chí cao hơn No-CMF một chút) vì Repair đúng là tối ưu cho retain — nhưng phần forget-class's class-mean, một khi đã trôi dạt xa khỏi phân phối thật trong pha Impair, không có cơ chế nào trong Repair để "kéo nó về" đúng vị trí hợp lý.

---

## 6. Bảng tổng hợp: đặc điểm thuật toán quyết định khả năng tương thích

| Method | Nguồn tín hiệu ascent | Có "đích" cố định? | Tính nhất quán hướng qua epoch | Dữ liệu có thật không? | Tương thích CMF |
|---|---|---|---|---|---|
| grad_ascent_descent | −CE trên nhãn thật | Không (unbounded) | Có, nhưng bão hoà quá nhanh | Thật | Trung tính (vấn đề tồn tại trước cả khi có CMF) |
| random_label | CE trên nhãn giả ngẫu nhiên | Không (đổi mỗi epoch) | **Không** | Thật | Tốt một phần (an toàn nhưng không đủ mạnh) |
| SalUn | Giống random_label + saliency mask cố định | Không | Không, và bị giới hạn thêm bởi mask | Thật | Tốt một phần (yếu hơn random_label do mask) |
| SCRUB | KL-divergence tới teacher cố định | **Có** (rời xa teacher) | **Có** (2 epoch liên tục cùng hướng) | Thật | **Rất kém** (vòng lặp khuếch đại mạnh) |
| TARUN | CE trên ảnh nhiễu tối ưu hoá | **Có** (tối đa hoá confidence forget-class) | **Có** (3+3 epoch cùng hướng) | **Không** (nhiễu tổng hợp) | **Kém nhất** (khuếch đại + nền dữ liệu không ổn định) |

**Quy luật rút ra**: mức độ tương thích với CMF tỉ lệ nghịch với "chất lượng" của tín hiệu ascent theo đúng nghĩa hẹp (có đích rõ ràng + nhất quán + chạy đủ lâu) — **những gì làm cho một phương pháp unlearning "tốt" khi đứng một mình lại chính là những gì khiến nó dễ bị CMF phá huỷ khi kết hợp**, vì CMF khuếch đại bất kỳ tín hiệu nhất quán nào mà không biết giới hạn dừng.
