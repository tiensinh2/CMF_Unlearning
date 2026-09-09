import copy, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# -----------------------------------------------------------
# new：number，to/for layer3 do & MSE to/foralign
# -----------------------------------------------------------
def distill_layer3(model, device, retain_loader, n_epoch=150, lr=3e-4):
    model.eval()

    # 1. Hook get layer3_in / out
    feat_in, feat_out = [], []

    def hfunc(m, inp, out):
        feat_in.append(inp[0].cpu())
        feat_out.append(out.cpu())

    h = model.layer3.register_forward_hook(hfunc)
    with torch.no_grad():
        for xb, _ in tqdm(retain_loader, desc="Hook layer3"):
            _ = model(xb.to(device))
    h.remove()

    x_in  = torch.cat(feat_in).half()   # [N,128,4,4] or [N,128,4,4]
    y_out = torch.cat(feat_out).half()  # [N,256,2,2]

    dl = DataLoader(TensorDataset(x_in, y_out),
                    batch_size=512, shuffle=True,
                    num_workers=4, pin_memory=True)

    # 2.  layer3 and
    new_l3 = copy.deepcopy(model.layer3).to(device)
    for p in new_l3.parameters():
        nn.init.zeros_(p)

    opt = torch.optim.Adam(new_l3.parameters(), lr=lr, weight_decay=1e-4)
    mse = nn.MSELoss()

    # 3. training
    for ep in range(1, n_epoch+1):
        if ep % 20 == 0:
            with torch.no_grad():
                w = new_l3[0].conv1.weight        #  1  BasicBlock 's/of conv1
                print("mean|w| =", w.abs().mean().item())
        loss_sum = 0.
        for xb, yb in dl:
            xb = xb.to(device).float()
            yb = yb.to(device).float()

            pred = new_l3(xb)
            loss = mse(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * xb.size(0)
        print(f"[Distill E{ep:03d}]  MSE={loss_sum/len(x_in):.6f}")

    # 4. weightand BN
    model.layer3.load_state_dict(new_l3.state_dict())
    model.train(False)
    with torch.no_grad():
        for xb, _ in retain_loader:
            _ = model(xb.to(device))
    return model


# -----------------------------------------------------------
# after's/of random_label_unlearn (onlyadd 3 line/execute)
# -----------------------------------------------------------
def random_label_unlearn_layer3(args, model, device,
                         retain_loader, forget_loader,
                         train_loader, test_loader,
                         optimizer, epochs, test_forget_loader, **kwargs):
    from utils import test
    from unlearn.tools import freeze_except_last_layer, zero_except_last_layer
    import torch.optim as optim
    import copy, torch

    # ---------- Stage A: random-label unlearning ----------
    if getattr(args, "freeze_except_last", False):
        freeze_except_last_layer(model, args.freeze_except_last)
    elif getattr(args, "zero_last_layer", False):
        zero_except_last_layer(model, args.zero_last_layer)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=True)

    num_classes = args.num_classes
    forget_classes = set(args.unlearn_class)
    valid = [c for c in range(num_classes) if c not in forget_classes]
    all_classes = torch.arange(num_classes, device=device)
    choices = torch.tensor(valid, device=device) if valid else None

    full_forget = copy.deepcopy(forget_loader.dataset)
    gen_f = torch.Generator().manual_seed(args.seed)
    forget_sub, _ = torch.utils.data.random_split(
        full_forget, [args.num_forget_samples, len(full_forget) - args.num_forget_samples],
        generator=gen_f,
    )
    full_retain = copy.deepcopy(retain_loader.dataset)
    gen_r = torch.Generator().manual_seed(args.seed + 1)
    retain_sub, _ = torch.utils.data.random_split(
        full_retain, [args.num_retain_samples, len(full_retain) - args.num_retain_samples],
        generator=gen_r,
    )
    mixed_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([forget_sub, retain_sub]),
        batch_size=args.batch_size, shuffle=True,
    )

    model.train()
    for epoch in range(1, epochs + 1):
        for inputs, labels in mixed_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            forget_mask = torch.zeros_like(labels, dtype=torch.bool)
            for cls in forget_classes:
                forget_mask |= (labels == cls)
            idx_forget = forget_mask.nonzero(as_tuple=True)[0]
            if idx_forget.numel():
                if choices is not None:
                    rand = choices[torch.randint(0, len(valid), (idx_forget.numel(),), device=device)]
                else:
                    true_lbls = labels[idx_forget]
                    rand = torch.stack([
                        all_classes[all_classes != true_lbls[k]][
                            torch.randint(0, num_classes - 1, (1,)).item()
                        ]
                        for k in range(len(true_lbls))
                    ])
                labels[idx_forget] = rand
            optimizer.zero_grad()
            F.cross_entropy(model(inputs), labels).backward()
            optimizer.step()
            if getattr(args, "dry_run", False):
                break
        if getattr(args, "dry_run", False):
            break

    # ---------- Stage B: re-distill layer3 via MSE on retain activations ----------
    if getattr(args, "distill_layer3", True):
        if not hasattr(model, "layer3"):
            print("[random_label_layer3] Warning: model has no 'layer3' attribute — skipping Stage B.")
        else:
            n_epoch = getattr(args, "distill_layer3_epochs", 150)
            lr_distill = getattr(args, "distill_layer3_lr", 3e-4)
            print(f"\n=== Stage-B: Distill layer3 (MSE, {n_epoch} epochs) ===")
            model = distill_layer3(model, device, retain_loader,
                                   n_epoch=n_epoch, lr=lr_distill)

    return model
