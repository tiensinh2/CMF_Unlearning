import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from unlearn.tools import apply_prep


def tarun_CMF_unlearn(args, model, device, retain_loader, forget_loader, train_loader, test_loader, train_dataset, val_index=None, **kwargs):
    from unlearn.tools import maybe_eval_and_save
    from utils import test

    import copy, torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, ConcatDataset
    from utils import test, get_model
    import evaluation
    from evaluation.nc_cmf import ncc_mismatch


    print("[Before Unlearning] Evaluating CMF model")
    print("model.args.CMF_momentum =", getattr(model.args, "CMF_momentum", None))

    # ========== 1) optimalifydevice ==========
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True
    )

    
    
    batch_size = args.batch_size
    impair_lr = args.tarun_impair_lr
    repair_lr = args.lr
    targets = np.array(train_dataset.targets)
    val_index_arr = np.array(val_index) if val_index is not None else np.arange(len(targets))
    forget_set = set(args.unlearn_class)
    # For whole-class removal: only retain-class samples. For stratified (forget_set==all
    # classes), val_index_arr already contains only retain samples — use all of them.
    retain_classes = [i for i in range(args.num_classes) if i not in forget_set]
    if retain_classes:
        index_list = []
        for i in retain_classes:
            class_i_index = np.intersect1d(np.where(i == targets)[0], val_index_arr)
            index_list.extend(class_i_index[:int(args.tarun_samples_per_class)])
    else:
        # Stratified split: every class has retain samples in val_index_arr
        index_list = []
        for i in range(args.num_classes):
            class_i_index = np.intersect1d(np.where(i == targets)[0], val_index_arr)
            index_list.extend(class_i_index[:int(args.tarun_samples_per_class)])
    small_retain_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, index_list), batch_size=batch_size, shuffle=True
    )
    small_forget_loader = torch.utils.data.DataLoader(
        forget_loader.dataset, batch_size=batch_size, shuffle=True
    )
    is_vit = "vit" in args.arch.lower() #  vit_tiny, vit_small, vit_b_16, ...
    is_resnet = "resnet" in args.arch.lower()
    noises = {}
    for cls_num in args.unlearn_class:
        if "tinyimagenet" in args.dataset.lower() or "tiny-imagenet" in args.dataset.lower():
            if is_vit:
                noises[cls_num] = Noise(batch_size, 3, 224, 224).to(device)
            elif is_resnet:
                noises[cls_num] = Noise(batch_size, 3, 64, 64).to(device)
            else:
                noises[cls_num] = Noise(batch_size, 3, 32, 32).to(device)
        elif "imagenet" in args.dataset:
            noises[cls_num] = Noise(batch_size, 3, 224, 224).to(device)
        else:
            noises[cls_num] = Noise(batch_size, 3, 32, 32).to(device)
        opt = torch.optim.Adam(noises[cls_num].parameters(), lr=0.1)
        for epoch in range(5):
            total_loss = []
            for _ in range(8):
                inputs = noises[cls_num]()
                labels = torch.zeros(batch_size, device=device).long() + cls_num
                outputs = model(inputs)
                loss = -F.cross_entropy(outputs, labels) + 0.1 * torch.mean(torch.sum(inputs.square(), dim=[1, 2, 3]))
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss.append(loss.cpu().item())
            print(f"Loss: {np.mean(total_loss)}")
    batch_size = 128
    num_batches = 20
    noisy_data = []
    for cls_num in args.unlearn_class:
        for _ in range(num_batches):
            batch = noises[cls_num]().cpu().detach()
            for i in range(batch.size(0)):
                noisy_data.append((batch[i], torch.tensor(cls_num)))
    other_samples = [(x.cpu(), torch.tensor(y)) for x, y in small_retain_loader.dataset]
    noisy_data += other_samples
    noisy_loader = torch.utils.data.DataLoader(noisy_data, batch_size=batch_size, shuffle=True)
    
    
    model.eval()
    model.recompute_cmf(train_loader, device=device)
    # Impair step
    print("-"*100); print("Impair step on Forget Model"); print("-"*100)
    optimizer = torch.optim.Adam(model.parameters(), lr=impair_lr)
    for epoch in range(args.epochs_or_steps):
        model.train()
        running_loss, running_acc = 0.0, 0
        with torch.no_grad():
            W_fixed = model.CMFweights.weight.detach()
            temp = getattr(model.args, "temperature", 1.0)
        for i, (inputs, labels) in enumerate(noisy_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # ---- CMF head (fixed W) ----
            

            f = model.extract_features(inputs)
            z = model._preprocess_feats_for_cmf(f)
            logits = (z @ W_fixed.t()) * temp

            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            out = logits.argmax(dim=1)
            running_acc += (labels == out).sum().item()
        print(f"Train loss {epoch+1}: {running_loss/len(noisy_data)}, Train Acc: {running_acc*100/len(noisy_data)}%")
        model.eval()
        model.recompute_cmf(train_loader, device=device)
        model.train()

    print("-"*100); print("Evaluating Forget Model after Impairment"); print("-"*100)
    
    retain_acc, forget_acc, metric = test(
            model, device, test_loader,
            args.unlearn_class, args.class_label_names, args.num_classes,
            job_name=args.unlearn_method, set_name=f"Test Set"
        )    
    # Repair step
    print("-"*100); print("Repair step on Forget Model"); print("-"*100)
    heal_loader = torch.utils.data.DataLoader(other_samples, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=repair_lr)
    for epoch in range(args.epochs_or_steps):
        model.train()
        running_loss, running_acc = 0.0, 0
        with torch.no_grad():
            W_fixed = model.CMFweights.weight.detach()
            temp = getattr(model.args, "temperature", 1.0)
        for inputs, labels in heal_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            

            f = model.extract_features(inputs)
            z = model._preprocess_feats_for_cmf(f)
            logits = (z @ W_fixed.t()) * temp
            loss = F.cross_entropy(logits, labels)
            loss.backward(); optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            out = logits.argmax(dim=1)
            running_acc += (labels == out).sum().item()
        print(f"Train loss {epoch+1}: {running_loss/len(other_samples)}, Train Acc: {running_acc*100/len(other_samples)}%")
        model.eval()
        model.recompute_cmf(train_loader, device=device)
        model.train()

    retain_acc, forget_acc, metric = test(
            model, device, test_loader,
            args.unlearn_class, args.class_label_names, args.num_classes,
            job_name=args.unlearn_method, set_name=f"Test Set"
        )
    model.history_log = {
         "retain_acc": [retain_acc],
        "forget_acc": [forget_acc],
    }
    return model




def tarun_unlearn(args, model, device, retain_loader, forget_loader, train_loader, test_loader, train_dataset, val_index=None, **kwargs):
    from unlearn.tools import maybe_eval_and_save
    from utils import test
    batch_size = args.batch_size
    impair_lr = args.tarun_impair_lr
    repair_lr = args.lr
    targets = np.array(train_dataset.targets)
    val_index_arr = np.array(val_index) if val_index is not None else np.arange(len(targets))
    forget_set = set(args.unlearn_class)
    retain_classes = [i for i in range(args.num_classes) if i not in forget_set]
    if retain_classes:
        # Whole-class removal: sample only from classes not being forgotten.
        index_list = []
        for i in retain_classes:
            class_i_index = np.intersect1d(np.where(i == targets)[0], val_index_arr)
            index_list.extend(class_i_index[:int(args.tarun_samples_per_class)])
    else:
        # Stratified split: forget_set spans all classes, so val_index_arr already
        # contains only the retained (non-forget) indices across all classes.
        # Sample up to tarun_samples_per_class per class from val_index_arr.
        index_list = []
        for i in range(args.num_classes):
            class_i_index = np.intersect1d(
                np.where(targets == i)[0], val_index_arr
            )
            index_list.extend(class_i_index[:int(args.tarun_samples_per_class)])
        # Fallback: if still empty (e.g. val_index not provided), use all of val_index_arr.
        if not index_list:
            index_list = list(val_index_arr[:int(args.tarun_samples_per_class * args.num_classes)])
    if not index_list:
        raise ValueError(
            f"tarun_unlearn: retain index_list is empty. "
            f"val_index length={len(val_index_arr)}, retain_classes={retain_classes}"
        )
    small_retain_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, index_list), batch_size=batch_size, shuffle=True
    )
    small_forget_loader = torch.utils.data.DataLoader(
        forget_loader.dataset, batch_size=batch_size, shuffle=True
    )
    is_vit = "vit" in args.arch.lower() #  vit_tiny, vit_small, vit_b_16, ...
    is_resnet = "resnet" in args.arch.lower()
    noises = {}
    for cls_num in args.unlearn_class:
        if "tinyimagenet" in args.dataset.lower() or "tiny-imagenet" in args.dataset.lower():
            if is_vit:
                noises[cls_num] = Noise(batch_size, 3, 224, 224).to(device)
            elif is_resnet:
                noises[cls_num] = Noise(batch_size, 3, 64, 64).to(device)
            else:
                noises[cls_num] = Noise(batch_size, 3, 32, 32).to(device)
        elif "imagenet" in args.dataset:
            noises[cls_num] = Noise(batch_size, 3, 224, 224).to(device)
        else:
            noises[cls_num] = Noise(batch_size, 3, 32, 32).to(device)
        opt = torch.optim.Adam(noises[cls_num].parameters(), lr=0.1)
        for epoch in range(5):
            total_loss = []
            for _ in range(8):
                inputs = noises[cls_num]()
                labels = torch.zeros(batch_size, device=device).long() + cls_num
                outputs = model(inputs)
                loss = -F.cross_entropy(outputs, labels) + 0.1 * torch.mean(torch.sum(inputs.square(), dim=[1, 2, 3]))
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss.append(loss.cpu().item())
            print(f"Loss: {np.mean(total_loss)}")
    batch_size = 128
    num_batches = 20
    noisy_data = []
    for cls_num in args.unlearn_class:
        for _ in range(num_batches):
            batch = noises[cls_num]().cpu().detach()
            for i in range(batch.size(0)):
                noisy_data.append((batch[i], torch.tensor(cls_num)))
    other_samples = [(x.cpu(), torch.tensor(y)) for x, y in small_retain_loader.dataset]
    noisy_data += other_samples
    noisy_loader = torch.utils.data.DataLoader(noisy_data, batch_size=batch_size, shuffle=True)
    
    # Impair step
    print("-"*100); print("Impair step on Forget Model"); print("-"*100)
    optimizer = torch.optim.Adam(model.parameters(), lr=impair_lr)
    for epoch in range(args.epochs_or_steps):
        model.train()
        running_loss, running_acc = 0.0, 0
        for i, (inputs, labels) in enumerate(noisy_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward(); optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            out = outputs.argmax(dim=1)
            running_acc += (labels == out).sum().item()
        print(f"Train loss {epoch+1}: {running_loss/len(noisy_data)}, Train Acc: {running_acc*100/len(noisy_data)}%")


    print("-"*100); print("Evaluating Forget Model after Impairment"); print("-"*100)
    
    retain_acc, forget_acc, metric = test(
            model, device, test_loader,
            args.unlearn_class, args.class_label_names, args.num_classes,
            job_name=args.unlearn_method, set_name=f"Test Set"
        )    
    # Repair step
    print("-"*100); print("Repair step on Forget Model"); print("-"*100)
    heal_loader = torch.utils.data.DataLoader(other_samples, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=repair_lr)
    for epoch in range(args.epochs_or_steps):
        model.train()
        running_loss, running_acc = 0.0, 0
        for inputs, labels in heal_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward(); optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            out = outputs.argmax(dim=1)
            running_acc += (labels == out).sum().item()
        print(f"Train loss {epoch+1}: {running_loss/len(other_samples)}, Train Acc: {running_acc*100/len(other_samples)}%")
    
    retain_acc, forget_acc, metric = test(
            model, device, test_loader,
            args.unlearn_class, args.class_label_names, args.num_classes,
            job_name=args.unlearn_method, set_name=f"Test Set"
        )
    model.history_log = {
         "retain_acc": [retain_acc],
        "forget_acc": [forget_acc],
    }
    return model


class Noise(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad=True)
    def forward(self):
        return self.noise