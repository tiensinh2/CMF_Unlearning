# ==============================================================================
# EXPERIMENT CONFIGURATION & BENCHMARK SETTINGS
# Paper: "An Illusion of Unlearning? Assessing Machine Unlearning Through Internal Representations"
# Authors: Yichen Gao, Altay Unal, Akshay Rangamani, Zhihui Zhu
# ==============================================================================

# ==============================================================================
# 1. DATASET & UNLEARNING CLASS SETTINGS
# ==============================================================================

DATASET_CONFIGS = {
    "CIFAR-10": {
        "num_classes": 10,
        "image_size": 32,
        "single_class_forgetting": [
            [0], [1], [2], [3], [4], [5], [6], [7], [8], [9]
        ],
        "multi_class_forgetting_3_class": [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 5, 9],
            [2, 4, 8],
        ],
    },
    "CIFAR-100": {
        "num_classes": 100,
        "coarse_classes": 20,
        "fine_classes_per_coarse": 5,
        "image_size": 32,
        # Evaluates representative classes covering different superclasses
        # (e.g., classes 1 and 4 belong to the same coarse class, so only 1 is used)
        "single_class_forgetting": [
            [0], [1], [2], [3], [5]
        ],
        # 10 classes at a time (union of 2 coarse classes = 2 x 5 fine classes)
        "multi_class_forgetting_10_class": [
            [3, 15, 19, 21, 31, 38, 42, 43, 88, 97],
            [47, 52, 54, 56, 59, 62, 70, 82, 92, 96],
            [5, 20, 22, 25, 39, 40, 84, 86, 87, 94],
            [8, 13, 41, 48, 59, 69, 81, 85, 89, 90],
            [1, 4, 30, 32, 55, 67, 72, 73, 91, 95],
        ],
    },
    "Tiny-ImageNet": {
        "num_classes": 200,
        "image_size": 64,
    }
}

# ==============================================================================
# 2. MODEL ARCHITECTURE & ORIGINAL TRAINING / RETRAINING CONFIGURATIONS
# ==============================================================================

ORIGINAL_MODEL_TRAINING = {
    "ResNet": {
        "CIFAR-10": {
            "model": "ResNet-18",
            "train_from_scratch": True,
            "epochs": 300,
            "early_stopping_patience": 50,
            "batch_size": 128,
            "lr": 0.01,             # Table 4 specification (A.4 text states initial LR 0.05)
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "optimizer": "SGD",
            "lr_schedule": "cosine_with_warmup",
            "warmup_epochs": 5,
            "min_lr": 1e-5,
            "data_augmentation": ["RandomCrop(32, padding=4)", "RandomHorizontalFlip()"],
        },
        "CIFAR-100": {
            "model": "ResNet-18",
            "train_from_scratch": True,
            "epochs": 300,
            "early_stopping_patience": 50,
            "batch_size": 128,
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "optimizer": "SGD",
            "lr_schedule": "cosine_with_warmup",
            "warmup_epochs": 5,
            "min_lr": 1e-5,
            "data_augmentation": ["RandomCrop(32, padding=4)", "RandomHorizontalFlip()"],
        },
        "Tiny-ImageNet": {
            "model": "ResNet-50",
            "train_from_scratch": True,
            "epochs": 300,
            "early_stopping_patience": 50,
            "batch_size": 128,
            "lr": 0.05,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "optimizer": "SGD",
            "lr_schedule": "cosine_with_warmup",
            "warmup_epochs": 5,
            "min_lr": 1e-5,
            "data_augmentation": ["RandomCrop(64, padding=8)", "RandomHorizontalFlip()"],
        },
    },
    "ViT": {
        "common": {
            "model": "ViT-S/16",
            "initialization": "ImageNet-pretrained weights",
            "input_resolution": (224, 224),
            "epochs": 10,
            "batch_size": 128,
            "momentum": 0.9,
            "optimizer": "SGD",
            "data_augmentation": ["RandomCrop", "RandomHorizontalFlip", "Resize(224, 224)"],
        },
        "CIFAR-10": {
            "lr": 3e-4,
        },
        "CIFAR-100": {
            "lr": 3e-4,
        },
        "Tiny-ImageNet": {
            "lr": 1e-4,
        },
    }
}

# Retrain-on-Retain Baseline (Gold Standard)
RETAIN_ONLY_RETRAIN = {
    "ResNet": {
        "CIFAR-10": {
            "model": "ResNet-18",
            "epochs": 200,
            "batch_size": 128,
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "val_ratio": 0.1,
        },
        "CIFAR-100": {
            "model": "ResNet-18",
            "epochs": 200,
            "batch_size": 128,
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 5e-4,
        },
        "Tiny-ImageNet": {
            "model": "ResNet-50",
            "epochs": 150,
            "batch_size": 256,
            "lr": 0.05,
            "momentum": 0.9,
            "weight_decay": 5e-4,
        },
    },
    "ViT": {
        "CIFAR-10": {
            "model": "ViT-S/16",
            "epochs": 10,
            "batch_size": 128,
            "lr": 3e-4,
            "pretrained_backbone": True,
        },
        "CIFAR-100": {
            "model": "ViT-S/16",
            "epochs": 10,
            "batch_size": 128,
            "lr": 3e-4,
            "pretrained_backbone": True,
        },
        "Tiny-ImageNet": {
            "model": "ViT-S/16",
            "epochs": 10,
            "batch_size": 128,
            "lr": 1e-4,
            "pretrained_backbone": True,
        },
    }
}

# ==============================================================================
# 3. UNLEARNING METHODS HYPERPARAMETERS (TABLE 4: RESNET EXPERIMENTS)
# ==============================================================================

TABLE_4_RESNET_HYPERPARAMS = {
    "CIFAR-10": {
        "Original": {
            "model": "ResNet18", "epochs": 300, "batch_size": 128, "lr": 1e-2, "momentum": 0.9,
            "notes": "cosine LR; WD=5e-4"
        },
        "Retain-only Retrain": {
            "model": "ResNet18", "epochs": 200, "batch_size": 128, "lr": 1e-2, "momentum": 0.9,
            "notes": "WD=5e-4; val-ratio=0.1"
        },
        "Retain-only FT": {
            "model": "ResNet18", "epochs": 3, "batch_size": 128, "lr": 1e-3, "momentum": 0.9,
        },
        "Random Label": {
            "model": "ResNet18", "epochs": 3, "batch_size": 128, "lr": 1e-4, "momentum": 0.9,
        },
        "SalUN": {
            "model": "ResNet18", "epochs": 3, "batch_size": 128, "lr": 1e-4, "momentum": None,
            "threshold": 0.5,
        },
        "NegGrad+": {
            "model": "ResNet18", "epochs": 3, "batch_size": 128, "lr": 1e-4, "momentum": None,
            "grad_clip": 1.0,
        },
        "SCRUB": {
            "model": "ResNet18", "epochs": 3, "batch_size": 64, "lr": 1e-4, "momentum": None,
            "sgda_bsz": 64, "msteps": 2,
        },
        "UNSIR": {
            "model": "ResNet18", "epochs": 3, "batch_size": 128, "lr": 5e-5, "momentum": None,
            "notes": "3 epochs impair/repair training",
        },
        "SVD (TF)": {
            "model": "ResNet18", "epochs": None, "batch_size": 900, "lr": None, "momentum": None,
            "alpha_r": 1000, "alpha_f": 30, "training_free": True,
        },
        "Random Label + CMF": {
            "model": "ResNet18", "epochs": 4, "batch_size": 128, "lr": 2e-3, "momentum": None,
        },
        "SalUN + CMF": {
            "model": "ResNet18", "epochs": 4, "batch_size": 128, "lr": 2e-3, "momentum": None,
            "threshold": 0.5,
        },
        "NegGrad+ + CMF": {
            "model": "ResNet18", "epochs": 3, "batch_size": 128, "lr": 1e-4, "momentum": None,
            "grad_clip": 1.0,
        },
        "SCRUB + CMF": {
            "model": "ResNet18", "epochs": 3, "batch_size": 64, "lr": 5e-3, "momentum": None,
            "sgda_bsz": 64, "msteps": 2,
        },
        "UNSIR + CMF": {
            "model": "ResNet18", "epochs": 3, "batch_size": 128, "lr": 5e-5, "momentum": None,
            "notes": "3 epochs impair/repair training",
        },
    },
    "CIFAR-100": {
        "Original": {
            "model": "ResNet18", "epochs": 300, "batch_size": 128, "lr": 1e-2, "momentum": 0.9,
            "notes": "cosine LR; WD=5e-4"
        },
        "Retain-only Retrain": {
            "model": "ResNet18", "epochs": 200, "batch_size": 128, "lr": 1e-2, "momentum": 0.9,
            "notes": "WD=5e-4"
        },
        "Retain-only FT": {
            "model": "ResNet18", "epochs": 3, "batch_size": 128, "lr": 1e-3, "momentum": 0.9,
        },
        "Random Label": {
            "model": "ResNet18", "epochs": 3, "batch_size": 128, "lr": 3e-3, "momentum": None,
        },
        "SalUN": {
            "model": "ResNet18", "epochs": 3, "batch_size": 128, "lr": 1e-3, "momentum": None,
            "threshold": 0.5,
        },
        "NegGrad+": {
            "model": "ResNet18", "epochs": 3, "batch_size": 128, "lr": 5e-3, "momentum": None,
            "grad_clip": 1.0,
        },
        "SCRUB": {
            "model": "ResNet18", "epochs": 3, "batch_size": 64, "lr": 1e-3, "momentum": None,
            "sgda_bsz": 64, "msteps": 2,
        },
        "UNSIR": {
            "model": "ResNet18", "epochs": 3, "batch_size": 128, "lr": 3e-5, "momentum": None,
            "notes": "3 epochs impair/repair training",
        },
        "SVD (TF)": {
            "model": "ResNet18", "epochs": None, "batch_size": 990, "lr": None, "momentum": None,
            "alpha_r": 1000, "alpha_f": 30, "training_free": True,
        },
        "Random Label + CMF": {
            "model": "ResNet18", "epochs": 4, "batch_size": 128, "lr": 2e-3, "momentum": None,
        },
        "SalUN + CMF": {
            "model": "ResNet18", "epochs": 4, "batch_size": 128, "lr": 2e-3, "momentum": None,
            "threshold": 0.5,
        },
        "NegGrad+ + CMF": {
            "model": "ResNet18", "epochs": 3, "batch_size": 128, "lr": 1e-4, "momentum": None,
            "grad_clip": 1.0,
        },
        "SCRUB + CMF": {
            "model": "ResNet18", "epochs": 3, "batch_size": 64, "lr": 5e-3, "momentum": None,
            "sgda_bsz": 64, "msteps": 2,
        },
        "UNSIR + CMF": {
            "model": "ResNet18", "epochs": 3, "batch_size": 128, "lr": 5e-5, "momentum": None,
            "notes": "3 epochs impair/repair training",
        },
    },
    "Tiny-ImageNet": {
        "Original": {
            "model": "ResNet50", "epochs": 300, "batch_size": 128, "lr": 5e-2, "momentum": 0.9,
            "notes": "cosine LR; WD=5e-4"
        },
        "Retain-only Retrain": {
            "model": "ResNet50", "epochs": 150, "batch_size": 256, "lr": 5e-2, "momentum": 0.9,
            "notes": "WD=5e-4"
        },
        "Retain-only FT": {
            "model": "ResNet50", "epochs": 3, "batch_size": 128, "lr": 1e-3, "momentum": 0.9,
        },
        "Random Label": {
            "model": "ResNet50", "epochs": 3, "batch_size": 128, "lr": 5e-4, "momentum": None,
        },
        "SalUN": {
            "model": "ResNet50", "epochs": 3, "batch_size": 128, "lr": 5e-4, "momentum": None,
            "threshold": 0.5,
        },
        "NegGrad+": {
            "model": "ResNet50", "epochs": 3, "batch_size": 128, "lr": 5e-4, "momentum": None,
            "grad_clip": 1.0,
        },
        "SCRUB": {
            "model": "ResNet50", "epochs": 3, "batch_size": 64, "lr": 5e-3, "momentum": None,
            "sgda_bsz": 64, "msteps": 2,
        },
        "UNSIR": {
            "model": "ResNet50", "epochs": 3, "batch_size": 128, "lr": 2e-5, "momentum": None,
            "notes": "3 epochs impair/repair training",
        },
        "SVD (TF)": {
            "model": "ResNet50", "epochs": None, "batch_size": 999, "lr": None, "momentum": None,
            "alpha_r": 30, "alpha_f": 10, "training_free": True,
        },
        "Random Label + CMF": {
            "model": "ResNet50", "epochs": 4, "batch_size": 128, "lr": 1e-2, "momentum": None,
        },
        "SalUN + CMF": {
            "model": "ResNet50", "epochs": 4, "batch_size": 128, "lr": 1e-2, "momentum": None,
            "threshold": 0.5,
        },
        "NegGrad+ + CMF": {
            "model": "ResNet50", "epochs": 3, "batch_size": 128, "lr": 3e-5, "momentum": None,
            "grad_clip": 1.0,
        },
        "SCRUB + CMF": {
            "model": "ResNet50", "epochs": 3, "batch_size": 64, "lr": 1e-3, "momentum": None,
            "sgda_bsz": 64, "msteps": 2,
        },
        "UNSIR + CMF": {
            "model": "ResNet50", "epochs": 3, "batch_size": 128, "lr": 2e-5, "momentum": None,
            "notes": "3 epochs impair/repair training",
        },
    }
}

# ==============================================================================
# 4. UNLEARNING METHODS HYPERPARAMETERS (TABLE 5: VIT EXPERIMENTS)
# ==============================================================================

TABLE_5_VIT_HYPERPARAMS = {
    "CIFAR-10": {
        "Original": {"model": "ViT-S/16", "epochs": 10, "batch_size": 128, "lr": 3e-4, "notes": "pretrained backbone"},
        "Retrain": {"model": "ViT-S/16", "epochs": 10, "batch_size": 128, "lr": 3e-4, "notes": "pretrained backbone"},
        "Random Label": {"model": "ViT-S/16", "epochs": 3, "batch_size": 128, "lr": 3e-4},
        "SalUN": {"model": "ViT-S/16", "epochs": 3, "batch_size": 128, "lr": 3e-4, "threshold": 0.5},
        "NegGrad+": {"model": "ViT-S/16", "epochs": 3, "batch_size": 128, "lr": 3e-4, "grad_clip": 1.0},
        "Random Label + CMF": {"model": "ViT-S/16", "epochs": 4, "batch_size": 128, "lr": 1e-3},
        "SalUN + CMF": {"model": "ViT-S/16", "epochs": 4, "batch_size": 128, "lr": "3e-3 / 2e-3", "threshold": 0.5},
        "NegGrad+ + CMF": {"model": "ViT-S/16", "epochs": 3, "batch_size": 128, "lr": "5e-5 / 5e-4", "grad_clip": 1.0},
    },
    "CIFAR-100": {
        "Original": {"model": "ViT-S/16", "epochs": 10, "batch_size": 128, "lr": 3e-4, "notes": "pretrained backbone"},
        "Retrain": {"model": "ViT-S/16", "epochs": 10, "batch_size": 128, "lr": 3e-4, "notes": "pretrained backbone"},
        "Random Label": {"model": "ViT-S/16", "epochs": 3, "batch_size": 128, "lr": 3e-4},
        "SalUN": {"model": "ViT-S/16", "epochs": 3, "batch_size": 128, "lr": 3e-4, "threshold": 0.5},
        "NegGrad+": {"model": "ViT-S/16", "epochs": 3, "batch_size": 128, "lr": 3e-4, "grad_clip": 1.0},
        "Random Label + CMF": {"model": "ViT-S/16", "epochs": 4, "batch_size": 128, "lr": 2e-2},
        "SalUN + CMF": {"model": "ViT-S/16", "epochs": 4, "batch_size": 128, "lr": "2e-2 / 1e-2", "threshold": 0.5},
        "NegGrad+ + CMF": {"model": "ViT-S/16", "epochs": 3, "batch_size": 128, "lr": "5e-5 / 5e-4", "grad_clip": 1.0},
    },
    "Tiny-ImageNet": {
        "Original": {"model": "ViT-S/16", "epochs": 10, "batch_size": 128, "lr": 1e-4, "notes": "pretrained backbone"},
        "Retrain": {"model": "ViT-S/16", "epochs": 10, "batch_size": 128, "lr": 1e-4, "notes": "pretrained backbone"},
        "Random Label": {"model": "ViT-S/16", "epochs": 10, "batch_size": 128, "lr": 1e-4},
        "SalUN": {"model": "ViT-S/16", "epochs": 10, "batch_size": 128, "lr": 1e-4, "threshold": 0.5},
        "NegGrad+": {"model": "ViT-S/16", "epochs": 10, "batch_size": 128, "lr": 1e-4, "grad_clip": 1.0},
        "Random Label + CMF": {"model": "ViT-S/16", "epochs": 4, "batch_size": 128, "lr": 2e-3},
        "SalUN + CMF": {"model": "ViT-S/16", "epochs": 4, "batch_size": 128, "lr": 1e-2, "threshold": 0.5},
        "NegGrad+ + CMF": {"model": "ViT-S/16", "epochs": 3, "batch_size": 128, "lr": "1e-5 / 1e-2", "grad_clip": 1.0},
    }
}
