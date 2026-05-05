from __future__ import annotations


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_model(model_name: str = "efficientnet_b0", num_classes: int = 5, pretrained: bool = True):
    """Create a classifier. Prefer timm, with torchvision fallback for smoke runs."""
    try:
        import timm

        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except Exception as timm_error:
        try:
            from torchvision import models
            import torch.nn as nn

            if model_name in {"efficientnet_b0", "tf_efficientnet_b0"}:
                weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                model = models.efficientnet_b0(weights=weights)
                in_features = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(in_features, num_classes)
                return model
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        except Exception as fallback_error:
            raise RuntimeError(
                "Gagal membuat model. Install dependency dengan: pip install -r requirements.txt"
            ) from fallback_error
