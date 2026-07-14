"""
Model definition.

Deliberately a small, ImageNet-pretrained ResNet-18 rather than something
larger. On a dataset of a few hundred patients, a bigger backbone buys you
overfitting and a longer inference time on the Pi, not accuracy. Choosing the
smallest model that clears the clinical bar is an engineering decision worth
being able to defend out loud.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


def build_model(backbone: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """Binary classifier with a single logit output."""
    if backbone == "resnet18":
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, 1)
    elif backbone == "mobilenet_v3_small":
        # Worth trying when Pi latency is the binding constraint.
        weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.mobilenet_v3_small(weights=weights)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 1)
    else:
        raise ValueError(f"unsupported backbone: {backbone}")
    return net


def export_onnx(
    model: nn.Module,
    path: str = "artifacts/model.onnx",
    image_size: tuple[int, int] = (224, 224),
) -> str:
    """Export for edge deployment. ONNX Runtime on the Pi, not PyTorch."""
    model.eval()
    dummy = torch.randn(1, 3, *image_size)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["image"],
        output_names=["logit"],
        dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
    )
    return path
