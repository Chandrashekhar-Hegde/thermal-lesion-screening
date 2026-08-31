"""
Model definition.

Deliberately a small, ImageNet-pretrained ResNet-18 rather than something
larger. On a dataset of a few hundred patients, a bigger backbone buys you
overfitting and a longer inference time on the Pi, not accuracy. Choosing the
smallest model that clears the clinical bar is an engineering decision worth
being able to defend out loud.
"""

from __future__ import annotations

import inspect
from pathlib import Path

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
    path: str | Path = "artifacts/model.onnx",
    image_size: tuple[int, int] = (224, 224),
) -> str:
    """Export for edge deployment. ONNX Runtime on the Pi, not PyTorch."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    dummy = torch.randn(1, 3, *image_size, device=device)
    export_options = {
        "input_names": ["image"],
        "output_names": ["logit"],
        "opset_version": 18,
    }
    export_parameters = inspect.signature(torch.onnx.export).parameters
    if "dynamo" in export_parameters and "dynamic_shapes" in export_parameters:
        export_options.update(dynamo=True, dynamic_shapes=({0: "batch"},))
    else:  # PyTorch 2.0–2.5 legacy exporter
        export_options["dynamic_axes"] = {
            "image": {0: "batch"},
            "logit": {0: "batch"},
        }
    torch.onnx.export(model, dummy, str(output_path), **export_options)
    return str(output_path)
