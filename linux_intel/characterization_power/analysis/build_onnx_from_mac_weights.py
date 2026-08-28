#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torchvision


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    arrays = np.load(args.weights)
    model = torchvision.models.resnet50(weights=None).eval()
    conv_modules = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    bn_modules = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    if len(conv_modules) != 53 or len(bn_modules) != 53:
        raise RuntimeError("Unexpected torchvision ResNet50 module counts")

    with torch.no_grad():
        for index, module in enumerate(conv_modules):
            module.weight.copy_(torch.from_numpy(arrays[f"conv_{index:02d}"].astype(np.float32)))
        for module in bn_modules:
            module.weight.fill_(1.0)
            module.bias.zero_()
            module.running_mean.zero_()
            module.running_var.fill_(1.0 - module.eps)
        model.fc.weight.copy_(torch.from_numpy(arrays["fc_weight"].astype(np.float32)))
        model.fc.bias.copy_(torch.from_numpy(arrays["fc_bias"].astype(np.float32)))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        str(output_path),
        input_names=["x"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    metadata = {
        "source_weights": str(args.weights),
        "source_weights_sha256": sha256(args.weights),
        "onnx": str(output_path),
        "onnx_sha256": sha256(output_path),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "input_shape": [1, 3, 224, 224],
        "onnx_opset": 17,
    }
    Path(args.metadata).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
