#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torchvision
from coremltools.converters.mil.frontend.milproto.load import load


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlpackage", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    package = Path(args.mlpackage)
    model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = model.get_spec()
    program = load(
        spec,
        spec.specificationVersion,
        str(package / "Data" / "com.apple.CoreML" / "weights"),
    )
    operations = list(program.functions["main"].operations)
    conv_ops = [operation for operation in operations if operation.op_type == "conv"]
    linear_ops = [operation for operation in operations if operation.op_type == "linear"]

    torch_model = torchvision.models.resnet50(weights=None).eval()
    conv_modules = [module for module in torch_model.modules() if isinstance(module, torch.nn.Conv2d)]
    bn_modules = [module for module in torch_model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    if len(conv_ops) != 53 or len(conv_modules) != 53 or len(bn_modules) != 53 or len(linear_ops) != 1:
        raise RuntimeError("Unexpected ResNet50/CoreML operation counts")

    arrays = {}
    with torch.no_grad():
        for index, (operation, module) in enumerate(zip(conv_ops, conv_modules)):
            weight = np.asarray(operation.inputs["weight"].val)
            bias = np.asarray(operation.inputs["bias"].val)
            if tuple(weight.shape) != tuple(module.weight.shape):
                raise RuntimeError(f"Conv {index} shape mismatch: {weight.shape} != {tuple(module.weight.shape)}")
            if np.any(bias):
                raise RuntimeError(f"Conv {index} has non-zero fused bias")
            module.weight.copy_(torch.from_numpy(weight.astype(np.float32)))
            arrays[f"conv_{index:02d}"] = weight.astype(np.float16)

        for module in bn_modules:
            module.weight.fill_(1.0)
            module.bias.zero_()
            module.running_mean.zero_()
            module.running_var.fill_(1.0 - module.eps)

        linear = linear_ops[0]
        linear_weight = np.asarray(linear.inputs["weight"].val)
        linear_bias = np.asarray(linear.inputs["bias"].val)
        torch_model.fc.weight.copy_(torch.from_numpy(linear_weight.astype(np.float32)))
        torch_model.fc.bias.copy_(torch.from_numpy(linear_bias.astype(np.float32)))
        arrays["fc_weight"] = linear_weight.astype(np.float16)
        arrays["fc_bias"] = linear_bias.astype(np.float16)

    input_name = spec.description.input[0].name
    output_name = spec.description.output[0].name
    validation = {}
    for label, value in [("black", 0.0), ("white", 1.0)]:
        array = np.full((1, 3, 224, 224), value, dtype=np.float32)
        coreml_output = np.asarray(model.predict({input_name: array})[output_name])
        with torch.no_grad():
            torch_output = torch_model(torch.from_numpy(array)).numpy()
        error = np.abs(coreml_output - torch_output)
        validation[label] = {
            "coreml_mean": float(coreml_output.mean()),
            "torch_mean": float(torch_output.mean()),
            "max_abs_error": float(error.max()),
            "mean_abs_error": float(error.mean()),
            "relative_l2_error": float(np.linalg.norm(error) / max(np.linalg.norm(coreml_output), 1e-12)),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)
    metadata = {
        "mlpackage": str(package),
        "model_spec_sha256": sha256(package / "Data" / "com.apple.CoreML" / "model.mlmodel"),
        "coreml_weights_sha256": sha256(package / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"),
        "extracted_weights_sha256": sha256(output_path),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "conv_count": len(conv_ops),
        "linear_count": len(linear_ops),
        "input_name": input_name,
        "output_name": output_name,
        "validation": validation,
    }
    Path(args.metadata).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
