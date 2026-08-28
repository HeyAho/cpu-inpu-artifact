#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SEED = 20260716
MODEL_SOURCE = "pytorch/vision:v0.19.0"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_float_model(output_path, vision_source_dir=None):
    import torch
    import torchvision

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if vision_source_dir:
        requested_source = Path(vision_source_dir) / "torchvision" / "models" / "resnet.py"
        installed_source = Path(torchvision.models.resnet.__file__)
        if sha256(requested_source) != sha256(installed_source):
            raise RuntimeError("Installed torchvision ResNet source differs from requested v0.19.0 source")
    model = torchvision.models.resnet50(weights=None)
    model.eval()
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
    )
    return {
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "resnet_source_sha256": sha256(torchvision.models.resnet.__file__),
    }


def quantize_for_amd(float_path, int8_path):
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static
    from onnxruntime.quantization.calibrate import CalibrationMethod

    class AlignedCalibrationReader(CalibrationDataReader):
        def __init__(self):
            rng = np.random.default_rng(SEED)
            self.samples = [
                {"x": np.zeros((1, 3, 224, 224), dtype=np.float32)},
                {"x": np.ones((1, 3, 224, 224), dtype=np.float32)},
            ]
            self.samples.extend(
                {"x": rng.uniform(0.0, 1.0, (1, 3, 224, 224)).astype(np.float32)}
                for _ in range(30)
            )
            self.index = 0

        def get_next(self):
            if self.index >= len(self.samples):
                return None
            sample = self.samples[self.index]
            self.index += 1
            return sample

        def rewind(self):
            self.index = 0

    quantize_static(
        model_input=str(float_path),
        model_output=str(int8_path),
        calibration_data_reader=AlignedCalibrationReader(),
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        use_external_data_format=False,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
    )


def validate_model(path):
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    outputs = {}
    for label, value in [("black", 0.0), ("white", 1.0)]:
        array = np.full((1, 3, 224, 224), value, dtype=np.float32)
        output = session.run(None, {input_meta.name: array})[0]
        outputs[label] = {
            "shape": list(output.shape),
            "mean": float(output.mean()),
            "std": float(output.std()),
            "finite": bool(np.isfinite(output).all()),
        }
    return {
        "input_name": input_meta.name,
        "input_shape": list(input_meta.shape),
        "input_type": input_meta.type,
        "outputs": outputs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--quantize-amd", action="store_true")
    parser.add_argument("--vision-source-dir")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    float_path = outdir / "resnet50_torchvision019_untrained_seed20260716_b1.onnx"
    int8_path = outdir / "resnet50_torchvision019_untrained_seed20260716_b1_int8.onnx"

    framework = export_float_model(float_path, args.vision_source_dir)
    metadata = {
        "architecture": "ResNet50",
        "model_source": MODEL_SOURCE,
        "vision_source_dir": args.vision_source_dir,
        "weights": None,
        "seed": SEED,
        "batch_size": 1,
        "input_shape": [1, 3, 224, 224],
        "input_dtype": "float32",
        "black_value": 0.0,
        "white_value": 1.0,
        **framework,
        "onnx_opset": 17,
        "float_model": {
            "path": str(float_path),
            "sha256": sha256(float_path),
            "validation": validate_model(float_path),
        },
    }

    if args.quantize_amd:
        quantize_for_amd(float_path, int8_path)
        metadata["amd_int8_model"] = {
            "path": str(int8_path),
            "sha256": sha256(int8_path),
            "validation": validate_model(int8_path),
            "note": "Same ONNX graph and weights, statically quantized to QDQ INT8 for AMD NPU execution.",
        }

    (outdir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
