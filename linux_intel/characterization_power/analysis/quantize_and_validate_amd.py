#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static
from onnxruntime.quantization.calibrate import CalibrationMethod


SEED = 20260716


class CalibrationReader(CalibrationDataReader):
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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    outputs = {}
    for label, value in [("black", 0.0), ("white", 1.0)]:
        array = np.full((1, 3, 224, 224), value, dtype=np.float32)
        output = session.run(None, {input_meta.name: array})[0]
        outputs[label] = {
            "mean": float(output.mean()),
            "std": float(output.std()),
            "finite": bool(np.isfinite(output).all()),
        }
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    quantize_static(
        model_input=str(input_path),
        model_output=str(output_path),
        calibration_data_reader=CalibrationReader(),
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        use_external_data_format=False,
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )
    metadata = {
        "float_model": str(input_path),
        "float_sha256": sha256(input_path),
        "float_validation": validate(input_path),
        "int8_model": str(output_path),
        "int8_sha256": sha256(output_path),
        "int8_validation": validate(output_path),
        "calibration_seed": SEED,
        "calibration_samples": 32,
    }
    Path(args.metadata).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
