import os
import time
import numpy as np
import onnxruntime as ort
import coremltools as ct

import sys
from pathlib import Path

# Option A: Add the project root to sys.path dynamically (quickest fix)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from config import ARTIFACT_PATH

# Configuration
INPUT_SHAPE=(1,3,224,224)
ITERATIONS=50
WARMUP=10

#Fixed Input
np.random.seed(42)

x=np.random.randn(
        *INPUT_SHAPE).astype(np.float32)

# Helpers
def directory_size(path):
    total=0
    for root, _, files in os.walk(path):
        for file in files:
            total+=os.path.getsize(
                    os.path.join(root,file)
                    )
    return total/(1024**2)

def benchmark_onnx(path):
    session=ort.InferenceSession(
            path,
            providers=["CPUExecutionProvider"]
            )
    input_name=session.get_inputs()[0].name
    output_name=session.get_outputs()[0].name

    for _ in range(WARMUP):
        session.run([output_name],
                    {input_name:x}
                    )
        start=time.perf_counter()
        for _ in range(ITERATIONS):
            output=session.run([output_name],
                               {input_name:x})[0]
        elapsed=time.perf_counter()-start
    return (output,
                elapsed/ITERATIONS*1000,
                ITERATIONS/elapsed
                )
def benchmark_coreml(path):
    model=ct.models.MLModel(
            path, compute_units=ct.ComputeUnit.ALL)
    input_name=model.get_spec().description.input[0].name
    output_name=model.get_spec().description.output[0].name
    for _ in range(WARMUP):
        model.predict({
            input_name:x
            })
    start=time.perf_counter()
    for _ in range(ITERATIONS):
        output=model.predict({
            input_name:x
            })[output_name]
    elapsed=time.perf_counter()-start
    return (
            output,
            elapsed/ITERATIONS*1000,
            ITERATIONS/elapsed
            )
onnx_models={
        "ONNX FP32":str(ARTIFACT_PATH/"resnet18.onnx"),
        "ONNX FP16":str(ARTIFACT_PATH/"resnet18_fp16.onnx"),
        "ONNX INT8":str(ARTIFACT_PATH/"resnet18_int8.onnx"),
        }

results={}
print("\n===========ONNX==============")
for name, path in onnx_models.items():
    output,latency,throughput=benchmark_onnx(path)
    size=os.path.getsize(path)/(1024**2)
    results[name]={
            "output":output,
            "latency":latency,
            "throughput":throughput,
            "size":size,
            }

    print(
            f"{name:12} |"
            f"{size:8.2f} MB |"
            f"{latency:8.3f}ms |"
            f"{throughput:8.2f} samples/s"
            )


# BENCHMARK Core ML
coreml_models={
    "CoreML FP32":str(ARTIFACT_PATH/"resnet18_fp32.mlpackage"),
    "CoreML FP16":str(ARTIFACT_PATH/"resnet18_fp16.mlpackage"),
    "CoreML INT8":str(ARTIFACT_PATH/"resnet18_int8.mlpackage"),
        }

print("\n======== CORE ML =============")
for name, path in coreml_models.items():
    output, latency, throughput=benchmark_coreml(path)
    size=directory_size(path)
    results[name]={
            "output":output,
            "latency":latency,
            "throughput":throughput,
            "size":size,
            }
    print(
            f"{name:12} | "
            f"{size:8.2f} MB |"
            f"{latency:8.3f} ms | "
            f"{throughput:8.2f} samples/s"
            )


# Compare numerical outputs
reference=results["ONNX FP32"]["output"]
print("\n==========OUTPUT DIFFERENCES============")
for name, result in results.items():
    difference=np.abs(
            reference-result["output"]
            )
    print(
            f"{name:12}| "
            f"max={difference.max():.8f} | "
            f"mean={difference.mean():.8f} | "
            f"same_class="
            f"{np.argmax(reference)==np.argmax(result['output'])}"
            )
