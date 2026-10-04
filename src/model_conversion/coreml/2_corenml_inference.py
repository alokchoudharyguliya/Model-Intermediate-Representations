import time
import numpy as np
import coremltools as ct
import sys
from pathlib import Path

# Option A: Add the project root to sys.path dynamically (quickest fix)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from model_conversion.config.config import ARTIFACT_PATH
# 1. Fixed input
np.random.seed(42)
x=np.random.randn(
        1,3,224,224).astype(np.float32)

# 2. Benchmark Core ML with different compute units
compute_units={
        "CPU_ONLY":ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU":ct.ComputeUnit.CPU_AND_GPU,
        "ALL":ct.ComputeUnit.ALL,
        }

for name, compute_unit in compute_units.items():
    print("\n====={name}=====")
    model=ct.models.MLModel(
            str(ARTIFACT_PATH/"resnet18_fp32.mlpackage"),
            compute_units=compute_unit
            )
    # Core ML input/output names
    input_name=model.get_spec().description.input[0].name
    output_name=model.get_spec().description.output[0].name

    print("Input: ",input_name)
    print("Output: ", output_name)

    # Warm - Up
    for _ in range(10):
        result=model.predict({
            input_name:x})

    # Benchmark
    iterations=50
    start=time.perf_counter()
    for _ in range(iterations):
        result=model.predict({input_name:x})
    elapsed=time.perf_counter()-start
    latency=elapsed/iterations
    throughput=iterations/elapsed
    # Result
    output=result[output_name]
    print("Output Shape:",output.shape)
    print("Output Sample:",output[0,:5])
    print("Latency:",
          latency*1000,
          "ms")
    print("Throughput:",
          throughput,
          "samples/sec")
