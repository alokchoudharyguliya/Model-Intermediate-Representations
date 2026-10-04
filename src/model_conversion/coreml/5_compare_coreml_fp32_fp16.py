import time
import os
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

# 2. Load FP32 and FP16 Core ML models
models={
        "FP32":ct.models.MLModel(
            str(ARTIFACT_PATH/"resnet18_fp32.mlpackage"),
            compute_units=ct.ComputeUnit.ALL
            ),
        "FP16":ct.models.MLModel(
            str(ARTIFACT_PATH/"resnet18_fp16.mlpackage"),
            compute_units=ct.ComputeUnit.ALL
            ),
        }

# 3. Run both models once
outputs={}

for name,model in models.items():
    input_name=model.get_spec().description.input[0].name
    output_name=model.get_spec().description.output[0].name
    outputs[name]=model.predict({
        input_name:x
        })[output_name]


# 4. Compare outputs
difference=np.abs(
        outputs["FP32"]-outputs["FP16"]
        )
print("-----------FP32 vs FP16----------")
print("Maximum absolute difference",
      difference.max()
      )
print("Mean absolute difference",
      difference.mean()
      )
print("Same prediction:",
      np.argmax(outputs["FP32"])==np.argmax(outputs["FP16"])
      )

# 5. Benchmark both
print("\n------Latency-------")
iterations=50
for name, model in models.items():
    input_name=model.get_spec().description.input[0].name
    output_name=model.get_spec().description.output[0].name
    # Warm - up 
    for _ in range(10):
        model.predict({
            input_name:x
            })
    start=time.perf_counter()
    for _ in range(iterations):
        model.predict({
            input_name:x
            })
    elapsed=time.perf_counter()-start
    latency=elapsed/iterations
    print(f"{name}:{latency*1000:.3f}ms"
          )
# 6. Compare artifact sizes
print("\n ----- MODEL SIZE-------")
for name, path in {
    "FP32":str(ARTIFACT_PATH/"resnet18_fp32.mlpackage"),
    "FP16":str(ARTIFACT_PATH/"resnet18_fp16.mlpackage"),
        }.items():
    total_size=0
    for root, dirs, files in os.walk(path):
        for file in files:
            total_size+=os.path.getsize(
                    os.path.join(root,file)
                    )
    size_mb=total_size/(1034**2)
    print(f"{name};{size_mb:.2f}MB")
