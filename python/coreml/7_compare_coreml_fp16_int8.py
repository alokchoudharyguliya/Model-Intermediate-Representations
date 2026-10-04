import time, os
import numpy as np
import coremltools as ct

# 1. Fixed Input
np.random.seed(42)
x=np.random.randn(1,3,224,224).astype(np.float32)

# 2. Load FP16 and INT8 models
models={
    "FP16":ct.models.MLModel(
        "resnet18_fp16.mlpackage",
        compute_units=ct.ComputeUnit.ALL
        ),
    "INT8":ct.models.MLModel(
        "resnet18_it8.mlpackage",
        compute_units=ct.ComputeUnit.ALL
        ),
    }
# 3. Run Inference
outputs={}
for name, model in models.items():
    input_name=model.get_spec().description.input[0].name
    output_name=model.get_spec().description.output[0].name
    outputs[name]=model.predict({
        input_name:x
        })[output_name]

# 4. Compare outputs
difference=np.abs(
        outputs["FP16"]-outputs["INT8"]
        )
print("------FP16 vs INT8-------")
print(
        "Maxiumum absolute difference",
        difference.max()
        )
print("Mean absolute difference",
      difference.mean()
      )
print("Same prediction:",
      np.argmax(outputs["FP16"])==
      np.argmax(outputs["INT8"])
      )
# 5. Benchmark
print("\n ------ LATENCY ------")
iterations=50
for name, model in models.items():
    input_name=model.get_spec().description.input[0].name
    output_name=model.get_spec().description.output[0].name
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
    print(
            f"{name}:{(elapsed/iterations)*1000:.3f}ms"
            )

# 6. Artifact sizes
print("\n ------MODEL SIZE-------")
for name,path in {
        "FP16":str(ARTIFACT_PATH/"resnet18_fp16.mlpackage"),
        "INT8":str(ARTIFACT_PATH/"resnet18_int8.mlpackage"),
        }.items():
    total_size=0
    for root, dirs, files in os.walk(path):
        for file in files:
            total_size+=os.path.getsize(
                    os.path.join(root,file)
                    )
    print(
            f"{name}:{total_size/(1024**2):.2f} MB"
            )




