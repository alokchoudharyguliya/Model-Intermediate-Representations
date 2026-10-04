import time, os
import numpy as np
import onnxruntime as ort

# 1. Fixed Input
np.random.seed(42)
x=np.random.randn(1,3,224,224).astype(np.float32)

# 2. Load models
models={
    "FP32":"resnet18.onnx",
    "FP16":"resnet18_fp16.onnx",
    "INT8":"resnet18_int8.onnx",
}

sessions={}
for name, path in models.items():
    sessions[name]=ort.InferenceSession(
        path,
        providers=["CPUExecutionProvider"]
    )

# 3. Run one inference from each model

outputs={}
for name, session in sessions.items():
    input_name=session.get_inputs()[0].name
    output_name=session.get_outputs()[0].name
    outputs[name]=session.run(
        [output_name],
        {input_name:x}
    )[0]

# 4. Compare
reference=outputs["FP32"]
print("------OUTPUT DIFFERENCE------")
for name in ["FP16","INT8"]:
    difference=np.abs(
        reference-outputs[name]
    )
    print(f"\nFP32 vs {name}")
    print(
        "Max difference",
        difference.max()
    )
    print(
        "Mean difference",
        difference.mean()
    )
    print(
        "Same prediction",
        np.argmax(reference),
        np.argmax(outputs[name])
    )

# 5. Benchmark
print("\n ----- LATENCY ------- ")
iterations=100
for name, session in sessions.items():
    input_name=session.get_inputs()[0].name
    output_name=session.get_outputs()[0].name
    
    for _ in range(20):
        session.run(
            [output_name],
            {
                input_name:x
            }
        )
    start=time.perf_counter()
    for _ in range(iterations):
        session.run(
            [output_name],
            {input_name:x}
        )
    elapsed=time.perf_counter()-start
    latency=elapsed/iterations
    print(f"{name}:{latency*1000:.3f}ms")
    
# 6. Compare artifact sizes
print("\n -----Model sizes-----")
for name, path in models.items():
    size=os.path.getsize(path)/(1024**2)
    print(f"{name}:{size:.2f}MB")
        