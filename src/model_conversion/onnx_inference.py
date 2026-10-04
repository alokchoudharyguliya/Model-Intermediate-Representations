# ONNX Artifact -> ONNX Runtime

import time
import numpy as np
import onnxruntime as ort
session=ort.InferenceSession(
    "resnet18.onnx",
    provides=["CPUExecutionProvider"]
)

print("Execution Providers:")
print(session.get_providers())

input_info=session.get_inputs()[0]
output_info=session.get_outputs()[0]

print("\nInput")
print(" name :",input_info.name)
print(" shape:", input_info.shape)
print(" type:", input_info.type)

print("\nOutput")
print(" name: ",output_info.name)
print(" shape: ",output_info.shape)
print(" type: ",output_info.type)

x=np.random.randn(
    1,3,224,224
).astype(np.float32)

for _ in range(30):
    session.run(
        [output_info.name],
        {input_info.name:x}
    )
    
    
iterations=100
start=time.perf_counter()
for _ in range(iterations):
    output=session.run(
        [output_info.name],
        {input_info.name:x}
    )
elapsed=time.perf_counter()-start
latency=elapsed/iterations
throughput=iterations/elapsed
output=output[0]
print("\n ----- ONNX Runtime ------")
print("Output Shape:", output.shape)
print("Output sample:")
print(output[0,:15])
print("\n Latency",latency*1000, "ms")
print("Throughput", throughput, "samples/sec")
