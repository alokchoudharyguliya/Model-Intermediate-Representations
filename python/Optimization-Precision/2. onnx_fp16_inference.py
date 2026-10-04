import time, os
import numpy as np
import onnxruntime as ort

np.random.seed(42)
x=np.random.randn(
    1,3,224,224
).astype(np.float32)

fp32_session=ort.InferenceSession(
    "resnet18.onnx",
    providers=["CPUExecutionProvider"]
)
fp32_input=fp32_session.get_inputs()[0].name
fp32_output=fp32_session.get_outputs()[0].name

fp16_session=ort.InferenceSession(
    "resnet_fp16.onnx",
    providers=["CPUExecutionProvider"]
)
fp16_input=fp16_session.get_inputs()[0].name
fp16_output=fp16_session.get_outputs()[0].name

fp32_result=fp32_session.run(
    [fp32_output],
    {fp32_input:x}
)[0]



fp16_result=fp16_session.run(
    [fp16_output],
    {fp16_input:x}
)[0]

difference=np.abs(fp32_result-fp16_result)
print("---------------FP32 vs FP16---------------")
print("Maximum absolute difference",difference.max())

print("Mean absolute difference",difference.mean())

print("Same prediction",np.argmax(fp32_result==np.argmax(fp16_session)))

for _ in range(20):
    fp32_session.run(
        [fp32_output],
        {fp32_input:x}
    )
iterations=100

start=time.perf_counter()

for _ in range(iterations):
    fp32_session.run(
        [fp32_output],
        {fp32_input:x}
    )
fp32_time=time.perf_counter()-start


for _ in range(20):
    fp16_session.run(
        [fp16_output],
        {fp16_input:x}
    )


start=time.perf_counter()

for _ in range(iterations):
    fp16_session.run(
        [fp16_output],
        {fp16_input:x}
    )
fp16_time=time.perf_counter()-start
fp32_latency=fp32_time/iterations

fp16_latency=fp16_time/iterations

print("\n--------Latency\n")
print("FP32:",fp32_latency*1000,"ms")
print("FP16:",fp16_latency*1000,"ms")


fp32_size=os.path.getsize(
    "resnet18.onnx"
)/(1024**2)


fp16_size=os.path.getsize(
    "resnet_fp16.onnx"
)/(1024**2)

print("\n-------Model Size")
print("FP32:",fp32_size)
print("FP16:",fp16_size)


