import torch
import numpy as np
import onnxruntime as ort
from torchvision.models import resnet18

device=torch.device("cpu")
print(device)

model=resnet18(weights=None)

state_dict=torch.load(
    "resnet18.pth",
    map_location=device
)
model.load_state_dict(state_dict)
model.eval()



torch.manual_seed(42)
x=torch.randn(
    1,3,224,224
)

with torch.inference_mode():
    pytorch_output=model(x)
pytorch_output=pytorch_output.numpy()

session=ort.InferenceSession(
    "resnet18.onnx",
    providers=["CPUExecutionProvider"]
)
input_name=session.get_inputs()[0].name
output_name=session.get_outputs()[0].name

onnx_output=session.run(
    [output_name],
    {
        input_name:x.numpy()
    }
)[0]

difference=np.abs(pytorch_output-onnx_output)
max_difference=difference.max()
mean_difference=difference.mean()

print("------------OUTPUT COMPARISON------------")
print("PyTorch Output shape:",pytorch_output.shape)
print("ONNX Output shape:",onnx_output.shape)

print("\nMaximum absolute difference")
print(max_difference)

print("\nMean absolute difference:")
print(mean_difference)

is_close=np.allclose(
    pytorch_output,
    onnx_output,
    rtol=1e-4,
    atol=1e-5
)

print("\nOutputs Match:",is_close)

pytorch_class=np.argmax(pytorch_output,axis=1)[0]
onnx_class=np.argmax(onnx_output,axis=1)[0]
print("\nPytorch predicted class:",pytorch_class)
print("ONNX predicted class:",onnx_class)

print("Same Prediction",pytorch_class==onnx_class)

