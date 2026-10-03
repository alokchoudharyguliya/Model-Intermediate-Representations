import torch
from torchvision.models import resnet18
device=torch.device("cpu")
model=resnet18(weights=None)
stat_dict=torch.load(
    "resnet18.pth",
    map_location=device
)
x=torch.randn(1,3,224,224,device=device)
torch.onnx.export(model,(x,),"resnet18.onnx",input_names=["input"],output_names=["output"],dynamo=True)

print("Exported: resnet18.onnx")