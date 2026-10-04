import torch
from torchvision.models import resnet18

import sys
from pathlib import Path

# Option A: Add the project root to sys.path dynamically (quickest fix)
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from config import ARTIFACT_PATH

device=torch.device("cpu")
model=resnet18(weights=None)
stat_dict=torch.load(
    str(ARTIFACT_PATH/"resnet18.pth"),
    map_location=device
)
x=torch.randn(1,3,224,224,device=device)
torch.onnx.export(model,(x,),str(ARTIFACT_PATH/"resnet18.onnx"),input_names=["input"],output_names=["output"],dynamo=True)

print("Exported: resnet18.onnx")
