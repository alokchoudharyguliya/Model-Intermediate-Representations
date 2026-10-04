import torch
import coremltools as ct
from torchvision.models import resnet18

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from model_conversion.config.config import ARTIFACT_PATH

# 1. Load PyTorch model
model=resnet18(weights=None)
state_dict=torch.load(
    str(ARTIFACT_PATH/"resnet18.pth"),
    map_location="cpu"
    )
model.load_state_dict(state_dict)
model.eval()

x=torch.randn(
        1,3,224,224
        )
traced_model=torch.jit.trace(
        model,x
        )

# 4. Convert PyTorch -> Core ML FP16
coreml_model=ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="input",
                shape=x.shape
                )
            ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16
        )

# 5. Save FP6 artifact
coreml_model.save(str(ARTIFACT_PATH/"resnet18_fp16.mlpackage"))
print("Created: resnet18_fp16.mlpackage")

