# Core ML program default to FP16 in Core ML Tools, so we delibrately request FP32 first to establish clean baseline
import torch
import coremltools as ct
import numpy as np
from torchvision.models import resnet18
from pathlib import Path
import sys
ROOT_DIR=Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
from model_conversion.config.config import ARTIFACT_PATH

# 1. Load our PyTorch model artifact
model=resnet18(weights=None)
state_dict=torch.load(
        ARTIFACT_PATH/"resnet18.pth",
        map_location="cpu"
)
model.load_state_dict(state_dict)
model.eval()

# 2, Example input
x=torch.randn(
        1,3,224,224
        )

# 3. Capture the Pytorch model graph
traced_model=torch.jit.trace(
        model,
        x)
# 4. PyTorch -> Core ML
coreml_model=ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="input",
                shape=x.shape,
                dtype=np.float32)
            ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32
        )
# 5. Save Core ML artifact
coreml_model.save(
        ARTIFACT_PATH/"resnet18_fp32.mlpackage"
        )
print("Created: resnet18_fp32.mlpackage")
