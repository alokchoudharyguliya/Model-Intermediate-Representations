import torch
import numpy as np
import coremltools as ct
from torchvision.models import resnet18

# 1. Load PyTorch model
model=resnet18(weights=None)

state_dict=torch.load(
        "resnet18.pth",
        map_location="cpu"
        )
model.load_state_dict(state_dict)
model.eval()

# 2. Same fixed input
torch.manual_seed(42)
x=torch.randn(
        1,3,224,224)
# 3. PyTorch inference
with torch.inference_mode():
    pytorch_output=model(x).numpy()


# 4. Core ML inference
coreml_model=ct.models.MLModel(
        "resnet18_fp32.mlpackage",
        compute_utils=ct.ComputeUnit.ALL
        )
input_name=coreml_model.get_spec().description.input[0].name
output_name=coreml_model.get_spec().description.output[0].name

coreml_output=coreml_model.predict(
        {input_name:x.numpy()}
        )[output_name]

# 5. Compare
difference=np.abs(pytorch_output-coreml_output)
print("-------PyTorch vs CORE ML-------")
print("PyTorch shape:", pytorch_output.shape)
print("Core ML shape:",coreml_output.shape)

print(
        "\nMaximum absolute difference:",
        difference.max()
        )
print(
        "\nMean absolute difference:",
        difference.mean()
        )
print(
        "\nSame prediction:",
        np.argmax(pytorch_output)==np.argmax(coreml_output))
