import onnx
from onnxconverter_common import float16
import sys
from pathlib import Path

# Option A: Add the project root to sys.path dynamically (quickest fix)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from model_conversion.config.config import ARTIFACT_PATH

model=onnx.load(str(ARTIFACT_PATH/"resnet18.onnx"))


model_fp16=float16.convert_float_to_float16(
    model,keep_io_types=True
)
onnx.save(model_fp16,str(ARTIFACT_PATH/"resnet18_fp16.onnx"))
print("Created: resnet18_fp16.onnx")
