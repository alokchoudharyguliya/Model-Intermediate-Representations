import coremltools as ct
from coremltools.optimize.coreml import(
        OpLinearQuantizerConfig,
        linear_quantize_weights,
        )
import coremltools.optimize as cto
import sys
from pathlib import Path

# Option A: Add the project root to sys.path dynamically (quickest fix)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from model_conversion.config.config import ARTIFACT_PATH

# 1. Load FP16 Core ML Model
model=ct.models.MLModel(
        str(ARTIFACT_PATH/"resnet18_fp16.mlpackage")
        )

# 2. Define INT8 weight quantization
config_op=cto.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int8",
        )
config_=cto.coreml.OptimizationConfig(
                global_config=config_op)

# 3. Quantize weights
quantized_model=linear_quantize_weights(model,config=config_)

# 4. Save INT8 artifact
quantized_model.save(
        str(ARTIFACT_PATH/"resnet18_int8.mlpackage")
        )
print("Created: resnet18_int8.mlpackage")
