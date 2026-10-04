import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat
)

# 1. Calibration Data

class CalibrationReader(CalibrationDataReader):
    def __init__(self,model_path):
        self.session=ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name=self.session.get_inputs()[0].name
        
        np.random.seed(42)
        
        self.data=[
            {
                self.input_name:
                    np.random.randn(
                        1,3,224,224
                    ).astype(np.float32)
            }
            for _ in range(100)
        ]
        self.index=0
    
    def get_next(self):
        if self.index>=len(self.data):
            return None
        
        item=self.data[self.index]
        self.index+=1
        return item
    
# FP32 ONNX -> INT8 ONNX

reader=CalibrationReader(
    "resnet18.onnx"
)
quantize_static(
    model_input="resnet18.onnx",
    model_output="resnet18_int8.onnx",
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
)
print("Created: resnet18_int8.onnx")