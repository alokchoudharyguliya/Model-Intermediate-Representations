This is a learning demonstration of the onnx graph based representation of model

Converting a model from one representation/runtime format into another that a different inference engine can execute

Trained model
     │
     ▼
Framework representation
(PyTorch / TensorFlow / JAX)
     │
     │ export / convert
     ▼
Intermediate representation
(ONNX / MLIR / StableHLO / TorchScript)
     │
     │ compile / optimize
     ▼
Runtime-specific artifact
(TensorRT Engine / OpenVINO IR / TFLite / etc.)
     │
     ▼
Inference
(CPU / CUDA / GPU / accelerator)


