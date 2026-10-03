import onnx
model=onnx.load("resnet18.onnx")
onnx.checker.check_model(model)
print("ONNX model is valid")
print("\nInputs")
for tensor in model.graph.input:
    print(tensor.name,
          tensor.type.tensor_type.shape)
print("\nOutputs")
for tensor in model.graph.output:
    print(
        tensor.name,
        tensor.type.tensor_type.shape
    )
print("\nNumber of graph nodes:")
print(len(model.graph.node))
for node in model.graph.node[:10]:
    print(node.op_type)