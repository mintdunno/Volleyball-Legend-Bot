import onnx

input_model_path = "best.onnx"
output_model_path = "best_320.onnx"
target_size = 320

print(f"Loading {input_model_path}...")
model = onnx.load(input_model_path)

# Check graph inputs
for input_tensor in model.graph.input:
    curr_shape = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
    print(f"Current Input Shape: {curr_shape}")
    
    # Force modify dimensions to 320
    # Usually YOLO input is [1, 3, 640, 640]
    # We change dimensions 2 and 3
    input_tensor.type.tensor_type.shape.dim[2].dim_value = target_size
    input_tensor.type.tensor_type.shape.dim[3].dim_value = target_size
    
    print(f"New Input Shape: {input_tensor.type.tensor_type.shape.dim}")

print(f"Saving to {output_model_path}...")
onnx.save(model, output_model_path)
print("Done! Input size forced to 320x320.")
print("NOTE: This relies on the model being dynamic or resize-tolerant. If inference fails, we revert.")
