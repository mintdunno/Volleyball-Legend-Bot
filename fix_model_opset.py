import onnx
import shutil
import os

original_model = "best.onnx"
backup_model = "best.onnx.bak"
fixed_model = "best_fixed.onnx"

# Backup
if not os.path.exists(backup_model):
    shutil.copy(original_model, backup_model)
    print(f"Backed up {original_model} to {backup_model}")

try:
    # Load
    model = onnx.load(original_model)
    current_version = model.opset_import[0].version
    print(f"Current Opset Version: {current_version}")
    
    # Modify
    # We change it to 17, which is a very stable version for YOLO models and supported by most runtimes
    # Changing to 21 is also an option, but 17 is safer for "friendliness" with slightly older drivers/runtimes
    target_version = 17
    
    if current_version > target_version:
        print(f"Downgrading Opset from {current_version} to {target_version}...")
        model.opset_import[0].version = target_version
        onnx.save(model, original_model) # Overwrite original
        print(f"Success! {original_model} is now Opset {target_version}")
    else:
        print(f"Opset is already {current_version} (<= {target_version}). No change needed.")

except Exception as e:
    print(f"Failed to fix model: {e}")
