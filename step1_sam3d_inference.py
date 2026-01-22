#!/usr/bin/env python3
"""
Sam3D Inference Script (Step 1)
Runs Sam3D to get MHR parameters and saves them
"""
import sys
import os
from pathlib import Path

# Add sam-3d-body to path (relative to this script's directory)
SCRIPT_DIR = Path(__file__).parent.absolute()
SAM3D_BODY_DIR = SCRIPT_DIR / "sam-3d-body"
if str(SAM3D_BODY_DIR) not in sys.path:
    sys.path.insert(0, str(SAM3D_BODY_DIR))

import cv2
import torch
import numpy as np
import pickle

# Sam3D imports
from notebook.utils import setup_sam_3d_body

def run_sam3d_inference(image_path, output_dir="./output"):
    """
    Run Sam3D inference and save MHR parameters
    """
    print(f"\n{'='*60}")
    print(f"SAM3D INFERENCE")
    print(f"{'='*60}\n")
    
    # Step 1: Load Sam3D model
    print("1️⃣ Loading Sam3D model...")
    estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")
    print("   ✅ Sam3D loaded\n")
    
    # Step 2: Run Sam3D inference
    print(f"2️⃣ Processing image: {os.path.basename(image_path)}")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"   ❌ Failed to load image: {image_path}")
        return None
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    outputs = estimator.process_one_image(img_rgb)
    print("   ✅ Sam3D inference complete\n")
    
    # Step 3: Inspect outputs
    print("3️⃣ Sam3D Output Structure:")
    
    # outputs is a list of detections
    if isinstance(outputs, list):
        print(f"   Found {len(outputs)} detection(s)")
        if len(outputs) == 0:
            print("   ❌ No person detected in image")
            return None
        # Use first detection
        output = outputs[0]
    else:
        output = outputs
    
    print(f"   Keys: {list(output.keys())}")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"   - {key}: Tensor {value.shape}")
        elif isinstance(value, np.ndarray):
            print(f"   - {key}: Array {value.shape}")
        else:
            print(f"   - {key}: {type(value)}")
    print()
    
    # Step 4: Save outputs
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sam3d_output.pkl")
    
    # Convert tensors to numpy for saving
    save_dict = {}
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            save_dict[key] = value.cpu().numpy()
        else:
            save_dict[key] = value
    
    with open(output_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"4️⃣ Saved outputs to: {output_path}\n")
    
    # Step 5: Visualize
    print("5️⃣ Creating visualization...")
    from tools.vis_utils import visualize_sample_together
    # visualize_sample_together expects a list
    rend_img = visualize_sample_together(img_bgr, [output], estimator.faces)
    vis_path = os.path.join(output_dir, f"{Path(image_path).stem}_sam3d_result.jpg")
    cv2.imwrite(vis_path, rend_img.astype(np.uint8))
    print(f"   ✅ Saved visualization to: {vis_path}\n")
    
    print(f"{'='*60}")
    print("✅ Sam3D inference complete!")
    print(f"{'='*60}\n")
    
    return save_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sam3D Inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default="./output", help="Output directory")
    
    args = parser.parse_args()
    
    run_sam3d_inference(args.image, args.output)
