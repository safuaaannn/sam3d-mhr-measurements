#!/usr/bin/env python3
"""
Step 2: MHR Measurement from Sam3D Output
Loads Sam3D predictions and calculates body measurements using MHR
Uses shared measurement calculator for consistency with interactive tool
"""
import sys
from pathlib import Path

# Add project root to path (relative to this script's directory)
SCRIPT_DIR = Path(__file__).parent.absolute()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pickle
import torch
import numpy as np

# MHR imports
from mhr.mhr import MHR
from measurement_calculator import MeasurementCalculator


def calculate_measurements_from_sam3d(sam3d_output_path, target_height_cm=175):
    """
    Calculate measurements from Sam3D output using MHR
    """
    print(f"\n{'='*60}")
    print("MHR MEASUREMENT CALCULATION")
    print(f"{'='*60}\n")
    
    # Step 1: Load Sam3D output
    print("1️⃣ Loading Sam3D output...")
    with open(sam3d_output_path, 'rb') as f:
        sam3d_data = pickle.load(f)
    print(f"   ✅ Loaded from: {sam3d_output_path}\n")
    
    # Step 2: Extract MHR parameters
    print("2️⃣ Extracting MHR parameters...")
    identity_coeffs = torch.from_numpy(sam3d_data['shape_params']).unsqueeze(0).float()
    model_parameters = torch.from_numpy(sam3d_data['mhr_model_params']).unsqueeze(0).float()
    face_expr_coeffs = torch.from_numpy(sam3d_data['expr_params']).unsqueeze(0).float()
    
    print(f"   - Identity coeffs: {identity_coeffs.shape}")
    print(f"   - Model parameters: {model_parameters.shape}")
    print(f"   - Expression coeffs: {face_expr_coeffs.shape}\n")
    
    # Step 3: Load MHR model and generate mesh
    print("3️⃣ Loading MHR model...")
    device = torch.device("cpu")
    mhr = MHR.from_files(device=device, lod=1)
    print("   ✅ MHR loaded\n")
    
    print("4️⃣ Generating high-quality mesh...")
    with torch.no_grad():
        vertices, skel_state = mhr(identity_coeffs, model_parameters, face_expr_coeffs)
    
    vertices_np = vertices[0].numpy()
    joints = skel_state[0, :, :3].numpy()
    faces = mhr.character.mesh.faces
    joint_names = mhr.character.skeleton.joint_names
    
    print(f"   - Vertices: {vertices_np.shape}")
    print(f"   - Joints: {joints.shape}")
    print(f"   ✅ Mesh generated\n")
    
    # Step 4: Calculate scaling
    print("5️⃣ Calculating scale factor...")
    y_min = vertices_np[:, 1].min()
    y_max = vertices_np[:, 1].max()
    measured_height = y_max - y_min
    scale_factor = target_height_cm / measured_height
    
    # Apply scaling to vertices and joints
    vertices_scaled = vertices_np * scale_factor
    joints_scaled = joints * scale_factor
    
    print(f"   - Original height: {measured_height:.2f} units")
    print(f"   - Target height: {target_height_cm:.2f} cm")
    print(f"   - Scale factor: {scale_factor:.4f}")
    print(f"   ✅ Scaling applied\n")
    
    # Step 5: Calculate measurements using shared calculator
    print("6️⃣ Calculating measurements...")
    calculator = MeasurementCalculator(
        vertices=vertices_scaled,
        faces=faces,
        joints=joints_scaled,
        joint_names=joint_names,
        scale_factor=1.0  # Already scaled to cm
    )
    
    measurements = calculator.calculate_all_measurements()
    print("   ✅ Measurements calculated\n")
    
    # Step 6: Display results
    labels = {
        'A': 'Head Circumference',
        'B': 'Neck Circumference',
        'C': 'Shoulder to Crotch Height',
        'D': 'Chest Circumference',
        'E': 'Waist Circumference',
        'F': 'Hip Circumference',
        'G': 'Wrist Right Circumference',
        'H': 'Bicep Right Circumference',
        'I': 'Forearm Right Circumference',
        'J': 'Arm Right Length',
        'K': 'Inside Leg Height',
        'L': 'Thigh Left Circumference',
        'M': 'Calf Left Circumference',
        'N': 'Ankle Left Circumference',
        'O': 'Shoulder Breadth',
        'P': 'Height',
    }
    
    print(f"{'='*60}")
    print(f"BODY MEASUREMENTS (Scaled to {target_height_cm} cm)")
    print(f"{'='*60}\n")
    
    for key in sorted(labels.keys()):
        label = labels[key]
        value = measurements.get(key, 0.0)
        print(f"{key} - {label:.<40} {value:>6.2f} cm")
    
    print(f"\n{'='*60}\n")
    
    return measurements


def save_measurements_to_csv(measurements, output_path, image_name="test3.jpeg", height=173.0):
    """
    Save measurements to CSV file
    
    Args:
        measurements: dict of measurement values
        output_path: path to save CSV file
        image_name: name of the input image
        height: target height in cm
    """
    import csv
    from datetime import datetime
    
    labels = {
        'A': 'Head Circumference',
        'B': 'Neck Circumference',
        'C': 'Shoulder to Crotch Height',
        'D': 'Chest Circumference',
        'E': 'Waist Circumference',
        'F': 'Hip Circumference',
        'G': 'Wrist Right Circumference',
        'H': 'Bicep Right Circumference',
        'I': 'Forearm Right Circumference',
        'J': 'Arm Right Length',
        'K': 'Inside Leg Height',
        'L': 'Thigh Left Circumference',
        'M': 'Calf Left Circumference',
        'N': 'Ankle Left Circumference',
        'O': 'Shoulder Breadth',
        'P': 'Height',
    }
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header with metadata
        writer.writerow(['Body Measurements Report'])
        writer.writerow(['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow(['Image:', image_name])
        writer.writerow(['Target Height:', f'{height} cm'])
        writer.writerow([])  # Empty row
        
        # Write measurement headers
        writer.writerow(['Code', 'Measurement', 'Value (cm)'])
        
        # Write measurements in order
        for key in sorted(labels.keys()):
            label = labels[key]
            value = measurements.get(key, 0.0)
            writer.writerow([key, label, f'{value:.2f}'])
    
    print(f"📄 Measurements saved to: {output_path}\n")


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Calculate body measurements from Sam3D output')
    parser.add_argument('--sam3d_output', type=str, required=True,
                       help='Path to Sam3D output pickle file')
    parser.add_argument('--height', type=float, default=175.0,
                       help='Target height in cm (default: 175.0)')
    parser.add_argument('--csv_output', type=str, default=None,
                       help='Path to save CSV file (default: output/measurements.csv)')
    
    args = parser.parse_args()
    
    measurements = calculate_measurements_from_sam3d(
        sam3d_output_path=args.sam3d_output,
        target_height_cm=args.height
    )
    
    # Save to CSV
    if args.csv_output is None:
        # Default: save to output directory with same name as pkl file
        pkl_path = Path(args.sam3d_output)
        csv_output = pkl_path.parent / "measurements.csv"
    else:
        csv_output = Path(args.csv_output)
    
    # Extract image name from path if possible
    image_name = "unknown"
    try:
        pkl_path = Path(args.sam3d_output)
        if pkl_path.stem.endswith("_output"):
            image_name = pkl_path.stem.replace("_sam3d_output", "") + ".jpeg"
    except:
        pass
    
    save_measurements_to_csv(
        measurements=measurements,
        output_path=str(csv_output),
        image_name=image_name,
        height=args.height
    )

