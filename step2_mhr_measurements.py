#!/usr/bin/env python3
"""
Step 2: MHR Measurement from Sam3D Output
Loads Sam3D predictions and calculates body measurements using MHR
"""
import sys
sys.path.insert(0, '/home/pj/Desktop/MHR')

import pickle
import torch
import numpy as np
import trimesh
from pathlib import Path

# MHR imports
from mhr.mhr import MHR

def get_circumference(mesh, center, normal):
    """Calculate circumference by slicing mesh"""
    slice_3d = mesh.section(plane_origin=center, plane_normal=normal)
    
    if slice_3d is None or slice_3d.is_empty:
        return 0.0
    
    slice_2d, _ = slice_3d.to_planar()
    polygons = slice_2d.polygons_full
    if not polygons:
        return 0.0
    
    return max(poly.length for poly in polygons)

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
    
    # Step 4: Calculate measurements
    print("5️⃣ Calculating measurements...")
    
    def get_j(name):
        return joints[joint_names.index(name)]
    
    # Calculate height
    y_min = vertices_np[:, 1].min()
    y_max = vertices_np[:, 1].max()
    measured_height = y_max - y_min
    scale = target_height_cm / measured_height
    
    measurements = {}
    measurements['P'] = target_height_cm
    
    # Create mesh for circumferences
    mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces, process=False)
    y_axis = [0, 1, 0]
    x_axis = [1, 0, 0]
    
    # Lengths
    y_shoulder = get_j('c_spine3')[1]
    y_crotch = get_j('root')[1]
    measurements['C'] = (y_shoulder - y_crotch) * scale
    measurements['K'] = (y_crotch - y_min) * scale
    
    # Arm length
    p_shoulder = get_j('r_uparm')
    p_elbow = get_j('r_lowarm')
    p_wrist = get_j('r_wrist')
    arm_len = np.linalg.norm(p_shoulder - p_elbow) + np.linalg.norm(p_elbow - p_wrist)
    measurements['J'] = arm_len * scale
    
    # Shoulder breadth
    p_l_shoulder = get_j('l_uparm')
    p_r_shoulder = get_j('r_uparm')
    measurements['O'] = np.linalg.norm(p_l_shoulder - p_r_shoulder) * scale
    
    # Circumferences
    measurements['A'] = get_circumference(mesh, get_j('r_eye'), y_axis) * scale
    p_neck_mid = 0.5 * (get_j('c_neck') + get_j('c_head'))
    measurements['B'] = get_circumference(mesh, p_neck_mid, y_axis) * scale
    measurements['D'] = get_circumference(mesh, get_j('c_spine2'), y_axis) * scale
    measurements['E'] = get_circumference(mesh, get_j('c_spine0'), y_axis) * scale
    measurements['F'] = get_circumference(mesh, get_j('root'), y_axis) * scale
    
    # Leg measurements
    p_l_hip = get_j('l_upleg')
    p_l_knee = get_j('l_lowleg')
    p_l_ankle = get_j('l_foot')
    mid_thigh = (p_l_hip + p_l_knee) / 2
    mid_calf = (p_l_knee + p_l_ankle) / 2
    measurements['L'] = get_circumference(mesh, mid_thigh, y_axis) * scale
    measurements['M'] = get_circumference(mesh, mid_calf, y_axis) * scale
    measurements['N'] = get_circumference(mesh, p_l_ankle + [0, 2, 0], y_axis) * scale
    
    # Arm circumferences
    measurements['G'] = get_circumference(mesh, get_j('r_wrist'), x_axis) * scale
    mid_bicep = (p_shoulder + p_elbow) / 2
    measurements['H'] = get_circumference(mesh, mid_bicep, x_axis) * scale
    mid_forearm = (p_elbow + p_wrist) / 2
    measurements['I'] = get_circumference(mesh, mid_forearm, x_axis) * scale
    
    print("   ✅ Measurements calculated\n")
    
    # Step 5: Display results
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
        'P': 'Height'
    }
    
    print(f"{'='*60}")
    print(f"BODY MEASUREMENTS (Scaled to {target_height_cm} cm)")
    print(f"{'='*60}\n")
    
    for key in sorted(measurements.keys()):
        print(f"{key} - {labels[key]:.<40} {measurements[key]:>6.2f} cm")
    
    print(f"\n{'='*60}\n")
    
    return measurements

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate measurements from Sam3D output")
    parser.add_argument("--sam3d_output", default="/home/pj/Desktop/MHR/output/sam3d_output.pkl", 
                        help="Path to Sam3D output pickle file")
    parser.add_argument("--height", type=float, default=175, help="Target height in cm")
    
    args = parser.parse_args()
    
    calculate_measurements_from_sam3d(args.sam3d_output, args.height)
