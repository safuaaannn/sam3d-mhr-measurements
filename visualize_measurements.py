#!/usr/bin/env python3
"""
Visualize measurement locations on the 3D mesh
Shows joints and slice planes used for measurements
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

def visualize_measurements(sam3d_output_path, output_path="./output/measurement_visualization.ply"):
    """
    Create a colored mesh showing measurement locations
    """
    # Load Sam3D output
    with open(sam3d_output_path, 'rb') as f:
        sam3d_data = pickle.load(f)
    
    # Extract MHR parameters
    identity_coeffs = torch.from_numpy(sam3d_data['shape_params']).unsqueeze(0).float()
    model_parameters = torch.from_numpy(sam3d_data['mhr_model_params']).unsqueeze(0).float()
    face_expr_coeffs = torch.from_numpy(sam3d_data['expr_params']).unsqueeze(0).float()
    
    # Load MHR and generate mesh
    device = torch.device("cpu")
    mhr = MHR.from_files(device=device, lod=1)
    
    with torch.no_grad():
        vertices, skel_state = mhr(identity_coeffs, model_parameters, face_expr_coeffs)
    
    vertices_np = vertices[0].numpy()
    joints = skel_state[0, :, :3].numpy()
    faces = mhr.character.mesh.faces
    joint_names = mhr.character.skeleton.joint_names
    
    def get_j(name):
        return joints[joint_names.index(name)]
    
    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces, process=False)
    
    # Initialize vertex colors (gray by default)
    vertex_colors = np.ones((len(vertices_np), 4)) * 128
    vertex_colors[:, 3] = 255  # Alpha
    
    # Define measurement planes and their colors with CORRECT anatomical positions
    measurement_planes = {
        'Head (A)': (get_j('c_head'), [255, 0, 0, 255]),  # Red - at head top
        'Neck (B)': (get_j('c_neck'), [255, 165, 0, 255]),  # Orange - at neck base
        'Chest (D)': (get_j('c_spine2'), [255, 255, 0, 255]),  # Yellow - at chest/bust
        'Waist (E)': (get_j('c_spine0'), [0, 255, 0, 255]),  # Green - at natural waist
        'Hip (F)': (get_j('root'), [0, 255, 255, 255]),  # Cyan - at hip level
        'Thigh (L)': ((get_j('l_upleg') + get_j('l_lowleg')) / 2, [0, 0, 255, 255]),  # Blue - mid thigh
        'Calf (M)': ((get_j('l_lowleg') + get_j('l_foot')) / 2, [128, 0, 255, 255]),  # Purple - mid calf
    }
    
    # Color vertices near each measurement plane
    y_axis = np.array([0, 1, 0])
    threshold = 0.8  # Reduced from 2.0 to 0.8 cm for tighter, more accurate bands
    
    for name, (center, color) in measurement_planes.items():
        # Find vertices close to the plane
        distances = np.abs(np.dot(vertices_np - center, y_axis))
        near_plane = distances < threshold
        vertex_colors[near_plane] = color
    
    # Mark key joints with spheres
    joint_spheres = []
    key_joints = {
        'r_eye': [255, 0, 0, 255],  # Head - Red
        'c_neck': [255, 165, 0, 255],  # Neck - Orange
        'c_spine2': [255, 255, 0, 255],  # Chest - Yellow
        'c_spine0': [0, 255, 0, 255],  # Waist - Green
        'root': [0, 255, 255, 255],  # Hip - Cyan
        'r_uparm': [255, 0, 255, 255],  # Shoulder - Magenta
        'r_lowarm': [255, 192, 203, 255],  # Elbow - Pink
        'r_wrist': [165, 42, 42, 255],  # Wrist - Brown
    }
    
    for joint_name, color in key_joints.items():
        pos = get_j(joint_name)
        sphere = trimesh.primitives.Sphere(radius=1.5, center=pos)
        sphere.visual.vertex_colors = color
        joint_spheres.append(sphere)
    
    # Save colored mesh
    mesh.visual.vertex_colors = vertex_colors.astype(np.uint8)
    
    # Combine mesh with joint spheres
    scene = trimesh.Scene([mesh] + joint_spheres)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(output_path))
    
    return str(output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize measurement locations")
    parser.add_argument("--sam3d_output", default="/home/pj/Desktop/MHR/output/sam3d_output.pkl",
                        help="Path to Sam3D output pickle file")
    parser.add_argument("--output", default="/home/pj/Desktop/MHR/output/measurement_visualization.ply",
                        help="Output path for colored mesh")
    
    args = parser.parse_args()
    
    visualize_measurements(args.sam3d_output, args.output)
