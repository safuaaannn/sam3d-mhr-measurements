#!/usr/bin/env python3
"""
Visualize measurement locations on the 3D mesh
Shows joints and slice planes used for measurements
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
import trimesh
from scipy.spatial.transform import Rotation

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
    
    # Define ALL 16 measurements with their colors and positions
    # Circumferences (horizontal slices - Y-axis normal)
    y_axis = np.array([0, 1, 0])
    x_axis = np.array([1, 0, 0])
    threshold = 0.8  # cm threshold for coloring vertices
    
    # Get joint positions for calculations
    p_eye = get_j('r_eye')
    p_neck = get_j('c_neck')
    p_head = get_j('c_head')
    p_spine2 = get_j('c_spine2')
    p_spine0 = get_j('c_spine0')
    p_root = get_j('root')
    p_l_upleg = get_j('l_upleg')
    p_l_lowleg = get_j('l_lowleg')
    p_l_foot = get_j('l_foot')
    p_r_uparm = get_j('r_uparm')
    p_r_lowarm = get_j('r_lowarm')
    p_r_wrist = get_j('r_wrist')
    
    # Calculate midpoints
    p_neck_mid = 0.5 * (p_neck + p_head)
    mid_thigh = (p_l_upleg + p_l_lowleg) / 2
    mid_calf = (p_l_lowleg + p_l_foot) / 2
    mid_bicep = (p_r_uparm + p_r_lowarm) / 2
    mid_forearm = (p_r_lowarm + p_r_wrist) / 2
    p_ankle_offset = p_l_foot + np.array([0, 2, 0])
    
    # Circumference measurements (horizontal slices)
    circumference_measurements = {
        'A': (p_eye, y_axis, [255, 0, 0, 255]),      # Red - Head
        'B': (p_neck_mid, y_axis, [255, 165, 0, 255]),  # Orange - Neck
        'D': (p_spine2, y_axis, [255, 255, 0, 255]),    # Yellow - Chest
        'E': (p_spine0, y_axis, [0, 255, 0, 255]),      # Green - Waist
        'F': (p_root, y_axis, [0, 255, 255, 255]),      # Cyan - Hip
        'L': (mid_thigh, y_axis, [0, 0, 255, 255]),     # Blue - Thigh
        'M': (mid_calf, y_axis, [128, 0, 255, 255]),    # Purple - Calf
        'N': (p_ankle_offset, y_axis, [255, 20, 147, 255]),  # Deep Pink - Ankle
    }
    
    # Vertical circumference measurements (X-axis normal for arms)
    vertical_circumference_measurements = {
        'G': (p_r_wrist, x_axis, [165, 42, 42, 255]),    # Brown - Wrist
        'H': (mid_bicep, x_axis, [255, 140, 0, 255]),    # Dark Orange - Bicep
        'I': (mid_forearm, x_axis, [255, 192, 203, 255]), # Pink - Forearm
    }
    
    # Color vertices for horizontal circumferences
    for code, (center, normal, color) in circumference_measurements.items():
        distances = np.abs(np.dot(vertices_np - center, normal))
        near_plane = distances < threshold
        vertex_colors[near_plane] = color
    
    # Color vertices for vertical circumferences (arms)
    for code, (center, normal, color) in vertical_circumference_measurements.items():
        distances = np.abs(np.dot(vertices_np - center, normal))
        near_plane = distances < threshold
        vertex_colors[near_plane] = color
    
    # Mark key joints with spheres and add lines for length measurements
    joint_spheres = []
    length_lines = []
    
    # Joint spheres for all measurement points
    joint_markers = {
        # Circumferences
        'r_eye': ([255, 0, 0, 255], 'A'),        # Red - Head
        'c_neck': ([255, 165, 0, 255], 'B'),     # Orange - Neck
        'c_spine2': ([255, 255, 0, 255], 'D'),   # Yellow - Chest
        'c_spine0': ([0, 255, 0, 255], 'E'),     # Green - Waist
        'root': ([0, 255, 255, 255], 'F'),       # Cyan - Hip
        'l_upleg': ([0, 0, 255, 255], 'L'),      # Blue - Thigh
        'l_lowleg': ([128, 0, 255, 255], 'M'),   # Purple - Calf
        'l_foot': ([255, 20, 147, 255], 'N'),    # Deep Pink - Ankle
        # Arm circumferences
        'r_wrist': ([165, 42, 42, 255], 'G'),    # Brown - Wrist
        'r_uparm': ([255, 140, 0, 255], 'H'),    # Dark Orange - Bicep
        'r_lowarm': ([255, 192, 203, 255], 'I'), # Pink - Forearm
        # Length measurements
        'c_spine3': ([255, 255, 255, 255], 'C'), # White - Shoulder (for C)
        'l_uparm': ([200, 200, 200, 255], 'O'),  # Light Gray - Left Shoulder (for O)
    }
    
    for joint_name, (color, code) in joint_markers.items():
        pos = get_j(joint_name)
        sphere = trimesh.primitives.Sphere(radius=1.5, center=pos)
        sphere.visual.vertex_colors = color
        joint_spheres.append(sphere)
    
    # Helper function to create a line cylinder between two points
    def create_line_cylinder(p1, p2, radius=0.3, color=[255, 255, 255, 255]):
        """Create a cylinder connecting two points"""
        direction = p2 - p1
        height = np.linalg.norm(direction)
        if height < 0.001:
            return None
        
        # Create cylinder along Z-axis
        cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=8)
        cylinder.visual.vertex_colors = color
        
        # Calculate rotation to align with direction
        z_axis = np.array([0, 0, 1])
        direction_normalized = direction / height
        
        # Use scipy's Rotation to align cylinder
        rotation_axis = np.cross(z_axis, direction_normalized)
        if np.linalg.norm(rotation_axis) < 1e-6:
            # Already aligned or opposite
            if np.dot(z_axis, direction_normalized) < 0:
                # Opposite direction
                rot = Rotation.from_rotvec(np.pi * np.array([1, 0, 0]))
            else:
                rot = Rotation.identity()
        else:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(z_axis, direction_normalized), -1, 1))
            rot = Rotation.from_rotvec(angle * rotation_axis)
        
        # Create 4x4 transform matrix
        midpoint = (p1 + p2) / 2.0
        transform = np.eye(4)
        transform[:3, :3] = rot.as_matrix()
        transform[:3, 3] = midpoint
        
        cylinder.apply_transform(transform)
        return cylinder
    
    # Length measurement C: Shoulder to Crotch (vertical line)
    p_spine3 = get_j('c_spine3')
    line_c = create_line_cylinder(p_spine3, p_root, 0.3, [255, 255, 255, 255])  # White
    if line_c:
        length_lines.append(line_c)
    
    # Length measurement J: Arm Length (line segments)
    p_r_uparm = get_j('r_uparm')
    p_r_lowarm = get_j('r_lowarm')
    p_r_wrist = get_j('r_wrist')
    # Upper arm segment
    line_j1 = create_line_cylinder(p_r_uparm, p_r_lowarm, 0.3, [255, 165, 0, 255])  # Orange
    if line_j1:
        length_lines.append(line_j1)
    # Forearm segment
    line_j2 = create_line_cylinder(p_r_lowarm, p_r_wrist, 0.3, [255, 165, 0, 255])  # Orange
    if line_j2:
        length_lines.append(line_j2)
    
    # Length measurement K: Inside Leg Height (vertical line)
    y_min = vertices_np[:, 1].min()
    p_foot_bottom = np.array([p_root[0], y_min, p_root[2]])
    line_k = create_line_cylinder(p_root, p_foot_bottom, 0.3, [255, 0, 255, 255])  # Magenta
    if line_k:
        length_lines.append(line_k)
    
    # Length measurement O: Shoulder Breadth (horizontal line)
    p_l_uparm = get_j('l_uparm')
    line_o = create_line_cylinder(p_l_uparm, p_r_uparm, 0.3, [200, 200, 200, 255])  # Light Gray
    if line_o:
        length_lines.append(line_o)
    
    # Save colored mesh
    mesh.visual.vertex_colors = vertex_colors.astype(np.uint8)
    
    # Combine mesh with joint spheres and length measurement lines
    scene = trimesh.Scene([mesh] + joint_spheres + length_lines)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(output_path))
    
    return str(output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize measurement locations")
    parser.add_argument("--sam3d_output", default="./output/sam3d_output.pkl",
                        help="Path to Sam3D output pickle file")
    parser.add_argument("--pkl", default=None,
                        help="Alternative argument name for --sam3d_output (for consistency)")
    parser.add_argument("--output", default="./output/measurement_visualization.ply",
                        help="Output path for colored mesh")
    parser.add_argument("--out", default=None,
                        help="Alternative argument name for --output (for consistency)")
    
    args = parser.parse_args()
    
    # Handle alternative argument names for consistency
    sam3d_output = args.pkl if args.pkl is not None else args.sam3d_output
    output_path = args.out if args.out is not None else args.output
    
    visualize_measurements(sam3d_output, output_path)
