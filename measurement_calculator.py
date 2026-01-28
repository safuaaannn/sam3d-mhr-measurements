#!/usr/bin/env python3
"""
Shared Measurement Calculator
Provides consistent measurement logic for both interactive and command-line tools
"""
import numpy as np
import pyvista as pv


class MeasurementCalculator:
    """
    Unified measurement calculator using PyVista for consistent results
    across interactive and command-line tools.
    """
    
    def __init__(self, vertices, faces, joints, joint_names, scale_factor=1.0):
        """
        Initialize calculator with mesh data
        
        Args:
            vertices: numpy array of mesh vertices (N, 3)
            faces: numpy array of mesh faces (M, 3)
            joints: numpy array of joint positions (J, 3)
            joint_names: list of joint names
            scale_factor: scaling factor to convert to cm (default: 1.0 if already in cm)
        """
        self.vertices = vertices
        self.faces = faces
        self.joints = joints
        self.joint_names = joint_names
        self.scale_factor = scale_factor
        
        # Create PyVista mesh
        n_faces = self.faces.shape[0]
        pad = 3 * np.ones((n_faces, 1), dtype=np.int64)
        pv_faces = np.hstack((pad, self.faces.astype(np.int64))).flatten()
        self.mesh = pv.PolyData(self.vertices, pv_faces)
    
    def get_joint(self, name):
        """Get joint position by name"""
        try:
            return self.joints[self.joint_names.index(name)]
        except:
            return np.mean(self.vertices, axis=0)
    
    def compute_circumference(self, center, normal, exclude_arms=True):
        """
        Compute circumference by slicing mesh.
        For torso measurements (exclude_arms=True), extracts only the largest contour
        to avoid including arms in A-Pose.
        
        Args:
            center: 3D point for slice center
            normal: 3D vector for slice plane normal
            exclude_arms: If True, extract only largest connected component (torso)
        
        Returns:
            circumference in cm
        """
        try:
            if np.linalg.norm(normal) < 1e-6:
                return 0.0
            normal = np.array(normal) / np.linalg.norm(normal)
            
            # Create the slice
            slice_mesh = self.mesh.slice(normal=normal, origin=center)
            
            if slice_mesh.n_points < 3:
                return 0.0
            
            # Helper to convert to PolyData
            def _as_polydata(ds):
                try:
                    if isinstance(ds, pv.PolyData):
                        return ds
                    return ds.extract_geometry()
                except Exception:
                    return pv.PolyData(ds)
            
            # Filter connected components for torso measurements
            if exclude_arms:
                try:
                    torso_contour = slice_mesh.connectivity(extraction_mode="largest")
                    torso_contour = _as_polydata(torso_contour)
                    if torso_contour.n_points < 3:
                        torso_contour = slice_mesh
                except Exception:
                    torso_contour = slice_mesh
            else:
                torso_contour = slice_mesh
            
            # Calculate perimeter
            if not hasattr(torso_contour, 'lines') or len(torso_contour.lines) == 0:
                return 0.0
            
            lines = torso_contour.lines
            points = torso_contour.points
            
            perimeter = 0.0
            i = 0
            while i < len(lines):
                n_pts = lines[i]
                i += 1
                segment_indices = lines[i:i+n_pts]
                for j in range(len(segment_indices) - 1):
                    p1 = points[segment_indices[j]]
                    p2 = points[segment_indices[j+1]]
                    dist = np.linalg.norm(np.array(p1) - np.array(p2))
                    perimeter += dist
                i += n_pts
            
            return perimeter
            
        except Exception as e:
            print(f"Circumference Error: {e}")
            return 0.0
    
    def compute_length(self, start_joint, end_joint):
        """
        Compute 3D Euclidean distance between two joints
        
        Args:
            start_joint: Starting joint name or special marker ('min_y', 'max_y')
            end_joint: Ending joint name or special marker ('min_y', 'max_y')
        
        Returns:
            distance in cm
        """
        if start_joint == 'min_y':
            start_pt = np.array([self.vertices[:, 0].mean(), self.vertices[:, 1].min(), self.vertices[:, 2].mean()])
        elif start_joint == 'max_y':
            start_pt = np.array([self.vertices[:, 0].mean(), self.vertices[:, 1].max(), self.vertices[:, 2].mean()])
        else:
            start_pt = self.get_joint(start_joint)
        
        if end_joint == 'min_y':
            end_pt = np.array([self.vertices[:, 0].mean(), self.vertices[:, 1].min(), self.vertices[:, 2].mean()])
        elif end_joint == 'max_y':
            end_pt = np.array([self.vertices[:, 0].mean(), self.vertices[:, 1].max(), self.vertices[:, 2].mean()])
        else:
            end_pt = self.get_joint(end_joint)
        
        return np.linalg.norm(end_pt - start_pt)
    
    def calculate_all_measurements(self):
        """
        Calculate all standard body measurements with corrected anatomical positions
        
        Returns:
            dict of measurements in cm
        """
        measurements = {}
        
        # Height (P)
        y_min = self.vertices[:, 1].min()
        y_max = self.vertices[:, 1].max()
        measurements['P'] = y_max - y_min
        
        # Head Circumference (A)
        measurements['A'] = self.compute_circumference(
            self.get_joint('r_eye'), 
            [0, 1, 0], 
            exclude_arms=False
        )
        
        # Neck Circumference (B) - positioned higher to avoid shoulders
        # At 70% from neck to head to ensure isolation from trapezius in A-Pose
        neck_j = self.get_joint('c_neck')
        head_j = self.get_joint('c_head')
        p_neck_high = neck_j + 0.7 * (head_j - neck_j)  # 70% toward head
        measurements['B'] = self.compute_circumference(
            p_neck_high, 
            [0, 1, 0], 
            exclude_arms=True  # Exclude shoulders/trapezius in A-Pose
        )
        
        # Chest Circumference (D) - with +11.4 cm offset
        measurements['D'] = self.compute_circumference(
            self.get_joint('c_spine2') + [0, 11.4, 0], 
            [0, 1, 0], 
            exclude_arms=True
        )
        
        # Waist Circumference (E) - with +15.5 cm offset
        measurements['E'] = self.compute_circumference(
            self.get_joint('c_spine0') + [0, 15.5, 0], 
            [0, 1, 0], 
            exclude_arms=True
        )
        
        # Hip Circumference (F)
        measurements['F'] = self.compute_circumference(
            self.get_joint('root'), 
            [0, 1, 0], 
            exclude_arms=True
        )
        
        # Thigh Circumference (L) - at upper thigh with +6.99 cm offset from midpoint
        p_l_hip = self.get_joint('l_upleg')
        p_l_knee = self.get_joint('l_lowleg')
        mid_thigh = (p_l_hip + p_l_knee) / 2 + np.array([0, 6.99, 0])  # Apply offset
        measurements['L'] = self.compute_circumference(
            mid_thigh, 
            [0, 1, 0], 
            exclude_arms=False
        )
        
        # Calf Circumference (M) - at maximum calf with +11.0 cm offset from midpoint
        p_l_ankle = self.get_joint('l_foot')
        mid_calf = (p_l_knee + p_l_ankle) / 2 + np.array([0, 11.0, 0])  # Apply offset
        measurements['M'] = self.compute_circumference(
            mid_calf, 
            [0, 1, 0], 
            exclude_arms=False
        )
        
        # Ankle Circumference (N)
        measurements['N'] = self.compute_circumference(
            p_l_ankle + [0, 2.0, 0], 
            [0, 1, 0], 
            exclude_arms=False
        )
        
        # Wrist Circumference (G)
        measurements['G'] = self.compute_circumference(
            self.get_joint('r_wrist'), 
            [1, 0, 0], 
            exclude_arms=False
        )
        
        # Bicep Circumference (H) - at mid-upper-arm
        p_shoulder = self.get_joint('r_uparm')
        p_elbow = self.get_joint('r_lowarm')
        mid_bicep = (p_shoulder + p_elbow) / 2
        measurements['H'] = self.compute_circumference(
            mid_bicep, 
            [1, 0, 0], 
            exclude_arms=False
        )
        
        # Forearm Circumference (I) - at mid-forearm
        p_wrist = self.get_joint('r_wrist')
        mid_forearm = (p_elbow + p_wrist) / 2
        measurements['I'] = self.compute_circumference(
            mid_forearm, 
            [1, 0, 0], 
            exclude_arms=False
        )
        
        # Shoulder-to-Crotch Height (C)
        y_shoulder = self.get_joint('c_spine3')[1]
        y_crotch = self.get_joint('root')[1]
        measurements['C'] = abs(y_shoulder - y_crotch)
        
        # Inside Leg Height (K)
        measurements['K'] = abs(y_crotch - y_min)
        
        # Arm Length (J)
        arm_len = (np.linalg.norm(p_shoulder - p_elbow) + 
                   np.linalg.norm(p_elbow - p_wrist))
        measurements['J'] = arm_len
        
        # Shoulder Breadth (O)
        p_l_shoulder = self.get_joint('l_uparm')
        p_r_shoulder = self.get_joint('r_uparm')
        measurements['O'] = np.linalg.norm(p_l_shoulder - p_r_shoulder)
        
        return measurements
