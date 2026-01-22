#!/usr/bin/env python3
"""
Interactive 3D Body Measurement Tool
Allows adjustment of measurement rings and real-time visualization
"""
import sys
from pathlib import Path

# Add project root to path (relative to this script's directory)
SCRIPT_DIR = Path(__file__).parent.absolute()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pyvista as pv
import numpy as np
import pickle
import torch
import argparse
import trimesh
import networkx as nx
from scipy.interpolate import interp1d
from mhr.mhr import MHR

class InteractiveBodyMeasurer:
    def __init__(self, pkl_path, target_height_cm=173.0):
        self.pkl_path = pkl_path
        self.target_height_cm = target_height_cm
        
        # 1. Load Data
        with open(self.pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        print("Loading MHR Model...")
        device = torch.device("cpu")
        self.mhr = MHR.from_files(device=device, lod=1)
        
        # 2. Generate Mesh
        print("Generating Mesh...")
        identity_coeffs = torch.from_numpy(data['shape_params']).unsqueeze(0).float()
        model_parameters = torch.from_numpy(data['mhr_model_params']).unsqueeze(0).float()
        face_expr_coeffs = torch.from_numpy(data['expr_params']).unsqueeze(0).float()
        
        with torch.no_grad():
            vertices, skel_state = self.mhr(identity_coeffs, model_parameters, face_expr_coeffs)
            
        self.vertices = vertices[0].numpy()
        self.joints = skel_state[0, :, :3].numpy()
        self.faces = self.mhr.character.mesh.faces
        self.joint_names = self.mhr.character.skeleton.joint_names
        
        # 3. Robust Scaling
        y_min = self.vertices[:, 1].min()
        y_max = self.vertices[:, 1].max()
        measured_height = y_max - y_min
        self.scale_factor = self.target_height_cm / measured_height
        
        self.vertices = self.vertices * self.scale_factor
        self.joints = self.joints * self.scale_factor
        
        print(f"Original Height: {measured_height:.4f}")
        print(f"Scale Factor: {self.scale_factor:.4f}")
        
        # 4. Create PyVista Mesh
        n_faces = self.faces.shape[0]
        pad = 3 * np.ones((n_faces, 1), dtype=np.int64)
        pv_faces = np.hstack((pad, self.faces.astype(np.int64))).flatten()
        
        self.mesh = pv.PolyData(self.vertices, pv_faces)
        
        # 4b. Create Trimesh object for accurate surface projection
        # This is needed for proper point-to-surface projection in picking
        self.trimesh_mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)
        
        # 5. Define ALL 16 Measurements
        self.measurements = {
            # Circumference measurements
            'Head (A)':    {'joint': 'r_eye',     'normal': [0, 1, 0], 'offset': 0.0, 'color': 'red', 'exclude_arms': False, 'type': 'circumference'},
            # Neck (B): Use neck-to-head vector for proper circular slice at mid-neck (light blue position)
            'Neck (B)':    {'joint': 'c_neck',   'normal': [0, 1, 0], 'neck_aligned': True, 'neck_height_ratio': 0.45, 'offset': 0.0, 'color': 'cyan', 'exclude_arms': False, 'type': 'circumference'},
            # Torso circumferences - use consistent torso vector (pelvis to neck) for perpendicular normal
            'Chest (D)':   {'joint': 'c_spine2', 'normal': [0, 1, 0], 'torso_aligned': True, 'offset': 0.0, 'color': 'yellow', 'exclude_arms': True, 'type': 'circumference'},
            'Waist (E)':   {'joint': 'c_spine0', 'normal': [0, 1, 0], 'torso_aligned': True, 'offset': 0.0, 'color': 'green', 'exclude_arms': True, 'type': 'circumference'},
            'Hip (F)':     {'joint': 'root',     'normal': [0, 1, 0], 'torso_aligned': True, 'offset': 0.0, 'color': 'cyan', 'exclude_arms': True, 'type': 'circumference'},
            'Thigh (L)':   {'joint': 'l_upleg',  'normal': [0, 1, 0], 'offset': -15.0, 'color': 'blue', 'exclude_arms': False, 'type': 'circumference', 'midpoint': ['l_upleg', 'l_lowleg']},
            'Calf (M)':    {'joint': 'l_lowleg', 'normal': [0, 1, 0], 'offset': -15.0, 'color': 'purple', 'exclude_arms': False, 'type': 'circumference', 'midpoint': ['l_lowleg', 'l_foot']},
            'Ankle (N)':   {'joint': 'l_foot',   'normal': [0, 1, 0], 'offset': 2.0, 'color': 'magenta', 'exclude_arms': False, 'type': 'circumference'},
            # Arm circumferences - use bone direction for proper circular slices
            'Wrist (G)':   {'joint': 'r_wrist',  'normal': [1, 0, 0], 'normal_from': ['r_lowarm', 'r_wrist'], 'offset': 0.0, 'color': 'brown', 'exclude_arms': False, 'type': 'circumference'},
            'Bicep (H)':   {'joint': 'r_uparm',  'normal': [1, 0, 0], 'normal_from': ['r_uparm', 'r_lowarm'], 'offset': 0.0, 'color': 'darkorange', 'exclude_arms': False, 'type': 'circumference', 'midpoint': ['r_uparm', 'r_lowarm']},
            'Forearm (I)': {'joint': 'r_lowarm', 'normal': [1, 0, 0], 'normal_from': ['r_lowarm', 'r_wrist'], 'offset': 0.0, 'color': 'pink', 'exclude_arms': False, 'type': 'circumference', 'midpoint': ['r_lowarm', 'r_wrist']},
        }
        
        # Length measurements
        self.length_measurements = {
            'Shoulder-Crotch (C)': {'start': 'c_spine3', 'end': 'root', 'color': 'white', 'type': 'length', 'geodesic': True},
            'Arm Length (J)': {'start': 'r_uparm', 'end': 'r_wrist', 'color': 'orange', 'type': 'length', 'segments': ['r_uparm', 'r_lowarm', 'r_wrist']},
            'Inside Leg (K)': {'start': 'root', 'end': 'min_y', 'color': 'magenta', 'type': 'length'},
            'Shoulder Width (O)': {'start': 'l_uparm', 'end': 'r_uparm', 'color': 'pink', 'type': 'length', 'geodesic': True},
            'Height (P)': {'start': 'max_y', 'end': 'min_y', 'color': 'gray', 'type': 'length'},
        }
        
        self.selected_points = []
        self.custom_measurements = []
        self.picking_enabled = False
        # Neck geodesic (B) override: user-defined landmarks on surface
        # Order recommended: back (nape) -> right -> front (suprasternal) -> left
        self._active_pick_mode = None  # None | "custom" | "neck" | "shoulder"
        self.neck_landmark_vertex_ids = []
        self.neck_landmark_points = []  # list[np.ndarray], world/surface points for neck picking
        self.neck_geodesic_path = None  # pv.PolyData polyline
        # Shoulder width geodesic: user picks 7 points for more accurate path
        self.shoulder_landmark_points = []  # list[np.ndarray], 7 points for shoulder width
        self.shoulder_landmark_vertex_ids = []  # vertex indices for geodesic calculation
        self.shoulder_geodesic_path = None  # pv.PolyData polyline through the 7 points
        self.shoulder_path_length = None  # Store the calculated path length (in cm) to avoid recalculation errors
        
        # Default vertex indices for shoulder breadth (from calibration)
        # These are used when no manual picking has been done
        # Order: Left → Left Mid-Left → Left Mid → Nape → Right Mid → Right Mid-Right → Right
        self.default_shoulder_vertex_indices = [6732, 6587, 6242, 5764, 7170, 7640, 7956]
        
        # Cache for default shoulder width calculation (to avoid recalculating on every update)
        self._default_shoulder_width_cached = None
        self._default_shoulder_path_cached = None
        
        self.plotter = pv.Plotter(title="Sam3D Interactive Measurements")
        
        # Enable pickpoint attribute for PyVista compatibility (required for cell picking)
        try:
            pv.set_new_attribute(self.plotter, 'pickpoint', None)
        except Exception:
            pass  # Attribute might already exist or not needed
        
        self.setup_ui()

    def get_joint(self, name):
        try:
            return self.joints[self.joint_names.index(name)]
        except:
            return np.mean(self.vertices, axis=0)

    def _compute_neck_frame(self):
        """
        Build a stable local coordinate frame for the neck:
        - axis: from neck -> head (up-ish)
        - lr: left->right shoulder direction
        - fwd: forward direction estimated from lr x axis
        Returns: (neck_center, axis, lr, fwd)
        """
        neck_j = np.array(self.get_joint("c_neck"), dtype=float)
        head_j = np.array(self.get_joint("c_head"), dtype=float)
        axis = head_j - neck_j
        axis_n = np.linalg.norm(axis)
        if axis_n < 1e-6:
            axis = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            axis = axis / axis_n

        l_sh = np.array(self.get_joint("l_uparm"), dtype=float)
        r_sh = np.array(self.get_joint("r_uparm"), dtype=float)
        lr = r_sh - l_sh
        lr_n = np.linalg.norm(lr)
        if lr_n < 1e-6:
            lr = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            lr = lr / lr_n

        fwd = np.cross(lr, axis)
        fwd_n = np.linalg.norm(fwd)
        if fwd_n < 1e-6:
            # fallback to global Z as "forward-ish"
            fwd = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            fwd = fwd / fwd_n

        neck_center = (neck_j + head_j) * 0.5
        return neck_center, axis, lr, fwd

    def _compute_torso_normal(self):
        """
        Calculate the torso vector (spine direction) from pelvis to neck,
        and return it as the plane normal for circular slices.
        
        CRITICAL: The plane normal MUST be the spine direction itself, NOT perpendicular to it.
        Using a perpendicular vector would create a vertical slice (head to toe) instead of
        a horizontal slice (around the waist/chest/hip).
        
        This ensures all torso measurements (Chest, Waist, Hip) use the same alignment
        and create proper circular cross-sections that are "square" to the body's pose.
        """
        try:
            # Get pelvis (bottom) and neck (top) joints
            pelvis = np.array(self.get_joint('root'), dtype=float)
            neck = np.array(self.get_joint('c_neck'), dtype=float)
            
            # Calculate spine direction (the "center pole" of the body)
            spine_vector = neck - pelvis
            spine_norm = np.linalg.norm(spine_vector)
            
            if spine_norm < 1e-6:
                # Fallback to vertical if spine is degenerate
                return np.array([0.0, 1.0, 0.0], dtype=float)
            
            # The plane normal IS the spine direction (not perpendicular to it!)
            # This creates a horizontal slice relative to the body's orientation
            spine_direction = spine_vector / spine_norm
            return spine_direction
                
        except Exception as e:
            print(f"Warning: Error computing torso normal: {e}")
            return np.array([0.0, 1.0, 0.0], dtype=float)
    
    def _compute_neck_normal(self):
        """
        Calculate the neck vector (from neck joint to head joint),
        and return it as the plane normal for the neck measurement.
        
        This creates a circular slice at the mid-neck position (light blue ring),
        which is more accurate than measuring at the base of the neck where
        shoulders (trapezius) cause oval distortions.
        """
        try:
            # Get neck and head joints
            neck_j = np.array(self.get_joint('c_neck'), dtype=float)
            head_j = np.array(self.get_joint('c_head'), dtype=float)
            
            # Calculate neck direction (from neck to head)
            neck_vector = head_j - neck_j
            neck_norm = np.linalg.norm(neck_vector)
            
            if neck_norm < 1e-6:
                # Fallback to vertical if neck vector is degenerate
                return np.array([0.0, 1.0, 0.0], dtype=float)
            
            # The plane normal IS the neck direction (same logic as torso)
            # This creates a horizontal slice relative to the neck's orientation
            neck_direction = neck_vector / neck_norm
            return neck_direction
                
        except Exception as e:
            print(f"Warning: Error computing neck normal: {e}")
            return np.array([0.0, 1.0, 0.0], dtype=float)

    def _update_neck_hint_overlay(self, render=False):
        """
        Add a subtle overlay marking the approximate neck measurement zone and a tilted guide ring.
        This is only a visual hint to match the reference image (tilted path), not the final geodesic.
        """
        try:
            self.plotter.remove_actor("neck_region_hint")
        except Exception:
            pass
        try:
            self.plotter.remove_actor("neck_guide_ring")
        except Exception:
            pass

        try:
            neck_center, axis, lr, fwd = self._compute_neck_frame()

            # Neck region band (helps user see where to click)
            # Units here are ~cm in this tool.
            p = self.mesh.points
            r = 22.0
            y_min = float(neck_center[1] - 22.0)
            y_max = float(neck_center[1] + 12.0)
            mask = (np.linalg.norm(p - neck_center, axis=1) < r) & (p[:, 1] > y_min) & (p[:, 1] < y_max)
            region = self.mesh.extract_points(mask, adjacent_cells=True).extract_surface().triangulate()
            if getattr(region, "n_points", 0) > 50:
                self.plotter.add_mesh(
                    region,
                    color="#ff9f1a",  # warm orange
                    opacity=0.12,
                    name="neck_region_hint",
                    render=False,
                    lighting=False,
                )

            # Tilted guide ring (planar slice with a forward tilt)
            # We tilt the plane normal toward forward so the cut sits higher in back / lower in front visually.
            tilt = 0.65
            plane_normal = axis + tilt * fwd
            plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
            # Place the guide slightly below neck_center to better match "base of neck"
            origin = neck_center - 6.0 * axis
            guide = self.mesh.slice(normal=plane_normal, origin=origin)
            if getattr(guide, "n_points", 0) > 10:
                # keep nearest loop to neck center to avoid catching shoulders
                try:
                    conn = guide.connectivity()
                    if "RegionId" in conn.point_data:
                        best = None
                        best_d = float("inf")
                        for rid in np.unique(conn.point_data["RegionId"]):
                            seg = conn.threshold([rid - 0.1, rid + 0.1], scalars="RegionId").extract_surface()
                            if getattr(seg, "n_points", 0) < 5:
                                continue
                            d = np.min(np.linalg.norm(seg.points - origin, axis=1))
                            if d < best_d:
                                best_d = d
                                best = seg
                        if best is not None and getattr(best, "n_points", 0) > 5:
                            guide = best
                except Exception:
                    pass

                try:
                    tube = guide.tube(radius=0.9, capping=True)
                    if getattr(tube, "n_points", 0) > 0:
                        self.plotter.add_mesh(
                            tube,
                            color="#ff5a3d",
                            opacity=1.0,
                            name="neck_guide_ring",
                            render=False,
                            lighting=False,
                        )
                except Exception:
                    pass
        except Exception as e:
            print(f"Neck hint overlay error: {e}")
        finally:
            if render:
                try:
                    self.plotter.render()
                except Exception:
                    pass

    def compute_circumference(self, center, normal, exclude_arms=True, arm_only=False, target_joint_center=None):
        """
        Computes circumference by slicing mesh.
        For torso measurements (exclude_arms=True), extracts only the largest contour.
        For arm measurements (arm_only=True), filters to arm-only component.
        """
        try:
            if np.linalg.norm(normal) < 1e-6: 
                return 0.0, None
            normal = np.array(normal) / np.linalg.norm(normal)
            
            # Create the slice
            slice_mesh = self.mesh.slice(normal=normal, origin=center)
            
            if slice_mesh.n_points < 3:
                return 0.0, None
            
            def _as_polydata(ds):
                """Best-effort convert any VTK dataset to PolyData."""
                try:
                    if isinstance(ds, pv.PolyData):
                        return ds
                    # connectivity() can return UnstructuredGrid; extract geometry back to PolyData
                    return ds.extract_geometry()
                except Exception:
                    return pv.PolyData(ds)

            def _select_component_near_point(contour, target_pt):
                """
                Slice results can contain multiple closed loops (torso + arm + etc).
                This selects the connected component whose points are closest to target_pt.
                """
                try:
                    conn = contour.connectivity()
                    conn = _as_polydata(conn)
                    if conn.n_points < 3:
                        return contour
                    if "RegionId" not in conn.point_data:
                        return contour

                    region_ids = np.unique(conn.point_data["RegionId"])
                    best_id = None
                    best_dist = float("inf")
                    t = np.array(target_pt, dtype=float)
                    for rid in region_ids:
                        mask = conn.point_data["RegionId"] == rid
                        if np.count_nonzero(mask) < 3:
                            continue
                        pts = conn.points[mask]
                        d = np.min(np.linalg.norm(pts - t, axis=1))
                        if d < best_dist:
                            best_dist = d
                            best_id = rid
                    if best_id is None:
                        return contour

                    picked = conn.threshold([best_id - 0.1, best_id + 0.1], scalars="RegionId")
                    picked = _as_polydata(picked)
                    if picked.n_points < 3:
                        return contour
                    return picked
                except Exception as e:
                    print(f"Component select error: {e}")
                    return contour

            # Filter connected components
            if exclude_arms:
                # Extract largest component (torso)
                try:
                    torso_contour = slice_mesh.connectivity(extraction_mode="largest")
                    torso_contour = _as_polydata(torso_contour)
                    if torso_contour.n_points < 3:
                        torso_contour = slice_mesh
                except Exception:
                    torso_contour = slice_mesh
            elif arm_only and target_joint_center is not None:
                # For arms: pick the loop closest to the arm joint (avoids torso intersections)
                torso_contour = _select_component_near_point(slice_mesh, target_joint_center)
            else:
                torso_contour = slice_mesh
            
            # Calculate perimeter
            if not hasattr(torso_contour, 'lines') or len(torso_contour.lines) == 0:
                return 0.0, None
                
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
            
            # Return mesh with lines for visualization
            return perimeter, torso_contour
            
        except Exception as e:
            print(f"Circumference Error: {e}")
            return 0.0, None

    def compute_length(self, start_joint, end_joint, vertical_only=False):
        """
        Compute length between two joints.
        
        Args:
            start_joint: Starting joint name or special marker ('min_y', 'max_y')
            end_joint: Ending joint name or special marker ('min_y', 'max_y')
            vertical_only: If True, only compute vertical (Y-axis) distance. 
                          Used for measurements like Shoulder-Crotch (C) which should be vertical.
        
        Returns:
            Distance in meters (will be converted to cm by caller)
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
        
        if vertical_only:
            # For vertical measurements (like Shoulder-Crotch), use Y-axis difference only
            return abs(end_pt[1] - start_pt[1])
        else:
            # For other measurements, use 3D Euclidean distance
            return np.linalg.norm(end_pt - start_pt)
    
    def compute_shoulder_width_geodesic(self):
        """
        Compute shoulder width using default vertex indices (from calibration).
        Uses 7 default vertex indices to calculate a geodesic path across the back.
        This is the automatic/default measurement when no manual picking has been done.
        Results are cached to avoid recalculating on every update.
        """
        # Return cached result if available
        if self._default_shoulder_width_cached is not None and self._default_shoulder_path_cached is not None:
            return self._default_shoulder_width_cached, self._default_shoulder_path_cached
        
        try:
            # Use default vertex indices for shoulder breadth
            # These indices correspond to: Left → Left Mid-Left → Left Mid → Nape → Right Mid → Right Mid-Right → Right
            default_indices = self.default_shoulder_vertex_indices
            
            # Validate indices are within mesh bounds
            if len(default_indices) != 7:
                print(f"Warning: Expected 7 default vertex indices, got {len(default_indices)}")
                return 0.0, None
            
            max_idx = len(self.mesh.points) - 1
            for i, idx in enumerate(default_indices):
                if idx < 0 or idx > max_idx:
                    print(f"Warning: Default vertex index {i} ({idx}) is out of bounds (0-{max_idx})")
                    return 0.0, None
            
            # Get the 7 vertex coordinates
            p0 = np.array(self.mesh.points[default_indices[0]], dtype=float)  # Left shoulder
            p1 = np.array(self.mesh.points[default_indices[1]], dtype=float)  # Left mid-left
            p2 = np.array(self.mesh.points[default_indices[2]], dtype=float)  # Left mid
            p3 = np.array(self.mesh.points[default_indices[3]], dtype=float)  # Nape
            p4 = np.array(self.mesh.points[default_indices[4]], dtype=float)  # Right mid
            p5 = np.array(self.mesh.points[default_indices[5]], dtype=float)  # Right mid-right
            p6 = np.array(self.mesh.points[default_indices[6]], dtype=float)  # Right shoulder
            
            # Only print debug info once (on first calculation)
            print(f"Debug: Using default vertex indices for shoulder width: {default_indices}")
            print(f"Debug: Default points - Left: {p0}, LeftMidLeft: {p1}, LeftMid: {p2}, Nape: {p3}, RightMid: {p4}, RightMidRight: {p5}, Right: {p6}")
            
            # Calculate Geodesic (Surface Path) - 6 segments connecting 7 points
            # Segment 1: Left → Left Mid-Left
            path_01 = self.mesh.geodesic(default_indices[0], default_indices[1])
            # Segment 2: Left Mid-Left → Left Mid
            path_12 = self.mesh.geodesic(default_indices[1], default_indices[2])
            # Segment 3: Left Mid → Nape
            path_23 = self.mesh.geodesic(default_indices[2], default_indices[3])
            # Segment 4: Nape → Right Mid
            path_34 = self.mesh.geodesic(default_indices[3], default_indices[4])
            # Segment 5: Right Mid → Right Mid-Right
            path_45 = self.mesh.geodesic(default_indices[4], default_indices[5])
            # Segment 6: Right Mid-Right → Right
            path_56 = self.mesh.geodesic(default_indices[5], default_indices[6])
            
            # Check if all paths are valid
            if path_01 is None or path_12 is None or path_23 is None or path_34 is None or path_45 is None or path_56 is None:
                print("Warning: Geodesic calculation failed for default indices, using straight line")
                # Fallback to straight line through all 7 points
                combined_points = np.array([p0, p1, p2, p3, p4, p5, p6])
                full_path = pv.PolyData(combined_points)
                # Create line connectivity for 7 points
                full_path.lines = np.array([7, 0, 1, 2, 3, 4, 5, 6], dtype=np.int32)
                total_width = (np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + 
                             np.linalg.norm(p3 - p2) + np.linalg.norm(p4 - p3) +
                             np.linalg.norm(p5 - p4) + np.linalg.norm(p6 - p5))
                return total_width, full_path
            
            # Snap path endpoints to exact vertex coordinates
            points_01 = path_01.points.copy()
            points_12 = path_12.points.copy()
            points_23 = path_23.points.copy()
            points_34 = path_34.points.copy()
            points_45 = path_45.points.copy()
            points_56 = path_56.points.copy()
            
            # Replace endpoints with exact vertex coordinates
            if len(points_01) > 0:
                points_01[0] = p0
                points_01[-1] = p1
            if len(points_12) > 0:
                points_12[0] = p1
                points_12[-1] = p2
            if len(points_23) > 0:
                points_23[0] = p2
                points_23[-1] = p3
            if len(points_34) > 0:
                points_34[0] = p3
                points_34[-1] = p4
            if len(points_45) > 0:
                points_45[0] = p4
                points_45[-1] = p5
            if len(points_56) > 0:
                points_56[0] = p5
                points_56[-1] = p6
            
            # Combine all segments into one smooth line
            combined_points = points_01
            if len(points_12) > 1:
                combined_points = np.vstack([combined_points, points_12[1:]])
            else:
                combined_points = np.vstack([combined_points, points_12])
            
            if len(points_23) > 1:
                combined_points = np.vstack([combined_points, points_23[1:]])
            else:
                combined_points = np.vstack([combined_points, points_23])
            
            if len(points_34) > 1:
                combined_points = np.vstack([combined_points, points_34[1:]])
            else:
                combined_points = np.vstack([combined_points, points_34])
            
            if len(points_45) > 1:
                combined_points = np.vstack([combined_points, points_45[1:]])
            else:
                combined_points = np.vstack([combined_points, points_45])
            
            if len(points_56) > 1:
                combined_points = np.vstack([combined_points, points_56[1:]])
            else:
                combined_points = np.vstack([combined_points, points_56])
            
            # Ensure first and last points are exact
            combined_points[0] = p0
            combined_points[-1] = p6
            
            # Create PolyData with proper lines array
            full_path = pv.PolyData(combined_points)
            n_points = len(combined_points)
            lines_array = np.empty(n_points + 1, dtype=np.int32)
            lines_array[0] = n_points
            lines_array[1:] = np.arange(n_points, dtype=np.int32)
            full_path.lines = lines_array
            
            # Calculate Length using the snapped path
            total_width = 0.0
            for i in range(len(combined_points) - 1):
                segment_length = np.linalg.norm(combined_points[i+1] - combined_points[i])
                total_width += segment_length
            
            # Validate the measurement is reasonable
            straight_dist = (np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + 
                           np.linalg.norm(p3 - p2) + np.linalg.norm(p4 - p3) +
                           np.linalg.norm(p5 - p4) + np.linalg.norm(p6 - p5))
            
            if total_width < straight_dist * 0.95:
                print(f"Warning: Default shoulder width path too short ({total_width:.1f} cm), using straight line")
                total_width = straight_dist
            elif total_width > straight_dist * 2.0:
                print(f"Warning: Default shoulder width path too long ({total_width:.1f} cm), using straight line")
                total_width = straight_dist
            elif total_width > 100.0:
                print(f"Warning: Default shoulder width exceeds maximum ({total_width:.1f} cm), using straight line")
                total_width = straight_dist
            
            print(f"Debug: Default shoulder width calculated: {total_width:.2f} cm (cached)")
            
            # Cache the result to avoid recalculating on every update
            self._default_shoulder_width_cached = total_width
            self._default_shoulder_path_cached = full_path
            
            return total_width, full_path
            
        except Exception as e:
            print(f"Warning: Error computing default geodesic shoulder width: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, None
    
    def compute_shoulder_crotch_geodesic(self):
        """
        Compute shoulder-to-crotch measurement using geodesic path (diagonal line down the front).
        Uses 3 points (mid-shoulder → chest → crotch) to guide the path and prevent zig-zags.
        This follows the surface of the mesh from mid-shoulder to crotch point.
        """
        try:
            # 1. Find start point: Mid-shoulder (between neck and shoulder tip)
            neck_pt = np.array(self.get_joint('c_spine3'), dtype=float)  # Upper spine/neck
            left_shoulder_pt = np.array(self.get_joint('l_uparm'), dtype=float)
            right_shoulder_pt = np.array(self.get_joint('r_uparm'), dtype=float)
            
            # Mid-shoulder is the average of left and right shoulder tips
            # Then average with neck to get the mid-shoulder point
            shoulder_mid = (left_shoulder_pt + right_shoulder_pt) / 2.0
            mid_shoulder_coords = (neck_pt + shoulder_mid) / 2.0
            
            # Find closest vertex on mesh to mid-shoulder point (ensure it's on surface)
            start_idx = self.mesh.find_closest_point(mid_shoulder_coords)
            start_pt = self.mesh.points[start_idx]
            
            # 2. Find intermediate point: Chest/Nipple area (keeps path on front of body)
            # This prevents the geodesic from wrapping around the back
            chest_spine_pt = np.array(self.get_joint('c_spine2'), dtype=float)  # Mid-chest spine
            chest_idx = self.mesh.find_closest_point(chest_spine_pt)
            chest_pt = self.mesh.points[chest_idx]
            
            # 3. Find end point: Crotch point (lowest point in torso region, between legs)
            root_pt = np.array(self.get_joint('root'), dtype=float)  # Pelvis center
            
            # Find the lowest point in the center region (between legs)
            y_min = self.vertices[:, 1].min()
            y_max = self.vertices[:, 1].max()
            crotch_y = y_min + (y_max - y_min) * 0.48  # About 48% from top (crotch level)
            
            # Find vertices near the center (X=0) at crotch height
            center_mask = (
                (np.abs(self.mesh.points[:, 0]) < 5.0) &  # Near center (within 5cm)
                (self.mesh.points[:, 1] >= crotch_y - 5.0) &  # Near crotch height
                (self.mesh.points[:, 1] <= crotch_y + 5.0)
            )
            
            if np.any(center_mask):
                # Find the lowest point in this region (the actual crotch)
                center_candidates = self.mesh.points[center_mask]
                center_indices = np.where(center_mask)[0]
                crotch_idx = center_indices[np.argmin(center_candidates[:, 1])]  # Lowest Y
            else:
                # Fallback: use root joint (pelvis center)
                crotch_idx = self.mesh.find_closest_point(root_pt)
            
            end_pt = self.mesh.points[crotch_idx]
            
            # 4. Calculate geodesic path in two segments: mid-shoulder → chest → crotch
            # This forces the path to stay on the front of the body
            try:
                # Segment 1: Mid-shoulder to chest
                path_upper = self.mesh.geodesic(start_idx, chest_idx)
                
                # Segment 2: Chest to crotch
                path_lower = self.mesh.geodesic(chest_idx, crotch_idx)
                
                # Validate both paths are valid
                if (path_upper is not None and path_lower is not None and 
                    hasattr(path_upper, 'points') and hasattr(path_lower, 'points') and
                    len(path_upper.points) > 0 and len(path_lower.points) > 0):
                    # Combine paths
                    upper_points = path_upper.points
                    lower_points = path_lower.points
                    
                    # Remove duplicate point at junction (chest point appears in both)
                    if len(lower_points) > 1:
                        combined_points = np.vstack([upper_points, lower_points[1:]])
                    else:
                        combined_points = np.vstack([upper_points, lower_points])
                    
                    # Create combined path with correct lines array format
                    # PyVista lines format: [num_points, idx0, idx1, idx2, ...]
                    num_points = len(combined_points)
                    lines_array = np.empty(num_points + 1, dtype=np.int32)
                    lines_array[0] = num_points
                    lines_array[1:] = np.arange(num_points, dtype=np.int32)
                    
                    combined_path = pv.PolyData(combined_points)
                    combined_path.lines = lines_array
                    
                    # Calculate total length (mesh is already in cm, no conversion needed)
                    if hasattr(path_upper, 'length') and hasattr(path_lower, 'length'):
                        if path_upper.length is not None and path_lower.length is not None:
                            path_length = float(path_upper.length + path_lower.length)  # Already in cm
                        else:
                            # Manual calculation
                            path_length = 0.0
                            for i in range(len(combined_points) - 1):
                                path_length += np.linalg.norm(combined_points[i+1] - combined_points[i])
                    else:
                        # Manual calculation
                        path_length = 0.0
                        for i in range(len(combined_points) - 1):
                            path_length += np.linalg.norm(combined_points[i+1] - combined_points[i])
                    
                    # Validate: geodesic should be longer than straight line (diagonal)
                    straight_dist = np.linalg.norm(end_pt - start_pt)  # Already in cm
                    
                    if path_length < straight_dist * 0.95:
                        # Path is shorter than straight (impossible), reject
                        print(f"Warning: Shoulder-crotch geodesic path too short ({path_length:.1f} cm), using straight line")
                        path_length = straight_dist
                        line_points = np.array([start_pt, end_pt])
                        line_path = pv.PolyData(line_points)
                        line_path.lines = np.array([2, 0, 1])
                        return path_length, line_path
                    elif path_length > straight_dist * 2.0:
                        # Path is more than 2x straight distance (likely wrong route), reject
                        print(f"Warning: Shoulder-crotch geodesic path too long ({path_length:.1f} cm), using straight line")
                        path_length = straight_dist
                        line_points = np.array([start_pt, end_pt])
                        line_path = pv.PolyData(line_points)
                        line_path.lines = np.array([2, 0, 1])
                        return path_length, line_path
                    
                    return path_length, combined_path
                else:
                    # Geodesic failed, use straight line
                    straight_dist = np.linalg.norm(end_pt - start_pt)  # Already in cm
                    line_points = np.array([start_pt, end_pt])
                    line_path = pv.PolyData(line_points)
                    line_path.lines = np.array([2, 0, 1])
                    return straight_dist, line_path
            except Exception as e:
                print(f"Warning: Geodesic calculation failed for shoulder-crotch: {e}")
                # Fallback to straight line
                straight_dist = np.linalg.norm(end_pt - start_pt)  # Already in cm
                line_points = np.array([start_pt, end_pt])
                line_path = pv.PolyData(line_points)
                line_path.lines = np.array([2, 0, 1])
                return straight_dist, line_path
                
        except Exception as e:
            print(f"Warning: Error computing geodesic shoulder-crotch: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to straight line between joints
            try:
                start_pt = self.get_joint('c_spine3')
                end_pt = self.get_joint('root')
                straight_dist = np.linalg.norm(np.array(end_pt) - np.array(start_pt))
                return straight_dist, None  # Already in cm
            except:
                return 0.0, None

    def update_measurement_visuals(self):
        try:
            # Clear previous visuals
            for name in list(self.measurements.keys()) + list(self.length_measurements.keys()):
                try:
                    self.plotter.remove_actor(f"ring_{name}")
                except (KeyError, ValueError):
                    pass
                try:
                    self.plotter.remove_actor(f"text_{name}")
                except (KeyError, ValueError):
                    pass
                try:
                    self.plotter.remove_actor(f"line_{name}")
                except (KeyError, ValueError):
                    pass
            # Clear custom measurement rings
            for i in range(len(self.custom_measurements)):
                try:
                    self.plotter.remove_actor(f"custom_ring_{i}")
                except (KeyError, ValueError):
                    pass
            # Clear the consolidated measurement panel text
            try:
                self.plotter.remove_actor("measurements_panel")
            except (KeyError, ValueError):
                pass
            # Legacy headers (from earlier versions)
            for legacy in ("header_torso", "header_arms", "header_legs"):
                try:
                    self.plotter.remove_actor(legacy)
                except (KeyError, ValueError):
                    pass
        except Exception as e:
            print(f"Warning: Error clearing visuals: {e}")

        # Build a single stable measurement panel (upper-left)
        panel_lines = []
        panel_lines.append("HEAD & TORSO")
        
        torso_measurements = ['Head (A)', 'Neck (B)', 'Chest (D)', 'Waist (E)', 'Hip (F)']
        for name in torso_measurements:
            if name in self.measurements:
                data = self.measurements[name]
                if 'midpoint' in data and data['midpoint']:
                    j1 = self.get_joint(data['midpoint'][0])
                    j2 = self.get_joint(data['midpoint'][1])
                    center = (j1 + j2) / 2
                else:
                    center = self.get_joint(data['joint'])
                
                # Calculate normal for proper circular slices
                if data.get('neck_aligned', False):
                    # For neck measurement: use neck-to-head vector as normal
                    # This creates a circular slice at mid-neck (light blue position)
                    normal = self._compute_neck_normal()
                    # Position the measurement higher up the neck (40-50% from neck to head)
                    if 'neck_height_ratio' in data:
                        neck_j = np.array(self.get_joint('c_neck'), dtype=float)
                        head_j = np.array(self.get_joint('c_head'), dtype=float)
                        neck_vec = head_j - neck_j
                        center = neck_j + neck_vec * data['neck_height_ratio']
                    else:
                        # Fallback: use midpoint if ratio not specified
                        neck_j = np.array(self.get_joint('c_neck'), dtype=float)
                        head_j = np.array(self.get_joint('c_head'), dtype=float)
                        center = (neck_j + head_j) / 2
                elif data.get('torso_aligned', False):
                    # For torso measurements: use consistent torso-normal alignment
                    # This ensures Chest, Waist, and Hip all use the same perpendicular-to-spine normal
                    normal = self._compute_torso_normal()
                elif data.get('normal_from'):
                    # For arms: normal is parallel to bone direction (perpendicular to slice plane)
                    a = self.get_joint(data['normal_from'][0])
                    b = self.get_joint(data['normal_from'][1])
                    bone_dir = b - a
                    bone_dir_norm = np.linalg.norm(bone_dir)
                    
                    if bone_dir_norm > 1e-6:
                        normal = bone_dir / bone_dir_norm
                    else:
                        normal = np.array(data['normal'], dtype=float)
                        normal = normal / (np.linalg.norm(normal) + 1e-12)
                else:
                    normal = np.array(data['normal'], dtype=float)
                    normal = normal / (np.linalg.norm(normal) + 1e-12)
                
                # Apply offset (neck_aligned already positioned center, so just apply offset)
                offset = data.get('offset', 0.0)
                center = center + offset * normal
                
                # Neck (B): if user provided geodesic loop, use it instead of planar slice
                exclude_arms = data.get('exclude_arms', False)
                if name == 'Neck (B)' and self.neck_geodesic_path is not None and self.neck_geodesic_path.n_points > 2:
                    value = float(self.neck_geodesic_path.length)
                    slice_mesh = self.neck_geodesic_path
                    # Use cyan color for geodesic neck path (light blue)
                    ring_color = 'cyan'
                else:
                    value, slice_mesh = self.compute_circumference(center, normal, exclude_arms=exclude_arms)
                    ring_color = data['color']
                
                if slice_mesh is not None:
                    # For geodesic paths (polylines), ensure we create a proper tube
                    if name == 'Neck (B)' and self.neck_geodesic_path is not None:
                        try:
                            # Create tube from the geodesic polyline path
                            tube = slice_mesh.tube(radius=1.5, capping=False)
                            if tube is not None and tube.n_points > 0:
                                self.plotter.add_mesh(tube, color=ring_color, name=f"ring_{name}", 
                                                    opacity=1.0, render=False, lighting=False)
                        except Exception as e:
                            print(f"Warning: Could not create geodesic neck tube: {e}")
                    elif hasattr(slice_mesh, 'lines') and len(slice_mesh.lines) > 0:
                        try:
                            tube = slice_mesh.tube(radius=1.0, capping=True)
                            if tube is not None and tube.n_points > 0:
                                self.plotter.add_mesh(tube, color=ring_color, name=f"ring_{name}", 
                                                    opacity=1.0, render=False, lighting=False)
                        except Exception as e:
                            print(f"Warning: Could not create ring for {name}: {e}")
                
                code = name.split('(')[1].split(')')[0] if '(' in name else ''
                label = f"{code}  {name.split('(')[0].strip():<12} {value:>6.1f} cm"
                panel_lines.append(label)
        
        # Length measurements
        if 'Shoulder-Crotch (C)' in self.length_measurements:
            name = 'Shoulder-Crotch (C)'
            data = self.length_measurements[name]
            
            # Use geodesic path for diagonal line down the front of the body
            geodesic_path = None
            if data.get('geodesic', False):
                value, geodesic_path = self.compute_shoulder_crotch_geodesic()
            else:
                # Fallback to straight line
                value = self.compute_length(data['start'], data['end'], vertical_only=False)
                start_pt = self.get_joint(data['start'])
                end_pt = self.get_joint(data['end'])
                line_points = np.array([start_pt, end_pt])
                geodesic_path = pv.PolyData(line_points)
                geodesic_path.lines = np.array([2, 0, 1])
            
            # Visualize as a line/tube (diagonal path, not a ring)
            if geodesic_path is not None and geodesic_path.n_points > 0:
                try:
                    if geodesic_path.n_points > 2:
                        # Geodesic path with multiple points: create a thin tube
                        tube = geodesic_path.tube(radius=0.6, capping=False)
                        if tube is not None and tube.n_points > 0:
                            self.plotter.add_mesh(tube, color=data['color'], name=f"line_{name}", 
                                                opacity=1.0, render=False, lighting=False)
                        else:
                            # Fallback: draw line segments
                            for i in range(geodesic_path.n_points - 1):
                                line = pv.Line(geodesic_path.points[i], geodesic_path.points[i+1])
                                self.plotter.add_mesh(line, color=data['color'], line_width=5, 
                                                    name=f"line_{name}_seg{i}", render=False, lighting=False)
                    else:
                        # Straight line between points: draw as thick line
                        if geodesic_path.n_points == 2:
                            line = pv.Line(geodesic_path.points[0], geodesic_path.points[1])
                            self.plotter.add_mesh(line, color=data['color'], line_width=6, 
                                                name=f"line_{name}", render=False, lighting=False)
                except Exception as e:
                    print(f"Warning: Could not visualize shoulder-crotch path: {e}")
            
            code = name.split('(')[1].split(')')[0] if '(' in name else ''
            label = f"{code}  {name.split('(')[0].strip():<12} {value:>6.1f} cm"
            panel_lines.append(label)
        
        if 'Shoulder Width (O)' in self.length_measurements:
            name = 'Shoulder Width (O)'
            data = self.length_measurements[name]
            
            # Check if user has manually picked 7 points for geodesic measurement
            geodesic_path = None
            if self.shoulder_geodesic_path is not None and self.shoulder_geodesic_path.n_points >= 2:
                # Use the stored path length (calculated with spline smoothing and correction factor)
                # This avoids recalculation errors that cause huge values
                if self.shoulder_path_length is not None:
                    value = self.shoulder_path_length
                else:
                    # Fallback: recalculate if stored value is missing
                    # Calculate straight-line distance through all 7 points for validation
                    # Mesh is already in cm (scaled to target height), so no conversion needed
                    if len(self.shoulder_landmark_points) >= 7:
                        pt0 = self.shoulder_landmark_points[0]  # Left shoulder
                        pt1 = self.shoulder_landmark_points[1]  # Left mid-left
                        pt2 = self.shoulder_landmark_points[2]  # Left mid
                        pt3 = self.shoulder_landmark_points[3]  # Nape
                        pt4 = self.shoulder_landmark_points[4]  # Right mid
                        pt5 = self.shoulder_landmark_points[5]  # Right mid-right
                        pt6 = self.shoulder_landmark_points[6]  # Right shoulder
                        # Straight-line distance: left → left mid-left → left mid → nape → right mid → right mid-right → right
                        straight_dist = (np.linalg.norm(pt1 - pt0) + np.linalg.norm(pt2 - pt1) + 
                                       np.linalg.norm(pt3 - pt2) + np.linalg.norm(pt4 - pt3) +
                                       np.linalg.norm(pt5 - pt4) + np.linalg.norm(pt6 - pt5))  # Already in cm
                    elif len(self.shoulder_landmark_points) >= 2:
                        # Partial points - calculate what we can
                        straight_dist = 0.0
                        for i in range(len(self.shoulder_landmark_points) - 1):
                            straight_dist += np.linalg.norm(self.shoulder_landmark_points[i+1] - 
                                                           self.shoulder_landmark_points[i])
                    else:
                        straight_dist = 50.0  # fallback
                    
                    # Calculate path length along the geodesic path
                    # IMPORTANT: This is a fallback - the stored value should be used instead
                    # Mesh is already in cm (scaled to target height), so no conversion needed
                    path_points = self.shoulder_geodesic_path.points
                    if len(path_points) < 2:
                        value = straight_dist
                    else:
                        value = 0.0
                        for i in range(len(path_points) - 1):
                            value += np.linalg.norm(path_points[i+1] - path_points[i])
                        # Already in cm, no conversion needed
                    
                    # Validate the value is reasonable (shoulder width should be 30-60 cm for adults)
                    # Geodesic should be slightly longer than straight line (1.0-1.8x), but not more than 2.5x
                    if value < straight_dist * 0.95:
                        # Path is shorter than straight (impossible), use straight
                        print(f"Warning: Geodesic path too short ({value:.1f} cm), using straight-line distance")
                        value = straight_dist
                    elif value > straight_dist * 2.5:
                        # Path is more than 2.5x straight distance (likely wrong route), use straight
                        print(f"Warning: Geodesic path too long ({value:.1f} cm vs {straight_dist:.1f} cm straight), using straight-line distance")
                        value = straight_dist
                    elif value > 100.0:
                        # Absolute maximum: shoulder width should never exceed 100cm
                        print(f"Warning: Geodesic path exceeds maximum ({value:.1f} cm), using straight-line distance")
                        value = straight_dist
                    
                    # Store the calculated value for next time
                    self.shoulder_path_length = value
                
                geodesic_path = self.shoulder_geodesic_path
            # Use geodesic path for tape-measure style shoulder width (automatic)
            elif data.get('geodesic', False):
                value, geodesic_path = self.compute_shoulder_width_geodesic()
            else:
                # Fallback to straight line between joints
                value = self.compute_length(data['start'], data['end'])
                start_pt = self.get_joint(data['start'])
                end_pt = self.get_joint(data['end'])
                line = pv.Line(start_pt, end_pt)
                self.plotter.add_mesh(line, color=data['color'], line_width=4, 
                                    name=f"line_{name}", render=False, lighting=False)
            
            if geodesic_path is not None and geodesic_path.n_points > 0:
                # Visualize ONLY the path (line/tube), NOT a ring
                # This is a point-to-point surface path, not a circumference measurement
                try:
                    # Remove any existing ring visualization for shoulder width
                    try:
                        self.plotter.remove_actor(f"ring_{name}")
                    except (KeyError, ValueError):
                        pass
                    
                    # Create a tube or line to visualize the path
                    # Use red color for shoulder width (yoke measurement) like reference image
                    path_color = 'red' if name == 'Shoulder Width (O)' else data['color']
                    
                    if geodesic_path.n_points > 2:
                        # Geodesic path with multiple points: create a thin tube
                        tube = geodesic_path.tube(radius=0.6, capping=False)
                        if tube is not None and tube.n_points > 0:
                            self.plotter.add_mesh(tube, color=path_color, name=f"line_{name}", 
                                                opacity=1.0, render=False, lighting=False)
                        else:
                            # Fallback: draw line segments connecting the points
                            for i in range(geodesic_path.n_points - 1):
                                line = pv.Line(geodesic_path.points[i], geodesic_path.points[i+1])
                                self.plotter.add_mesh(line, color=path_color, line_width=5, 
                                                    name=f"line_{name}_seg{i}", render=False, lighting=False)
                    else:
                        # Straight line between points: draw as thick line
                        if geodesic_path.n_points == 2:
                            line = pv.Line(geodesic_path.points[0], geodesic_path.points[1])
                            self.plotter.add_mesh(line, color=path_color, line_width=6, 
                                                name=f"line_{name}", render=False, lighting=False)
                        else:
                            # Single point or empty - shouldn't happen, but handle it
                            print(f"Warning: Shoulder width path has {geodesic_path.n_points} points")
                except Exception as e:
                    print(f"Warning: Could not visualize shoulder width path: {e}")
                    import traceback
                    traceback.print_exc()
            
            code = name.split('(')[1].split(')')[0] if '(' in name else ''
            label = f"{code}  {name.split('(')[0].strip():<12} {value:>6.1f} cm"
            panel_lines.append(label)

        panel_lines.append("")
        panel_lines.append("ARMS")
        
        arm_measurements = ['Wrist (G)', 'Bicep (H)', 'Forearm (I)']
        for name in arm_measurements:
            if name in self.measurements:
                data = self.measurements[name]
                if 'midpoint' in data and data['midpoint']:
                    j1 = self.get_joint(data['midpoint'][0])
                    j2 = self.get_joint(data['midpoint'][1])
                    center = (j1 + j2) / 2
                else:
                    center = self.get_joint(data['joint'])
                
                # Calculate normal from bone direction for proper circular slices
                if data.get('normal_from'):
                    a = self.get_joint(data['normal_from'][0])
                    b = self.get_joint(data['normal_from'][1])
                    n_vec = b - a
                    if np.linalg.norm(n_vec) > 1e-6:
                        normal = n_vec / np.linalg.norm(n_vec)
                    else:
                        normal = np.array(data['normal'], dtype=float)
                        normal = normal / (np.linalg.norm(normal) + 1e-12)
                else:
                    normal = np.array(data['normal'], dtype=float)
                    normal = normal / (np.linalg.norm(normal) + 1e-12)

                offset = data.get('offset', 0.0)
                original_center = center.copy() if isinstance(center, np.ndarray) else np.array(center)
                center = center + offset * normal
                
                exclude_arms = data.get('exclude_arms', False)
                arm_only = name in ['Wrist (G)', 'Bicep (H)', 'Forearm (I)']
                value, slice_mesh = self.compute_circumference(center, normal, exclude_arms=exclude_arms, 
                                                               arm_only=arm_only, target_joint_center=original_center if arm_only else None)
                
                if slice_mesh is not None and hasattr(slice_mesh, 'lines') and len(slice_mesh.lines) > 0:
                    try:
                        tube = slice_mesh.tube(radius=1.0, capping=True)
                        if tube is not None and tube.n_points > 0:
                            self.plotter.add_mesh(tube, color=data['color'], name=f"ring_{name}", 
                                                opacity=1.0, render=False, lighting=False)
                    except Exception as e:
                        print(f"Warning: Could not create ring for {name}: {e}")
                
                code = name.split('(')[1].split(')')[0] if '(' in name else ''
                label = f"{code}  {name.split('(')[0].strip():<12} {value:>6.1f} cm"
                panel_lines.append(label)
        
        # Arm Length
        if 'Arm Length (J)' in self.length_measurements:
            data = self.length_measurements['Arm Length (J)']
            if 'segments' in data:
                total_length = 0.0
                segments = data['segments']
                for i in range(len(segments) - 1):
                    p1 = self.get_joint(segments[i])
                    p2 = self.get_joint(segments[i+1])
                    segment_length = np.linalg.norm(p2 - p1)
                    total_length += segment_length
                    line = pv.Line(p1, p2)
                    self.plotter.add_mesh(line, color=data['color'], line_width=4, 
                                        name=f"line_Arm Length (J)_seg{i}", render=False, lighting=False)
                panel_lines.append(f"J  Arm Length     {total_length:>6.1f} cm")

        panel_lines.append("")
        panel_lines.append("LEGS")
        
        leg_measurements = ['Thigh (L)', 'Calf (M)', 'Ankle (N)']
        for name in leg_measurements:
            if name in self.measurements:
                data = self.measurements[name]
                if 'midpoint' in data and data['midpoint']:
                    j1 = self.get_joint(data['midpoint'][0])
                    j2 = self.get_joint(data['midpoint'][1])
                    center = (j1 + j2) / 2
                else:
                    center = self.get_joint(data['joint'])
                
                offset = data.get('offset', 0.0)
                normal = np.array(data['normal']) / np.linalg.norm(data['normal'])
                center = center + offset * normal
                
                exclude_arms = data.get('exclude_arms', False)
                value, slice_mesh = self.compute_circumference(center, data['normal'], exclude_arms=exclude_arms)
                
                if slice_mesh is not None and hasattr(slice_mesh, 'lines') and len(slice_mesh.lines) > 0:
                    try:
                        tube = slice_mesh.tube(radius=1.0, capping=True)
                        if tube is not None and tube.n_points > 0:
                            self.plotter.add_mesh(tube, color=data['color'], name=f"ring_{name}", 
                                                opacity=1.0, render=False, lighting=False)
                    except Exception as e:
                        print(f"Warning: Could not create ring for {name}: {e}")
                
                code = name.split('(')[1].split(')')[0] if '(' in name else ''
                panel_lines.append(f"{code}  {name.split('(')[0].strip():<12} {value:>6.1f} cm")

        # Always show a neck-zone hint overlay (to match the reference "tilted" measurement area)
        # If a geodesic is already set, the user still benefits from the zone highlight.
        self._update_neck_hint_overlay(render=False)
        
        # Leg length measurements
        leg_length_measurements = ['Inside Leg (K)', 'Height (P)']
        for name in leg_length_measurements:
            if name in self.length_measurements:
                data = self.length_measurements[name]
                start_joint = data['start']
                end_joint = data['end']
                
                if name == 'Height (P)':
                    value = self.target_height_cm
                    y_max = self.vertices[:, 1].max()
                    y_min = self.vertices[:, 1].min()
                    mid_x = self.vertices[:, 0].mean()
                    mid_z = self.vertices[:, 2].mean()
                    start_pt = np.array([mid_x, y_max, mid_z])
                    end_pt = np.array([mid_x, y_min, mid_z])
                    line = pv.Line(start_pt, end_pt)
                    self.plotter.add_mesh(line, color=data['color'], line_width=3, 
                                        name=f"line_{name}", render=False, lighting=False)
                else:
                    value = self.compute_length(start_joint, end_joint)
                    if start_joint == 'min_y':
                        start_pt = np.array([self.vertices[:, 0].mean(), self.vertices[:, 1].min(), self.vertices[:, 2].mean()])
                    else:
                        start_pt = self.get_joint(start_joint)
                    
                    if end_joint == 'min_y':
                        end_pt = np.array([self.vertices[:, 0].mean(), self.vertices[:, 1].min(), self.vertices[:, 2].mean()])
                    else:
                        end_pt = self.get_joint(end_joint)
                    
                    line = pv.Line(start_pt, end_pt)
                    self.plotter.add_mesh(line, color=data['color'], line_width=4, 
                                        name=f"line_{name}", render=False, lighting=False)
                
                code = name.split('(')[1].split(')')[0] if '(' in name else ''
                panel_lines.append(f"{code}  {name.split('(')[0].strip():<12} {value:>6.1f} cm")

        # Visualize custom measurements
        for i, custom in enumerate(self.custom_measurements):
            try:
                value, slice_mesh = self.compute_circumference(
                    custom['center'], 
                    custom['normal'], 
                    exclude_arms=False
                )
                if slice_mesh is not None and slice_mesh.n_points > 0:
                    try:
                        tube = slice_mesh.tube(radius=1.0, capping=True)
                        if tube is not None and tube.n_points > 0:
                            self.plotter.add_mesh(
                                tube, 
                                color='cyan', 
                                name=f"custom_ring_{i}", 
                                opacity=1.0, 
                                render=False, 
                                lighting=False
                            )
                    except Exception as e:
                        print(f"Warning: Could not create custom ring {i}: {e}")
            except Exception as e:
                print(f"Warning: Could not visualize custom measurement {i}: {e}")

        # Add consolidated measurement text panel (stable anchor)
        try:
            panel_text = "\n".join(panel_lines)
            self.plotter.add_text(
                panel_text,
                position="upper_left",
                font_size=13,
                color="white",
                name="measurements_panel",
                render=False,
                font="courier",
            )
        except Exception as e:
            print(f"Warning: Could not add measurements panel: {e}")

        try:
            self.plotter.render()
        except Exception as e:
            print(f"Warning: Error during render: {e}")

    def create_slider_callback(self, name):
        def callback(value):
            if name in self.measurements:
                self.measurements[name]['offset'] = value
                self.update_measurement_visuals()
        return callback

    def toggle_picking(self):
        """Toggle Picking Mode"""
        # If neck picking is active, don't allow toggling regular picking
        if self._active_pick_mode == "neck":
            self.plotter.add_text("✗ Finish neck picking first (pick 4 points)", 
                                position='upper_left', font_size=13, color='red', 
                                name='status_msg', font='courier')
            return
        if self._active_pick_mode == "shoulder":
            self.plotter.add_text("✗ Finish shoulder width picking first (pick 7 points)", 
                                position='upper_left', font_size=13, color='red', 
                                name='status_msg', font='courier')
            return
        
        self.picking_enabled = not self.picking_enabled
        if self.picking_enabled:
            # Disable any existing picking first
            try:
                self.plotter.disable_picking()
                self.plotter.untrack_click_position(side='left')
            except Exception:
                pass
            
            # Use track_click_position to avoid pickpoint attribute issue
            def regular_pick_callback(point):
                if self._active_pick_mode is None and self.picking_enabled:
                    self._handle_regular_pick(point)
            
            self.plotter.track_click_position(callback=regular_pick_callback, side='left')
            self._active_pick_mode = None  # Regular picking mode
            
            self.plotter.add_text("✓ PICKING MODE: ON | Left Click on surface to pick points", 
                                position='upper_left', font_size=13, color='lightgreen', 
                                name='status_msg', font='courier')
        else:
            # Disable picking
            try:
                self.plotter.disable_picking()
                self.plotter.untrack_click_position(side='left')
            except Exception:
                pass
            self.picking_enabled = False
            self._active_pick_mode = None
            self.plotter.add_text("Ready. Press 'P' to start picking points.", 
                                position='upper_left', font_size=13, color='lightgreen', 
                                name='status_msg', font='courier')

    def _handle_regular_pick(self, picked_pt):
        """Handle a regular point pick (for custom ring creation) - allows free 3D picking"""
        if picked_pt is None:
            return

        picked_pt = np.array(picked_pt)
        
        # Store freely picked point (no mesh surface constraint)
        self.selected_points.append(picked_pt.copy())
        
        sphere = pv.Sphere(radius=1.5, center=picked_pt)
        self.plotter.add_mesh(sphere, color='red', name=f"pt_{len(self.selected_points)}", render=False)
        
        if len(self.selected_points) > 1:
            p1 = self.selected_points[-2]
            p2 = self.selected_points[-1]
            line = pv.Line(p1, p2)
            self.plotter.add_mesh(line, color='red', line_width=4, name=f"line_{len(self.selected_points)}", render=False)
            
        if len(self.selected_points) > 2:
            p_start = self.selected_points[0]
            p_end = self.selected_points[-1]
            line = pv.Line(p_start, p_end)
            self.plotter.add_mesh(line, color='pink', line_width=2, style='wireframe', name="preview_close_line", render=False)
            
        status = f"✓ Picked: {len(self.selected_points)} points | Press 'C' to create custom ring"
        self.plotter.add_text(status, position='upper_left', font_size=13, color='yellow', 
                            name='status_msg', font='courier')
        self.plotter.render()

    def undo_last_point(self):
        """Undo the last picked point"""
        if not self.selected_points:
            return
            
        idx = len(self.selected_points)
        try:
            self.plotter.remove_actor(f"pt_{idx}")
        except KeyError:
            pass
        if idx > 1:
            try:
                self.plotter.remove_actor(f"line_{idx}")
            except KeyError:
                pass
        try:
            self.plotter.remove_actor("preview_close_line")
        except KeyError:
            pass
        
        self.selected_points.pop()
        
        status = f"✓ Undone. Points remaining: {len(self.selected_points)}"
        if self.picking_enabled:
            self.plotter.add_text(status + " | Left Click to pick", position='upper_left', 
                                font_size=13, color='lightgreen', name='status_msg', font='courier')
        
        self.plotter.render()

    def create_custom_ring(self):
        if len(self.selected_points) < 3:
            self.plotter.add_text("✗ Error: Pick at least 7 points first!", position='upper_left', 
                                font_size=13, color='red', name='status_msg', font='courier')
            return
            
        points = np.array(self.selected_points)
        centroid = np.mean(points, axis=0)
        pts_centered = points - centroid
        u, s, vh = np.linalg.svd(pts_centered)
        normal = vh[2, :]
        if normal[1] < 0: normal = -normal
            
        self.custom_measurements.append({'center': centroid, 'normal': normal})
        
        # Cleanup
        for i in range(1, len(self.selected_points) + 1):
            try:
                self.plotter.remove_actor(f"pt_{i}")
            except KeyError:
                pass
            try:
                self.plotter.remove_actor(f"line_{i}")
            except KeyError:
                pass
        try:
            self.plotter.remove_actor("preview_close_line")
        except KeyError:
            pass
        self.selected_points = []
        
        # Disable picking after creating ring
        self.picking_enabled = False
        self._active_pick_mode = None
        try:
            self.plotter.untrack_click_position(side='left')
        except Exception:
            pass
        
        self.plotter.add_text("✓ Custom ring created! Press 'P' to pick more points.", 
                            position='upper_left', font_size=13, color='lightgreen', 
                            name='status_msg', font='courier')
        self.update_measurement_visuals()

    def start_neck_landmark_picking(self):
        """
        Start picking 4 landmarks for the neck base (B) geodesic loop.
        Recommended pick order: back (nape) -> right -> front (suprasternal) -> left.
        """
        # Disable any existing picking first
        try:
            self.plotter.disable_picking()
            self.plotter.untrack_click_position(side='left')
        except Exception:
            pass
        
        # Clear previous neck picks visuals
        for i in range(1, 10):
            try:
                self.plotter.remove_actor(f"neck_pt_{i}")
            except (KeyError, ValueError):
                pass
        self.neck_landmark_vertex_ids = []
        self.neck_landmark_points = []
        self.neck_geodesic_path = None
        self._active_pick_mode = "neck"
        self.picking_enabled = True
        
        # Use track_click_position callback to avoid pickpoint attribute issue
        # This bypasses PyVista's internal picking mechanism
        def custom_click_callback(point):
            if self._active_pick_mode != "neck":
                return
            self._handle_neck_pick(point)
        
        self.plotter.track_click_position(callback=custom_click_callback, side='left')
        
        self.plotter.add_text(
            "Neck mode: pick 4 points (back→right→front→left).",
            position='upper_left',
            font_size=13,
            color='yellow',
            name='status_msg',
            font='courier',
        )
        self.plotter.render()
    
    def _handle_neck_pick(self, picked_pt):
        """Handle a neck landmark pick - allows free 3D picking without mesh constraints"""
        if picked_pt is None:
            return
        
        picked_pt = np.array(picked_pt)
        
        # Store the freely picked point (no mesh surface constraint)
        self.neck_landmark_points.append(picked_pt.copy())
        
        # For geodesic calculation, we'll project to mesh surface later
        # But store the free 3D point for visualization
        try:
            vid = int(np.argmin(np.linalg.norm(self.mesh.points - picked_pt, axis=1)))
            self.neck_landmark_vertex_ids.append(vid)
        except Exception:
            self.neck_landmark_vertex_ids.append(None)
        sphere = pv.Sphere(radius=1.8, center=picked_pt)
        self.plotter.add_mesh(sphere, color='orange', name=f"neck_pt_{len(self.neck_landmark_vertex_ids)}", render=False)
        remaining = 4 - len(self.neck_landmark_points)
        if remaining > 0:
            self.plotter.add_text(
                f"Neck landmarks: {len(self.neck_landmark_points)}/4 (pick {remaining} more)",
                position='upper_left',
                font_size=13,
                color='yellow',
                name='status_msg',
                font='courier',
            )
            self.plotter.render()
            return

        # Build geodesic loop and exit mode
        self._build_neck_geodesic_from_landmarks()
        self._active_pick_mode = None
        self.picking_enabled = False
        try:
            self.plotter.untrack_click_position(side='left')
        except Exception:
            pass
        self.plotter.add_text(
            "✓ Neck geodesic set. Neck (B) now follows surface contour.",
            position='upper_left',
            font_size=13,
            color='lightgreen',
            name='status_msg',
            font='courier',
        )
        self.update_measurement_visuals()
    
    def _build_neck_geodesic_from_landmarks(self):
        """Create a closed polyline (pv.PolyData) from 4 landmark vertex ids."""
        try:
            # Prefer using picked points (more robust) and constrain the geodesic to a neck-only region
            pts_in = list(self.neck_landmark_points[:4]) if self.neck_landmark_points else []
            if len(pts_in) < 2:
                return

            # Estimate neck axis from skeleton (head - neck)
            neck_j = self.get_joint("c_neck")
            head_j = self.get_joint("c_head")
            axis = np.array(head_j) - np.array(neck_j)
            axis_n = np.linalg.norm(axis)
            if axis_n < 1e-6:
                axis = np.array([0.0, 1.0, 0.0])
            else:
                axis = axis / axis_n

            # Build a local frame (u, v) perpendicular to axis to sort points around the neck
            tmp = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
            u = np.cross(axis, tmp)
            u = u / (np.linalg.norm(u) + 1e-12)
            v = np.cross(axis, u)
            v = v / (np.linalg.norm(v) + 1e-12)

            center = np.mean(np.vstack(pts_in), axis=0)
            # Compute angles in the neck-local plane and sort around
            angs = []
            for p in pts_in:
                d = np.array(p) - center
                x = float(np.dot(d, u))
                y = float(np.dot(d, v))
                angs.append(np.arctan2(y, x))
            order = np.argsort(np.array(angs))
            pts_sorted = [pts_in[i] for i in order]

            # Constrain geodesics to a neck-only submesh to prevent shortcuts across head/torso
            # Units in this tool are ~cm, so these radii are in cm.
            neck_center = (np.array(neck_j) + np.array(head_j)) * 0.5
            r = 18.0
            y_min = float(min(neck_j[1], head_j[1]) - 18.0)
            y_max = float(max(neck_j[1], head_j[1]) + 8.0)
            p = self.mesh.points
            mask = (np.linalg.norm(p - neck_center, axis=1) < r) & (p[:, 1] > y_min) & (p[:, 1] < y_max)
            neck_region = self.mesh.extract_points(mask, adjacent_cells=True)
            neck_region = neck_region.extract_surface().triangulate()
            if getattr(neck_region, "n_points", 0) < 50:
                neck_region = self.mesh

            # Convert sorted freely picked points -> closest vertex ids on neck region
            # This projects free 3D points to the mesh surface for geodesic calculation
            ids = []
            for p0 in pts_sorted:
                try:
                    # Project free point to nearest mesh surface point
                    vid = int(neck_region.find_closest_point(p0))
                except Exception:
                    vid = int(np.argmin(np.linalg.norm(neck_region.points - np.array(p0), axis=1)))
                ids.append(vid)

            paths = []
            total_len = 0.0
            for a, b in zip(ids, ids[1:] + ids[:1]):
                seg = neck_region.geodesic(a, b)
                if seg is None or getattr(seg, "n_points", 0) < 2:
                    continue
                paths.append(seg)
                total_len += float(seg.length)

            if not paths:
                return

            # Combine geodesic segments into one smooth closed polyline
            # Remove duplicate endpoints between segments to ensure smooth connection
            all_pts = []
            for i, seg in enumerate(paths):
                pts = seg.points
                if i == 0:
                    # First segment: include all points
                    all_pts.append(pts)
                else:
                    # Subsequent segments: skip first point (it's the same as last point of previous segment)
                    if len(pts) > 1:
                        all_pts.append(pts[1:])
            
            # Ensure we close the loop: if last point != first point, add first point at the end
            if len(all_pts) > 0:
                pts_cat = np.vstack(all_pts)
                first_pt = all_pts[0][0]
                last_pt = pts_cat[-1]
                if np.linalg.norm(first_pt - last_pt) > 0.1:  # If not already closed
                    pts_cat = np.vstack([pts_cat, first_pt.reshape(1, -1)])
                
                # Create a single continuous polyline: [n_total, 0, 1, 2, ..., n_total-1]
                n_total = len(pts_cat)
                lines_cat = np.concatenate(([n_total], np.arange(n_total, dtype=np.int64))).astype(np.int64)
                self.neck_geodesic_path = pv.PolyData(pts_cat, lines_cat)
            else:
                self.neck_geodesic_path = None
        except Exception as e:
            print(f"Neck geodesic build error: {e}")
            import traceback
            traceback.print_exc()
            self.neck_geodesic_path = None

    def start_shoulder_width_picking(self):
        """
        Start picking 7 points for shoulder width geodesic measurement (tape measure style).
        Pick points across the back: left shoulder → left mid-left → left mid → nape → right mid → right mid-right → right shoulder.
        """
        # Disable any existing picking first
        try:
            self.plotter.disable_picking()
            self.plotter.untrack_click_position(side='left')
        except Exception:
            pass
        
        # Clear previous shoulder picks visuals
        for i in range(1, 8):
            try:
                self.plotter.remove_actor(f"shoulder_pt_{i}")
            except (KeyError, ValueError):
                pass
        self.shoulder_landmark_points = []
        self.shoulder_landmark_vertex_ids = []
        self.shoulder_geodesic_path = None
        self.shoulder_path_length = None
        # Clear cache when starting manual picking (so default won't be used)
        self._default_shoulder_width_cached = None
        self._default_shoulder_path_cached = None
        self._active_pick_mode = "shoulder"
        self.picking_enabled = True
        
        # Use track_click_position with improved ray-trace intersection
        def shoulder_click_callback(point):
            if self._active_pick_mode != "shoulder":
                return
            # The point from track_click_position is along the camera ray
            # We need to find where it intersects the mesh using ray-tracing
            self._handle_shoulder_pick_with_ray_trace(point)
        
        self.plotter.track_click_position(callback=shoulder_click_callback, side='left')
        
        self.plotter.add_text(
            "Shoulder Width mode: pick 7 points (left shoulder → left mid-left → left mid → nape → right mid → right mid-right → right shoulder).",
            position='upper_left',
            font_size=13,
            color='yellow',
            name='status_msg',
            font='courier',
        )
        self.plotter.render()
    
    def _handle_shoulder_pick_with_ray_trace(self, picked_pt):
        """Handle shoulder pick by ray-tracing to find actual mesh intersection"""
        if picked_pt is None:
            return
        
        picked_pt = np.array(picked_pt)
        
        # Use PyVista's ray_trace to find the actual intersection point with the mesh
        try:
            # Get camera position
            camera_pos = np.array(self.plotter.camera_position[0])
            
            # Calculate ray direction from camera to picked point
            ray_dir = picked_pt - camera_pos
            ray_dir_norm = np.linalg.norm(ray_dir)
            if ray_dir_norm < 1e-6:
                return
            ray_dir = ray_dir / ray_dir_norm
            
            # Ray-trace from camera through picked point to find mesh intersection
            # Use a long ray (1000 units) to ensure we hit the mesh
            ray_end = camera_pos + ray_dir * 1000.0
            
            # Find intersection with mesh
            intersection, cell_ids = self.mesh.ray_trace(camera_pos, ray_end, first_point=True)
            
            if intersection is not None and len(intersection) > 0:
                # Use the first intersection point (closest to camera)
                intersection_pt = intersection[0]
                self._handle_shoulder_pick(intersection_pt)
            else:
                # No intersection found, fall back to projection
                self._handle_shoulder_pick(picked_pt)
        except Exception as e:
            print(f"Warning: Ray-trace failed: {e}, using direct projection")
            self._handle_shoulder_pick(picked_pt)
    
    def _handle_shoulder_pick(self, picked_input):
        """
        Robust handler: Guarantees we save XYZ coordinates [x, y, z], never just an Index ID.
        This fixes the "inhomogeneous shape" error and ensures accurate measurements.
        """
        point_coords = None
        
        # 1. Convert whatever the picker gave us into a valid [x,y,z] array
        if isinstance(picked_input, (int, np.integer)):
            # If it gave us an Index ID, look up the coordinates
            point_coords = np.array(self.mesh.points[picked_input], dtype=float)
            print(f"Info: Received index {picked_input}, converted to coordinates: {point_coords}")
        elif hasattr(picked_input, '__len__') and len(picked_input) == 3:
            # If it gave us coordinates, make sure it's a numpy array
            point_coords = np.array(picked_input, dtype=float)
        else:
            # Fallback: Raycast to find the exact point on surface
            try:
                pos = self.plotter.pick_mouse_position()
                if pos is not None and len(pos) >= 3:
                    idx = self.mesh.find_closest_point(pos[:3])
                    point_coords = np.array(self.mesh.points[idx], dtype=float)
                else:
                    # Last resort: use input as-is if it's array-like
                    point_coords = np.array(picked_input, dtype=float)
            except Exception:
                point_coords = np.array(picked_input, dtype=float) if hasattr(picked_input, '__len__') else None
        
        if point_coords is None:
            print(f"Warning: Could not extract valid coordinates from input: {picked_input}")
            return
        
        # Ensure it's a 1D array of 3 elements
        if point_coords.ndim > 1:
            point_coords = point_coords.flatten()[:3]
        if len(point_coords) != 3:
            print(f"Warning: Invalid coordinate shape: {point_coords.shape}")
            return
        
        # 2. Project to mesh surface to get accurate point
        try:
            closest_points, dists, vertex_indices = trimesh.proximity.closest_point(
                self.trimesh_mesh, [point_coords]
            )
            point_coords = np.array(closest_points[0], dtype=float)
            vertex_idx = int(vertex_indices[0])
            dist_to_mesh = float(dists[0])  # Already in cm
            
            if dist_to_mesh > 10.0:
                print(f"Info: Point projected to mesh surface ({dist_to_mesh:.1f} cm away)")
        except Exception as e:
            print(f"Warning: Error projecting to mesh: {e}, using point as-is")
            # Fallback: find closest vertex
            vertex_distances = np.linalg.norm(self.mesh.points - point_coords, axis=1)
            vertex_idx = int(np.argmin(vertex_distances))
            point_coords = np.array(self.mesh.points[vertex_idx], dtype=float)
        
        # 3. Append the CLEAN coordinate to the list (always XYZ, never index)
        if self.shoulder_landmark_points is None:
            self.shoulder_landmark_points = []
        if self.shoulder_landmark_vertex_ids is None:
            self.shoulder_landmark_vertex_ids = []
            
        self.shoulder_landmark_points.append(point_coords.copy())
        self.shoulder_landmark_vertex_ids.append(vertex_idx)
        print(f"✓ Point {len(self.shoulder_landmark_points)} Added: {point_coords}")
        
        # 4. Validate point is reasonable relative to existing points
        if len(self.shoulder_landmark_points) > 1:
            prev_point = np.array(self.shoulder_landmark_points[-2], dtype=float)
            dist_to_prev = np.linalg.norm(point_coords - prev_point)  # Already in cm
            
            # Shoulder width points should be within 60cm of each other
            if dist_to_prev > 60.0:
                print(f"Warning: Point is {dist_to_prev:.1f} cm from previous point. This seems too far for shoulder width.")
        
        # 5. Visual Feedback (Show the user where they clicked)
        sphere = pv.Sphere(radius=2.0, center=point_coords)
        point_num = len(self.shoulder_landmark_points)
        if point_num == 1:
            color = 'cyan'  # First point (left shoulder)
        elif point_num == 2:
            color = 'lightblue'  # Second point (left mid-left)
        elif point_num == 3:
            color = 'blue'  # Third point (left mid)
        elif point_num == 4:
            color = 'yellow'  # Fourth point (nape/neck base)
        elif point_num == 5:
            color = 'orange'  # Fifth point (right mid)
        elif point_num == 6:
            color = 'red'  # Sixth point (right mid-right)
        else:
            color = 'magenta'  # Seventh point (right shoulder)
        self.plotter.add_mesh(sphere, color=color, name=f"shoulder_pt_{point_num}", render=False)
        
        remaining = 7 - len(self.shoulder_landmark_points)
        if remaining > 0:
            if point_num == 1:
                side = "left mid-left (between left shoulder and left mid)"
            elif point_num == 2:
                side = "left mid (between left mid-left and nape)"
            elif point_num == 3:
                side = "nape of neck (back)"
            elif point_num == 4:
                side = "right mid (between nape and right mid-right)"
            elif point_num == 5:
                side = "right mid-right (between right mid and right shoulder)"
            elif point_num == 6:
                side = "right shoulder tip"
            else:
                side = "unknown"
            self.plotter.add_text(
                f"Shoulder width: {point_num}/7 (pick {side})",
                position='upper_left',
                font_size=13,
                color='yellow',
                name='status_msg',
                font='courier',
            )
            self.plotter.render()
            return
        
        # 6. All 7 points picked - calculate the "yoke" measurement
        self._build_yoke_measurement()
        self._active_pick_mode = None
        self.picking_enabled = False
        try:
            self.plotter.untrack_click_position(side='left')
        except Exception:
            pass
        self.plotter.add_text(
            "✓ Shoulder width (yoke) set. Measurement follows surface path across back.",
            position='upper_left',
            font_size=13,
            color='lightgreen',
            name='status_msg',
            font='courier',
        )
        self.update_measurement_visuals()
    
    def _build_yoke_measurement(self):
        """
        Calculate the "Yoke" measurement using 7 points for more accurate path:
        Left Shoulder → Left Mid-Left → Left Mid → Nape → Right Mid → Right Mid-Right → Right Shoulder.
        This creates a smooth path across the back following the surface.
        """
        try:
            # 1. Get our 7 clean points (guaranteed to be XYZ coordinates, not indices)
            if len(self.shoulder_landmark_points) < 7:
                print(f"Error: Need 7 points, got {len(self.shoulder_landmark_points)}")
                return
            
            p0 = np.array(self.shoulder_landmark_points[0], dtype=float)  # Left shoulder
            p1 = np.array(self.shoulder_landmark_points[1], dtype=float)  # Left mid-left
            p2 = np.array(self.shoulder_landmark_points[2], dtype=float)  # Left mid
            p3 = np.array(self.shoulder_landmark_points[3], dtype=float)  # Nape/Neck Base
            p4 = np.array(self.shoulder_landmark_points[4], dtype=float)  # Right mid
            p5 = np.array(self.shoulder_landmark_points[5], dtype=float)  # Right mid-right
            p6 = np.array(self.shoulder_landmark_points[6], dtype=float)  # Right shoulder
            
            # Validate points are arrays of 3 elements
            for i, pt in enumerate([p0, p1, p2, p3, p4, p5, p6]):
                if not isinstance(pt, np.ndarray) or len(pt) != 3:
                    print(f"Error: Point {i} is invalid: {pt} (type: {type(pt)})")
                    return
            
            print(f"Debug: Yoke points (7 points) - Left: {p0}, LeftMidLeft: {p1}, LeftMid: {p2}, Nape: {p3}, RightMid: {p4}, RightMidRight: {p5}, Right: {p6}")
            
            # 2. Find their Mesh Indices (for geodesic calculation)
            idx0 = self.mesh.find_closest_point(p0)
            idx1 = self.mesh.find_closest_point(p1)
            idx2 = self.mesh.find_closest_point(p2)
            idx3 = self.mesh.find_closest_point(p3)
            idx4 = self.mesh.find_closest_point(p4)
            idx5 = self.mesh.find_closest_point(p5)
            idx6 = self.mesh.find_closest_point(p6)
            
            print(f"Debug: Vertex indices - {idx0}, {idx1}, {idx2}, {idx3}, {idx4}, {idx5}, {idx6}")
            
            # 3. Calculate Geodesic (Surface Path) - 6 segments connecting 7 points
            # Segment 1: Left → Left Mid-Left
            path_01 = self.mesh.geodesic(idx0, idx1)
            # Segment 2: Left Mid-Left → Left Mid
            path_12 = self.mesh.geodesic(idx1, idx2)
            # Segment 3: Left Mid → Nape
            path_23 = self.mesh.geodesic(idx2, idx3)
            # Segment 4: Nape → Right Mid
            path_34 = self.mesh.geodesic(idx3, idx4)
            # Segment 5: Right Mid → Right Mid-Right
            path_45 = self.mesh.geodesic(idx4, idx5)
            # Segment 6: Right Mid-Right → Right
            path_56 = self.mesh.geodesic(idx5, idx6)
            
            # Check if all paths are valid
            if path_01 is None or path_12 is None or path_23 is None or path_34 is None or path_45 is None or path_56 is None:
                print("Warning: Geodesic calculation failed, using straight line")
                # Fallback to straight line through all 7 points
                combined_points = np.array([p0, p1, p2, p3, p4, p5, p6])
                full_path = pv.PolyData(combined_points)
                # Create line connectivity for 7 points
                full_path.lines = np.array([7, 0, 1, 2, 3, 4, 5, 6], dtype=np.int32)
                total_width = (np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + 
                             np.linalg.norm(p3 - p2) + np.linalg.norm(p4 - p3) +
                             np.linalg.norm(p5 - p4) + np.linalg.norm(p6 - p5))
            else:
                # 4. ACCURACY FIX: Snap path endpoints to exact picked points
                # The geodesic path goes from vertex to vertex, but we want it to pass
                # exactly through the picked points (where the spheres are)
                points_01 = path_01.points.copy()
                points_12 = path_12.points.copy()
                points_23 = path_23.points.copy()
                points_34 = path_34.points.copy()
                points_45 = path_45.points.copy()
                points_56 = path_56.points.copy()
                
                # Replace endpoints with exact picked points
                if len(points_01) > 0:
                    points_01[0] = p0  # Left shoulder
                    points_01[-1] = p1  # Left mid-left
                if len(points_12) > 0:
                    points_12[0] = p1  # Left mid-left (should match path_01 end)
                    points_12[-1] = p2  # Left mid
                if len(points_23) > 0:
                    points_23[0] = p2  # Left mid (should match path_12 end)
                    points_23[-1] = p3  # Nape
                if len(points_34) > 0:
                    points_34[0] = p3  # Nape (should match path_23 end)
                    points_34[-1] = p4  # Right mid
                if len(points_45) > 0:
                    points_45[0] = p4  # Right mid (should match path_34 end)
                    points_45[-1] = p5  # Right mid-right
                if len(points_56) > 0:
                    points_56[0] = p5  # Right mid-right (should match path_45 end)
                    points_56[-1] = p6  # Right shoulder
                
                # 5. Combine all segments into one smooth line
                # Remove duplicate points at segment boundaries
                combined_points = points_01
                if len(points_12) > 1:
                    combined_points = np.vstack([combined_points, points_12[1:]])
                else:
                    combined_points = np.vstack([combined_points, points_12])
                
                if len(points_23) > 1:
                    combined_points = np.vstack([combined_points, points_23[1:]])
                else:
                    combined_points = np.vstack([combined_points, points_23])
                
                if len(points_34) > 1:
                    combined_points = np.vstack([combined_points, points_34[1:]])
                else:
                    combined_points = np.vstack([combined_points, points_34])
                
                if len(points_45) > 1:
                    combined_points = np.vstack([combined_points, points_45[1:]])
                else:
                    combined_points = np.vstack([combined_points, points_45])
                
                if len(points_56) > 1:
                    combined_points = np.vstack([combined_points, points_56[1:]])
                else:
                    combined_points = np.vstack([combined_points, points_56])
                
                # Verify the path passes through all 7 exact points
                check_points = [p0, p1, p2, p3, p4, p5, p6]
                check_indices = [0]  # First point
                # Find approximate indices for intermediate points
                for i, check_pt in enumerate(check_points[1:], 1):
                    # Find the closest point in combined_points to this check point
                    distances = np.linalg.norm(combined_points - check_pt, axis=1)
                    closest_idx = np.argmin(distances)
                    check_indices.append(closest_idx)
                    if distances[closest_idx] > 0.1:  # More than 1mm off
                        print(f"Warning: Path point {i} is {distances[closest_idx]:.3f} cm from picked point, correcting...")
                        combined_points[closest_idx] = check_pt
                
                # Ensure first and last points are exact
                combined_points[0] = p0
                combined_points[-1] = p6
                
                # Create PolyData with proper lines array
                full_path = pv.PolyData(combined_points)
                n_points = len(combined_points)
                lines_array = np.empty(n_points + 1, dtype=np.int32)
                lines_array[0] = n_points
                lines_array[1:] = np.arange(n_points, dtype=np.int32)
                full_path.lines = lines_array
                
                # 6. Calculate Length using the snapped path (more accurate)
                # Since we snapped endpoints to exact picked points, we need to recalculate
                # the length using the actual combined_points array
                total_width = 0.0
                for i in range(len(combined_points) - 1):
                    segment_length = np.linalg.norm(combined_points[i+1] - combined_points[i])
                    total_width += segment_length
                
                # The length is already in cm (mesh is scaled to target height)
                print(f"Debug: Path length calculated from {len(combined_points)} points: {total_width:.2f} cm")
            
            # Validate the measurement is reasonable
            # Calculate straight-line distance through all 7 points
            straight_dist = (np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + 
                           np.linalg.norm(p3 - p2) + np.linalg.norm(p4 - p3) +
                           np.linalg.norm(p5 - p4) + np.linalg.norm(p6 - p5))
            if total_width < straight_dist * 0.95:
                print(f"Warning: Yoke path too short ({total_width:.1f} cm), using straight line")
                total_width = straight_dist
            elif total_width > straight_dist * 2.0:
                print(f"Warning: Yoke path too long ({total_width:.1f} cm), using straight line")
                total_width = straight_dist
            elif total_width > 100.0:
                print(f"Warning: Yoke path exceeds maximum ({total_width:.1f} cm), using straight line")
                total_width = straight_dist
            
            print(f"SHOULDER WIDTH (Yoke): {total_width:.2f} cm")
            
            # 6. Store the path and length
            self.shoulder_geodesic_path = full_path
            self.shoulder_path_length = total_width
            
            # 7. Render the Red Line (like reference image)
            try:
                if full_path.n_points > 2:
                    tube = full_path.tube(radius=0.6, capping=False)
                    if tube is not None and tube.n_points > 0:
                        self.plotter.add_mesh(
                            tube, 
                            color='red', 
                            name="vis_shoulder_width_yoke",
                            opacity=1.0,
                            render=False,
                            lighting=False
                        )
                    else:
                        # Fallback: draw line segments
                        for i in range(full_path.n_points - 1):
                            line = pv.Line(full_path.points[i], full_path.points[i+1])
                            self.plotter.add_mesh(line, color='red', line_width=5, 
                                                name=f"yoke_seg_{i}", render=False, lighting=False)
                else:
                    # Straight line
                    line = pv.Line(full_path.points[0], full_path.points[1])
                    self.plotter.add_mesh(line, color='red', line_width=6, 
                                        name="vis_shoulder_width_yoke", render=False, lighting=False)
            except Exception as e:
                print(f"Warning: Could not render yoke path: {e}")
                
        except Exception as e:
            print(f"Error calculating yoke: {e}")
            import traceback
            traceback.print_exc()
            # Reset points for next try
            self.shoulder_landmark_points = []
            self.shoulder_landmark_vertex_ids = []

    def setup_ui(self):
        # Set background
        self.plotter.background_color = 'black'
        
        # Add mesh
        self.plotter.add_mesh(self.mesh, color='lightgrey', opacity=0.7, show_edges=False, 
                             smooth_shading=True, specular=0.6, ambient=0.3, diffuse=0.7, render=False)

        # Add initial neck-zone hint overlay so the correct area is visually marked immediately
        self._update_neck_hint_overlay(render=False)
        
        # Organize sliders
        measurements_list = list(self.measurements.keys())
        
        torso_group = ['Head (A)', 'Neck (B)', 'Chest (D)', 'Waist (E)', 'Hip (F)']
        torso_group = [m for m in torso_group if m in measurements_list]
        
        arm_group = ['Wrist (G)', 'Bicep (H)', 'Forearm (I)']
        arm_group = [m for m in arm_group if m in measurements_list]
        
        leg_group = ['Thigh (L)', 'Calf (M)', 'Ankle (N)']
        leg_group = [m for m in leg_group if m in measurements_list]
        
        # Add sliders
        y_pos = 0.80
        for name in torso_group:
            code = name.split('(')[1].split(')')[0] if '(' in name else ''
            self.plotter.add_slider_widget(
                callback=self.create_slider_callback(name),
                rng=[-20, 20], 
                value=self.measurements[name]['offset'],
                title=code,
                pointa=(0.72, y_pos),
                pointb=(0.88, y_pos),
                style='modern',
                tube_width=0.01,
                slider_width=0.02
            )
            y_pos -= 0.09
        
        y_pos = 0.50
        for name in arm_group:
            code = name.split('(')[1].split(')')[0] if '(' in name else ''
            self.plotter.add_slider_widget(
                callback=self.create_slider_callback(name),
                rng=[-20, 20],
                value=self.measurements[name]['offset'],
                title=code,
                pointa=(0.72, y_pos),
                pointb=(0.88, y_pos),
                style='modern',
                tube_width=0.01,
                slider_width=0.02
            )
            y_pos -= 0.09
        
        y_pos = 0.50
        for name in leg_group:
            code = name.split('(')[1].split(')')[0] if '(' in name else ''
            self.plotter.add_slider_widget(
                callback=self.create_slider_callback(name),
                rng=[-20, 20],
                value=self.measurements[name]['offset'],
                title=code,
                pointa=(0.92, y_pos),
                pointb=(0.98, y_pos),
                style='modern',
                tube_width=0.01,
                slider_width=0.02
            )
            y_pos -= 0.09
        
        # Add controls text
        controls_text = (
            "INTERACTIVE CONTROLS:\n"
            "[P] Toggle Picking\n"
            "[C] Create Custom Ring\n"
            "[Z] Undo Last Point\n"
            "[B] Set Neck (B) via Geodesic\n"
            "[O] Set Shoulder Width (O) via Geodesic (pick 7 points)\n"
            "[R] Reset View\n\n"
            "SLIDERS:\n"
            "Adjust measurement height offset (cm)\n\n"
            "Note: Measurements use one side only:\n"
            "Left leg (L/M/N)\n"
            "Right arm (G/H/I)"
        )
        self.plotter.add_text(controls_text, position='upper_right', font_size=11, 
                            color='white', name='controls_text', font='courier')
        
        # Add keyboard shortcuts
        self.plotter.add_key_event('p', self.toggle_picking)
        self.plotter.add_key_event('P', self.toggle_picking)
        self.plotter.add_key_event('c', self.create_custom_ring)
        self.plotter.add_key_event('C', self.create_custom_ring)
        self.plotter.add_key_event('z', self.undo_last_point)
        self.plotter.add_key_event('Z', self.undo_last_point)
        self.plotter.add_key_event('b', self.start_neck_landmark_picking)
        self.plotter.add_key_event('B', self.start_neck_landmark_picking)
        self.plotter.add_key_event('o', self.start_shoulder_width_picking)
        self.plotter.add_key_event('O', self.start_shoulder_width_picking)
        self.plotter.add_key_event('r', lambda: self.plotter.reset_camera())
        self.plotter.add_key_event('R', lambda: self.plotter.reset_camera())

        try:
            self.update_measurement_visuals()
        except Exception as e:
            print(f"Warning: Error during initial visualization: {e}")
            import traceback
            traceback.print_exc()
        
        # Show plotter and keep it open
        self.plotter.show(interactive=True, auto_close=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive 3D Body Measurement Tool")
    parser.add_argument("--pkl", type=str, required=True, help="Path to Sam3D output pickle file")
    parser.add_argument("--height", type=float, default=173.0, help="Target height in cm")
    
    args = parser.parse_args()
    
    try:
        print("Initializing Interactive Body Measurer...")
        app = InteractiveBodyMeasurer(args.pkl, args.height)
        print("Application started successfully!")
    except Exception as e:
        print(f"ERROR: Failed to start application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

