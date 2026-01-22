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
        
        # 5. Define ALL 16 Measurements
        self.measurements = {
            # Circumference measurements
            'Head (A)':    {'joint': 'r_eye',     'normal': [0, 1, 0], 'offset': 0.0, 'color': 'red', 'exclude_arms': False, 'type': 'circumference'},
            'Neck (B)':    {'joint': 'c_neck',   'normal': [0, 1, 0], 'offset': 0.0, 'color': 'orange', 'exclude_arms': False, 'type': 'circumference', 'midpoint': ['c_neck', 'c_head']},
            'Chest (D)':   {'joint': 'c_spine2', 'normal': [0, 1, 0], 'offset': 0.0, 'color': 'yellow', 'exclude_arms': True, 'type': 'circumference'},
            'Waist (E)':   {'joint': 'c_spine0', 'normal': [0, 1, 0], 'offset': 0.0, 'color': 'green', 'exclude_arms': True, 'type': 'circumference'},
            'Hip (F)':     {'joint': 'root',     'normal': [0, 1, 0], 'offset': 0.0, 'color': 'cyan', 'exclude_arms': True, 'type': 'circumference'},
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
            'Shoulder-Crotch (C)': {'start': 'c_spine3', 'end': 'root', 'color': 'white', 'type': 'length'},
            'Arm Length (J)': {'start': 'r_uparm', 'end': 'r_wrist', 'color': 'orange', 'type': 'length', 'segments': ['r_uparm', 'r_lowarm', 'r_wrist']},
            'Inside Leg (K)': {'start': 'root', 'end': 'min_y', 'color': 'magenta', 'type': 'length'},
            'Shoulder Width (O)': {'start': 'l_uparm', 'end': 'r_uparm', 'color': 'lightgray', 'type': 'length'},
            'Height (P)': {'start': 'max_y', 'end': 'min_y', 'color': 'gray', 'type': 'length'},
        }
        
        self.selected_points = []
        self.custom_measurements = []
        self.picking_enabled = False
        # Neck geodesic (B) override: user-defined landmarks on surface
        # Order recommended: back (nape) -> right -> front (suprasternal) -> left
        self._active_pick_mode = None  # None | "custom" | "neck"
        self.neck_landmark_vertex_ids = []
        self.neck_landmark_points = []  # list[np.ndarray], world/surface points for neck picking
        self.neck_geodesic_path = None  # pv.PolyData polyline
        
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

    def compute_length(self, start_joint, end_joint):
        """Compute length between two joints"""
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
                
                offset = data.get('offset', 0.0)
                normal = np.array(data['normal']) / np.linalg.norm(data['normal'])
                center = center + offset * normal
                
                # Neck (B): if user provided geodesic loop, use it instead of planar slice
                exclude_arms = data.get('exclude_arms', False)
                if name == 'Neck (B)' and self.neck_geodesic_path is not None and self.neck_geodesic_path.n_points > 2:
                    value = float(self.neck_geodesic_path.length)
                    slice_mesh = self.neck_geodesic_path
                    # Use red color for geodesic neck path to match reference
                    ring_color = 'red'
                else:
                    value, slice_mesh = self.compute_circumference(center, data['normal'], exclude_arms=exclude_arms)
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
            value = self.compute_length(data['start'], data['end'])
            start_pt = self.get_joint(data['start'])
            end_pt = self.get_joint(data['end'])
            line = pv.Line(start_pt, end_pt)
            self.plotter.add_mesh(line, color=data['color'], line_width=4, 
                                name=f"line_{name}", render=False, lighting=False)
            code = name.split('(')[1].split(')')[0] if '(' in name else ''
            label = f"{code}  {name.split('(')[0].strip():<12} {value:>6.1f} cm"
            panel_lines.append(label)
        
        if 'Shoulder Width (O)' in self.length_measurements:
            name = 'Shoulder Width (O)'
            data = self.length_measurements[name]
            value = self.compute_length(data['start'], data['end'])
            start_pt = self.get_joint(data['start'])
            end_pt = self.get_joint(data['end'])
            line = pv.Line(start_pt, end_pt)
            self.plotter.add_mesh(line, color=data['color'], line_width=4, 
                                name=f"line_{name}", render=False, lighting=False)
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
            self.plotter.add_text("✗ Error: Pick at least 3 points first!", position='upper_left', 
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

