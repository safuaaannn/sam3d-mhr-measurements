#!/usr/bin/env python3
"""
Complete Gradio Interface for Sam3D + MHR Body Measurement System
This version uses subprocess to call Sam3D and MHR separately
"""
import gradio as gr
import cv2
import numpy as np
import subprocess
import pickle
import tempfile
import os
from pathlib import Path

def process_image(image, target_height):
    """
    Complete pipeline using subprocess calls
    """
    try:
        # Save uploaded image to temp file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_image_path = f.name
            cv2.imwrite(temp_image_path, image)
        
        # Create temp output directory
        temp_output_dir = tempfile.mkdtemp()
        
        # Step 1: Run Sam3D inference (using bash script with proper conda activation)
        print("Running Sam3D inference...")
        
        # Create temporary bash script for Sam3D
        sam3d_script = f"""#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam_3d_body
python /home/pj/Desktop/MHR/step1_sam3d_inference.py --image {temp_image_path} --output {temp_output_dir}
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            script_path = f.name
            f.write(sam3d_script)
        
        os.chmod(script_path, 0o755)
        
        result = subprocess.run(
            ['bash', script_path],
            capture_output=True,
            text=True
        )
        
        os.unlink(script_path)
        
        if result.returncode != 0:
            error_msg = result.stderr[:200] if result.stderr else "Unknown error"
            return [[f"❌ Sam3D Error: {error_msg}"]], None, None
        
        # Check if Sam3D output exists
        sam3d_output_path = os.path.join(temp_output_dir, "sam3d_output.pkl")
        if not os.path.exists(sam3d_output_path):
            return [["❌", "Sam3D failed to generate output", ""]], None, None
        
        # Load Sam3D visualization
        sam3d_vis_path = os.path.join(temp_output_dir, f"{Path(temp_image_path).stem}_sam3d_result.jpg")
        sam3d_vis = cv2.imread(sam3d_vis_path) if os.path.exists(sam3d_vis_path) else None
        
        # Step 2: Run MHR measurements (in pixi environment)
        print("Calculating measurements...")
        mhr_cmd = f"""
cd /home/pj/Desktop/MHR
/home/pj/.pixi/bin/pixi run python step2_mhr_measurements.py --sam3d_output {sam3d_output_path} --height {target_height}
"""
        result = subprocess.run(
            mhr_cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            error_msg = result.stderr[:200] if result.stderr else "Unknown error"
            return [[f"❌ MHR Error: {error_msg}"]], sam3d_vis, None
        
        # Parse measurements from output
        measurements_text = result.stdout
        table_data = []
        
        # Extract measurements from the output
        lines = measurements_text.split('\n')
        in_measurements = False
        for line in lines:
            if 'BODY MEASUREMENTS' in line:
                in_measurements = True
                continue
            if in_measurements and ' - ' in line and '...' in line:
                # Format: "A - Head Circumference......................  57.58 cm"
                parts = line.split(' - ', 1)  # Split only on first ' - '
                if len(parts) == 2:
                    code = parts[0].strip()
                    # Remove dots and extract name and value
                    rest = parts[1].replace('.', ' ').split()
                    if len(rest) >= 3:
                        # Name is everything except last 2 items (value + "cm")
                        name = ' '.join(rest[:-2])
                        value = f"{rest[-2]} {rest[-1]}"  # "57.58 cm"
                        table_data.append([code, name, value])
        
        # Step 3: Generate colored visualization (in pixi environment)
        print("Creating colored visualization...")
        vis_cmd = f"""
cd /home/pj/Desktop/MHR
/home/pj/.pixi/bin/pixi run python visualize_measurements.py --sam3d_output {sam3d_output_path} --output {temp_output_dir}/measurement_visualization.ply
"""
        subprocess.run(vis_cmd, shell=True, executable='/bin/bash')
        
        colored_mesh_path = os.path.join(temp_output_dir, "measurement_visualization.ply")
        
        # Cleanup temp image
        os.unlink(temp_image_path)
        
        return table_data, sam3d_vis, colored_mesh_path if os.path.exists(colored_mesh_path) else None
        
    except Exception as e:
        import traceback
        error_msg = str(e)[:200]
        return [[f"❌ Error: {error_msg}"]], None, None

# Create Gradio interface
with gr.Blocks(title="Advanced Body Measurement System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧍 Advanced Body Measurement System")
    gr.Markdown("### OpenPose → Sam3D → MHR Pipeline")
    gr.Markdown("Upload a photo of a person in upright position to get detailed anthropometric measurements.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Upload Person Photo")
            image_input = gr.Image(label="Input Image", type="numpy")
            height_input = gr.Slider(
                minimum=150, 
                maximum=220, 
                value=173, 
                step=1,
                label="Target Height (cm)"
            )
            process_btn = gr.Button("🔬 Process Image", variant="primary", size="lg")
            
            gr.Markdown("### 📋 Tips for Best Results:")
            gr.Markdown("""
            - ✅ Use a clear, well-lit photo
            - ✅ Person should be in upright position
            - ✅ Full body visible (head to feet)
            - ✅ Minimal clothing occlusion
            - ✅ Neutral background preferred
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Body Measurements")
            measurements_table = gr.Dataframe(
                headers=["Code", "Measurement", "Value"],
                label="Measurements (Scaled to Target Height)",
                interactive=False
            )
            
            gr.Markdown("### 🎨 Color Legend")
            gr.Markdown("""
            - 🔴 **Red** - Head circumference (A)
            - 🟠 **Orange** - Neck circumference (B)
            - 🟡 **Yellow** - Chest circumference (D)
            - 🟢 **Green** - Waist circumference (E)
            - 🔵 **Cyan** - Hip circumference (F)
            - 🔵 **Blue** - Thigh circumference (L)
            - 🟣 **Purple** - Calf circumference (M)
            - ⚫ **Spheres** - Key joint locations
            """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎭 Sam3D Body Reconstruction")
            sam3d_output = gr.Image(label="Sam3D Visualization")
        
        with gr.Column():
            gr.Markdown("### 📐 3D Model Download")
            colored_mesh_output = gr.File(label="Download 3D Mesh with Measurement Zones (.ply)")
            gr.Markdown("*Open the .ply file in MeshLab or Blender to view the 3D visualization!*")
    
    # Process button click
    process_btn.click(
        fn=process_image,
        inputs=[image_input, height_input],
        outputs=[measurements_table, sam3d_output, colored_mesh_output]
    )
    
    gr.Markdown("""
    ---
    ### 🔧 Technical Details
    
    **Pipeline Components:**
    1. **Detectron2** - Detects 2D keypoints (body, face, hands)
    2. **Sam3D** - Meta's state-of-the-art 3D human mesh recovery
    3. **MHR** - Momentum Human Rig parametric body model
    4. **Measurements** - Precise anthropometric measurements
    
    **Measurements Provided:**
    - Head, Neck, Chest, Waist, Hip circumferences
    - Bicep, Forearm, Wrist circumferences  
    - Thigh, Calf, Ankle circumferences
    - Arm length, Inside leg height, Shoulder breadth
    - Full body height (used as reference)
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
