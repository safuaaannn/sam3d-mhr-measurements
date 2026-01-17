#!/bin/bash
# Run Sam3D inference on test3.jpeg

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam_3d_body

# Run Sam3D inference
python /home/pj/Desktop/MHR/step1_sam3d_inference.py --image /home/pj/Desktop/MHR/test3.jpeg --output /home/pj/Desktop/MHR/output
