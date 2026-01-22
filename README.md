# Sam3D + MHR Body Measurement System

Extract 16 precise anthropometric measurements from a single photo using Sam3D and MHR (Momentum Human Rig).

## 🎯 Features

- **Single Image Input** → 16 body measurements
- **Sam3D** - State-of-the-art 3D body reconstruction from Meta
- **MHR** - High-fidelity parametric human body model
- **Colored 3D Visualization** - See measurement zones on the mesh
- **Gradio Web Interface** - Easy-to-use web UI
- **Height Calibration** - Scale measurements to target height

## 📊 Measurements Provided

| Code | Measurement | Code | Measurement |
|------|-------------|------|-------------|
| A | Head Circumference | I | Forearm Right Circumference |
| B | Neck Circumference | J | Arm Right Length |
| C | Shoulder to Crotch Height | K | Inside Leg Height |
| D | Chest Circumference | L | Thigh Left Circumference |
| E | Waist Circumference | M | Calf Left Circumference |
| F | Hip Circumference | N | Ankle Left Circumference |
| G | Wrist Right Circumference | O | Shoulder Breadth |
| H | Bicep Right Circumference | P | Height (target) |

## 🚀 Quick Start

### Prerequisites
- CUDA-capable GPU (recommended)
- Conda or Miniconda
- Pixi package manager

### Installation

1. **Clone Repository**
```bash
git clone https://github.com/safuaaannn/sam3d-mhr-measurements.git
cd sam3d-mhr-measurements
```

2. **Download Model Assets**

**MHR Assets:**
```bash
curl -OL https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip
unzip assets.zip
```

**Sam3D Repository:**
```bash
git clone https://github.com/facebookresearch/sam-3d-body.git
```

3. **Setup Environments**

**For MHR (using Pixi):**
```bash
pixi install
```

**For Sam3D (using Conda):**
```bash
conda create -n sam_3d_body python=3.11 -y
conda activate sam_3d_body
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core pyrootutils webdataset networkx==3.2.1 roma joblib huggingface_hub
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
pip install git+https://github.com/microsoft/MoGe.git
```

4. **Download Sam3D Model Checkpoint**
```bash
cd sam-3d-body
python download_model.py  # Requires HuggingFace token
cd ..
```

## 💻 Usage

### Method 1: Command Line (3 Steps)

**Step 1: Run Sam3D Inference**
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam_3d_body
python step1_sam3d_inference.py --image /path/to/your/image.jpg --output ./output
```

**Step 2: Calculate Measurements**
```bash
pixi run python step2_mhr_measurements.py --sam3d_output ./output/sam3d_output.pkl --height 173
```

**Step 3: Generate Colored Visualization**
```bash
pixi run python visualize_measurements.py --sam3d_output ./output/sam3d_output.pkl --output ./output/colored_mesh.ply
# Or using alternative argument names:
pixi run python visualize_measurements.py --pkl ./output/sam3d_output.pkl --out ./output/colored_mesh.ply
```

**Step 4: Interactive 3D Measurement Tool**
```bash
pixi run python interactive_measurements.py --pkl ./output/sam3d_output.pkl --height 173
```

### Method 2: Web Interface (Gradio)

```bash
bash launch_app.sh
```

Then open http://localhost:7860 in your browser.

Upload an image, set target height, and get instant results!

## 📁 Project Structure

```
sam3d-mhr-measurements/
├── step1_sam3d_inference.py    # Sam3D inference script
├── step2_mhr_measurements.py   # Measurement calculation
├── visualize_measurements.py   # 3D visualization generator
├── app_sam3d.py               # Gradio web interface
├── launch_app.sh              # Launch Gradio app
├── run_sam3d.sh              # Quick Sam3D runner
├── mhr/                      # MHR model code
├── assets/                   # MHR model files (download separately)
└── sam-3d-body/             # Sam3D repository (clone separately)
```

## 🎨 Visualization

The colored 3D mesh shows measurement zones:
- 🔴 **Red** - Head circumference
- 🟠 **Orange** - Neck circumference
- 🟡 **Yellow** - Chest circumference
- 🟢 **Green** - Waist circumference
- 🔵 **Cyan** - Hip circumference
- 🔵 **Blue** - Thigh circumference
- 🟣 **Purple** - Calf circumference

Open the `.ply` file in MeshLab or Blender to view.

## 🔧 Customization

### Change Target Height
```bash
# Default is 173 cm, change to any value
pixi run python step2_mhr_measurements.py --height 180
```

### Process Different Image
```bash
# Edit run_sam3d.sh or specify directly
python step1_sam3d_inference.py --image /path/to/new/image.jpg --output ./output
```

## 📝 Requirements

- Python 3.11+
- CUDA 12.1+ (for GPU acceleration)
- 16GB+ RAM
- 10GB+ disk space (for models)

## 🔍 Verification

Before using the system, verify your setup:

```bash
python verify_setup.py
```

This will check:
- ✓ Project structure and required files
- ✓ MHR assets and model files
- ✓ Python dependencies (Pixi environment)
- ✓ Sam3D repository and conda environment
- ✓ MHR module import and initialization

## 🐛 Troubleshooting

**CUDA Error:**
- Ensure you activate conda environment properly: `source ~/miniconda3/etc/profile.d/conda.sh`
- Don't use `conda run`, use manual activation

**Model Not Found:**
- Download MHR assets: `curl -OL https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip`
- Download Sam3D checkpoint via `download_model.py`

**Gradio Not Working:**
- Install opencv and gradio in pixi: `pixi add opencv gradio`

**Path Issues:**
- All scripts now use relative paths automatically
- Ensure you run scripts from the project root directory
- Use `verify_setup.py` to check configuration

## 📄 License

This project combines:
- **MHR** - Meta Platforms, Inc. (see MHR LICENSE)
- **Sam3D** - Meta Platforms, Inc. (see Sam3D LICENSE)
- **This Integration** - MIT License

## 🙏 Acknowledgments

- [MHR (Momentum Human Rig)](https://github.com/facebookresearch/MHR) by Meta
- [Sam3D Body](https://github.com/facebookresearch/sam-3d-body) by Meta
- Built with PyTorch, Gradio, and Trimesh

## 📧 Contact

For issues and questions, please open a GitHub issue.

---

**⭐ Star this repo if you find it useful!**
