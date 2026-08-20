# Dental Arch Retainer Generation Pipeline

ML-based pipeline for automatic generation of dental retainers from 3D tooth meshes. Uses deep learning (PointNet, DGCNN) to segment dental arches, predict rotation alignment, and compute trimming distances for flat, precision cuts.

## Pipeline Overview

**Stage 1: Angle Prediction**
- Predicts Euler angles (X, Y, Z rotations) to align arch geometry
- Ensures flat, level base after trimming

**Phase 1: Arch Segmentation (PointNet)**
- Classifies mesh vertices as upper/lower/discard
- Segments combined tooth mesh into separate upper and lower arches

**Phase 2: Distance Regression**
- Predicts 4 cutting distances: [Z-depth, X-left, X-right, Y-back]
- Applies bbox-based cuts for clean, flat trimming surfaces

## Setup

### Requirements
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- Git LFS (for large model/data files)
- 3D Slicer

### Installation

```bash
# Install Git LFS (one-time setup)
# Ubuntu/Debian: sudo apt-get install git-lfs
# macOS: brew install git-lfs
# Then: git lfs install

# Clone the repo
git clone https://github.com/YOUR_USERNAME/AILabProject.git
cd AILabProject

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
### File Conversion
Run the following command in the 3D Slicer Python Console using the path you put the project in (this could look different based on whether or not you are on windows, mac or linux.
```
exec(open('/path/to//project/slicerconversion.py').read())
```
### Training

```bash
python plane_based_training.py \
  --epochs 50 \
  --patience 15 \
  --lr 0.001 \
  --back-cuts 1
```

Expects training data structure:
```
/path/to/data/
├── Before/
│   └── scan*.stl
└── After/
    └── scan*upper_after.stl, scan*lower_after.stl
```

### Inference

```bash
python plane_based_inference.py \
  --input /path/to/combined_mesh.stl \
  --output /path/to/output/ \
  --checkpoint-dir ./checkpoints
```

Generates:
- `upper_retainer.stl` - Trimmed upper arch with flat base
- `lower_retainer.stl` - Trimmed lower arch with flat base

## Project Structure

```
.
├── plane_based_training.py      # Training pipeline
├── plane_based_inference.py     # Inference pipeline
├── requirements.txt              # Python dependencies
├── checkpoints/                  # Trained models (Git LFS)
│   ├── angle_predictor.pt
│   ├── arch_classifier.pt
│   └── distance_regressor.pt
└── data/                         # Training datasets (Git LFS)
    └── *.stl, *.npz files
```

## Key Features

- **Angle-first pipeline**: Rotates mesh before trimming for flat, level bases
- **Adaptive thresholding**: Smart vertex classification using percentile-based metrics
- **Plane-fitted cuts**: Uses SVD-based plane fitting for precise geometric cuts
- **GPU acceleration**: CUDA-optimized k-NN and point cloud operations
- **Batch processing**: Efficient chunking for large meshes

## Notes

- Models trained on dental arch data with ~1000+ scans
- Best results with 1000-point sampled meshes
- Trimming happens after alignment to ensure orthogonal cuts
- See `.gitattributes` for Git LFS configuration

## License

[Add your license here]

## Contact

For questions or issues, contact the development team.
