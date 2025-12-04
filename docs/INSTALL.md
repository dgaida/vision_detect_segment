# Installation Guide

Complete installation guide for the `vision_detect_segment` package.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Quick Installation](#quick-installation)
- [Detailed Installation Steps](#detailed-installation-steps)
- [Model-Specific Dependencies](#model-specific-dependencies)
- [Redis Installation](#redis-installation)
- [GPU Setup](#gpu-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Optional Components](#optional-components)

---

## System Requirements

### Minimum Requirements

- **Operating System:** Linux, macOS, or Windows 10/11
- **Python:** 3.8 or higher
- **RAM:** 8 GB minimum (16 GB recommended)
- **Storage:** 10 GB free space (for models and dependencies)
- **Redis Server:** 5.0 or higher

### GPU Requirements (Optional but Recommended)

- **NVIDIA GPU** with CUDA support
- **CUDA:** 11.8 or higher
- **cuDNN:** Compatible with your CUDA version
- **VRAM:** 4 GB minimum (8+ GB recommended)

---

## Quick Installation

### For Users (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/dgaida/vision_detect_segment.git
cd vision_detect_segment

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install the package
pip install -e .

# 4. Install Redis (if not already installed)
# See Redis Installation section below
```

### For Developers

```bash
# Clone and install with development dependencies
git clone https://github.com/dgaida/vision_detect_segment.git
cd vision_detect_segment

python -m venv venv
source venv/bin/activate

pip install -e .
pip install -r requirements-test.txt

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

---

## Detailed Installation Steps

### Step 1: Python Environment Setup

#### Using venv (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

#### Using conda

```bash
# Create conda environment
conda create -n vision_detect python=3.11
conda activate vision_detect
```

### Step 2: Install Core Dependencies

```bash
# Update pip
pip install --upgrade pip

# Install PyTorch (GPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Or for CPU-only:
pip install torch torchvision
```

### Step 3: Install vision_detect_segment

```bash
# Install from source (editable mode)
pip install -e .

# Or install specific models (see Model-Specific Dependencies)
```

### Step 4: Install redis_robot_comm

The package depends on `redis_robot_comm` for Redis communication:

```bash
pip install git+https://github.com/dgaida/redis_robot_comm.git
```

---

## Model-Specific Dependencies

Different detection models require different dependencies. Install only what you need.

### OWL-V2 (Open-Vocabulary Detection)

```bash
pip install transformers>=4.30.0
pip install timm  # For vision transformers
```

**Model Download:**
Models are downloaded automatically on first use (~400 MB).

### YOLO-World (Real-Time Detection)

```bash
pip install ultralytics>=8.0.0
```

**Model Download:**
```bash
# Download model manually (optional)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-worldv2.pt
```

### YOLOE (Unified Detection & Segmentation)

```bash
pip install ultralytics>=8.3.0
```

**Important:** YOLOE requires Ultralytics 8.3.0 or higher.

**Model Download:**
Models are downloaded automatically on first use:
- `yoloe-11s-seg.pt` (~25 MB)
- `yoloe-11m-seg.pt` (~52 MB)
- `yoloe-11l-seg.pt` (~66 MB)
- `yoloe-v8s-seg.pt` (~23 MB)
- `yoloe-v8m-seg.pt` (~50 MB)
- `yoloe-v8l-seg.pt` (~64 MB)

Prompt-free variants (with `-pf` suffix) are also available.

### Grounding-DINO (Text-Guided Detection)

```bash
pip install transformers>=4.30.0
```

### Segmentation Models

#### FastSAM (Included with Ultralytics)

```bash
pip install ultralytics
```

#### SAM2 (High-Quality Segmentation)

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

Or with pip:
```bash
pip install segment-anything-2
```

---

## Redis Installation

Redis is required for image streaming and detection result publishing.

### Using Docker (Recommended)

```bash
# Pull and run Redis
docker pull redis:alpine
docker run -d -p 6379:6379 --name redis-vision redis:alpine

# Verify Redis is running
docker ps | grep redis
```

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install redis-server

# Start Redis
sudo systemctl start redis-server

# Enable on boot
sudo systemctl enable redis-server

# Verify
redis-cli ping  # Should return "PONG"
```

### macOS

```bash
# Using Homebrew
brew install redis

# Start Redis
brew services start redis

# Verify
redis-cli ping
```

### Windows

```bash
# Using WSL2 (recommended)
wsl --install
# Then follow Ubuntu instructions

# Or download Windows port:
# https://github.com/microsoftarchive/redis/releases
```

### Redis Configuration (Optional)

Create `redis.conf` for custom configuration:

```conf
# Basic configuration
bind 0.0.0.0
port 6379
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000
```

Start Redis with custom config:
```bash
redis-server /path/to/redis.conf
```

---

## GPU Setup

### CUDA Installation

#### Linux

```bash
# Check current CUDA version
nvcc --version

# Install CUDA 11.8 (example for Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

#### Windows

1. Download CUDA Toolkit from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
2. Run installer and follow prompts
3. Add CUDA to PATH:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
   ```

### Verify GPU Setup

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

Expected output (with GPU):
```
CUDA available: True
CUDA version: 11.8
GPU count: 1
GPU name: NVIDIA GeForce RTX 3090
```

---

## Verification

### Test Installation

```bash
# Run test script
python main.py
```

### Test Individual Components

```python
# Test imports
from vision_detect_segment import VisualCortex, get_default_config
from redis_robot_comm import RedisImageStreamer

print("✓ Imports successful")

# Test GPU
import torch
if torch.cuda.is_available():
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ GPU not available, using CPU")

# Test Redis connection
try:
    from redis_robot_comm import RedisImageStreamer
    streamer = RedisImageStreamer()
    print("✓ Redis connection successful")
except Exception as e:
    print(f"✗ Redis connection failed: {e}")

# Test model loading
try:
    config = get_default_config("yoloe-11l")
    cortex = VisualCortex("yoloe-11l", device="auto", config=config)
    print("✓ Model loading successful")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_config.py -v

# Run with coverage
pytest tests/ --cov=vision_detect_segment --cov-report=html
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# Use smaller model
cortex = VisualCortex("yoloe-11s", device="cuda")  # Instead of 11l

# Use CPU
cortex = VisualCortex("yoloe-11l", device="cpu")

# Clear GPU cache
cortex.clear_cache()

# Reduce image size
image = cv2.resize(image, (640, 480))
```

#### 2. Redis Connection Failed

**Symptoms:**
```
RedisConnectionError: Redis connection failed on localhost:6379
```

**Solutions:**
```bash
# Check if Redis is running
redis-cli ping

# Start Redis
# Docker:
docker start redis-vision

# Linux:
sudo systemctl start redis-server

# macOS:
brew services start redis

# Check Redis logs
docker logs redis-vision  # Docker
tail -f /var/log/redis/redis-server.log  # Linux
```

#### 3. Model Download Fails

**Symptoms:**
```
ModelLoadError: Failed to load model 'yoloe-11l'
```

**Solutions:**
```bash
# Download model manually
mkdir -p ~/.cache/ultralytics
cd ~/.cache/ultralytics
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yoloe-11l-seg.pt

# Check internet connection
ping github.com

# Use HTTP proxy (if needed)
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

#### 4. Import Errors

**Symptoms:**
```
ImportError: No module named 'transformers'
```

**Solutions:**
```bash
# Install missing dependencies
pip install transformers  # For OWL-V2/Grounding-DINO
pip install ultralytics>=8.3.0  # For YOLO-World/YOLOE

# Verify installation
pip list | grep transformers
pip list | grep ultralytics

# Reinstall if needed
pip install --force-reinstall transformers
```

#### 5. YOLOE Not Available

**Symptoms:**
```
DependencyError: ultralytics (with YOLOE support)
```

**Solutions:**
```bash
# Upgrade Ultralytics to 8.3.0+
pip install -U ultralytics>=8.3.0

# Verify version
python -c "import ultralytics; print(ultralytics.__version__)"

# Should show 8.3.0 or higher
```

#### 6. Windows-Specific Issues

**Long Path Names:**
```bash
# Enable long paths in Windows
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

**Missing Visual C++ Redistributables:**
```
Download and install:
https://aka.ms/vs/17/release/vc_redist.x64.exe
```

---

## Optional Components

### Development Tools

```bash
# Code formatting
pip install black ruff

# Type checking
pip install mypy

# Documentation
pip install sphinx sphinx-rtd-theme

# Jupyter notebooks
pip install jupyter notebook
```

### Visualization Tools

```bash
# Advanced visualization
pip install matplotlib seaborn plotly

# Video processing
pip install moviepy

# 3D visualization
pip install open3d
```

### Additional Utilities

```bash
# Performance profiling
pip install py-spy memory-profiler

# Progress bars
pip install tqdm

# Configuration management
pip install pydantic python-dotenv
```

---

## Post-Installation Configuration

### Environment Variables

Create `.env` file in project root:

```bash
# Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Model cache directory
MODEL_CACHE_DIR=~/.cache/vision_models

# GPU settings
CUDA_VISIBLE_DEVICES=0

# Logging
LOG_LEVEL=INFO
```

### Configuration File

Create `config.yaml`:

```yaml
model:
  name: yoloe-11l
  confidence_threshold: 0.25
  device: cuda

redis:
  host: localhost
  port: 6379
  stream_name: robot_camera

annotation:
  show_labels: true
  show_confidence: true
  resize_scale_factor: 2.0
```

---

## Updating

### Update vision_detect_segment

```bash
cd vision_detect_segment
git pull origin main
pip install -e . --upgrade
```

### Update Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Update Models

Models are cached. To force re-download:

```bash
# Clear model cache
rm -rf ~/.cache/ultralytics
rm -rf ~/.cache/torch/hub

# Models will re-download on next use
```

---

## Uninstallation

### Remove Package

```bash
pip uninstall vision_detect_segment
pip uninstall redis_robot_comm
```

### Clean Up

```bash
# Remove virtual environment
rm -rf venv

# Remove model cache
rm -rf ~/.cache/ultralytics
rm -rf ~/.cache/torch/hub

# Stop and remove Redis container
docker stop redis-vision
docker rm redis-vision
```

---

## Additional Resources

- **Documentation:** [docs/README.md](README.md)
- **API Reference:** [docs/api.md](api.md)
- **Workflow Guide:** [docs/vision_workflow_doc.md](vision_workflow_doc.md)
- **GitHub Repository:** https://github.com/dgaida/vision_detect_segment
- **Issue Tracker:** https://github.com/dgaida/vision_detect_segment/issues

---

## Getting Help

If you encounter issues not covered in this guide:

1. Check the [Troubleshooting](#troubleshooting) section
2. Search [existing issues](https://github.com/dgaida/vision_detect_segment/issues)
3. Create a new issue with:
   - Your system information (OS, Python version, GPU)
   - Complete error message and stack trace
   - Steps to reproduce the problem
   - Output of `pip list`

---

## License

MIT License - see [LICENSE](../LICENSE) file for details.

---

**Last Updated:** December 2025  
**Package Version:** 0.2.0
