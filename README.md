# Deep Histo Image AI (first stable version)

Deep Histo Image is an advanced platform designed to simplify the use of modern artificial intelligence technologies in digital pathology. It enables pathologists, researchers, and developers to efficiently analyze histological images using state-of-the-art deep learning methods.

**Platform Objectives:**
✔ Automate routine histological image analysis processes
✔ Make modern AI algorithms accessible to doctors and researchers
✔ Promote the development of digital pathology

## Project Structure

```folders
DeepHistoImageAI/
├── config/             # Configuration files
│   └── settings.ini    # Application settings
├── data/              # Application data and database
├── docs/              # Documentation
├── input_folder/      # Source images for analysis
├── logs/              # Application logs
├── models/            # Deep learning models storage
├── output_folder/     # Predicted masks output
├── src/               # Source code
│   ├── migrations/    # Database migrations
│   └── tests/         # Test files
├── requirements.txt   # Python dependencies
└── README.md         # Project documentation
```

## Setup

CUDA is required for GPU support. If you don't have it, you can use CPU instead.

Install CUDA Toolkit to operating system: CUDA Toolkit Downloads

<https://developer.nvidia.com/cuda-downloads>

usefull tool: nvidia-smi (check GPU status)
A command line utility to help manage and monitor NVIDIA GPU devices.
If cuda toolkit installed correctly, you can use nvidia-smi for check GPU status.

```bash
nvidia-smi
```

1.Create and activate virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

check if CUDA is installed in the python environment:

```python
>>> import torch
>>> torch.cuda.is_available()
False
```

if False, you need to install CUDA. Example for CUDA 12.1:

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
```

If True, you can use GPU  for inference.

2.Install dependencies:

```bash
pip install -r requirements.txt
```

3.Start program:

```bash
python main.py
```

or with windows:

```bash
start.bat
```

## Features

- Import and manage deep learning models
- Predict image masks using selected models
- Support for various image formats (`.tif`, `.jpeg`, `.png`)
- Configurable batch processing
- Progress tracking for predictions
- Model metadata management through SQLite database

## Configuration

The application settings can be modified in `config/settings.ini`:

- Input/output paths
- Device selection (CPU/CUDA)
- Batch size and workers
- Logging preferences

## Supported Image Formats

- Input: `.tif`, `.jpeg`, `.png`
- Output: `.tif` (masks)
