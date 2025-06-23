import glob
import os
import subprocess
import sys
import shutil

import torch

APP_NAME = "DeepHistoImageAI"
ENTRY_POINT = "main.py"
OUTPUT_DIR = "build"
ICON_PATH = "assets/icon.ico"

# Ensure empty directories contain placeholder files
REQUIRED_DIRS = ["data", "logs", "models", "input_folder", "output_folder"]
for directory in REQUIRED_DIRS:
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, ".keep"), "w") as f:
        f.write("")

INCLUDE_DATA_FILES = [
    "config/settings.ini=config/settings.ini",
    "data/.keep=data/.keep",
    "logs/.keep=logs/.keep",
    "models/.keep=models/.keep",
    "input_folder/.keep=input_folder/.keep",
    "output_folder/.keep=output_folder/.keep",
]
EXCLUDE_MODULES = ["XNNPACK.lib"]
EXCLUDE_DIRS = ["venv", "build", "src/tests"]

#  1. Find CUDA libraries
cuda_lib_paths = [
    os.path.join(os.path.dirname(torch.__file__), "lib"),  # PyTorch CUDA
    "C:/Program Files/NVIDIA Corporation/CUDA/lib/x64",
    # Windows (standard CUDA path)
    # "/usr/local/cuda/lib64",  # Linux
]

cuda_include_cmds = []
for path in cuda_lib_paths:
    if os.path.exists(path):
        print(f"Found CUDA libraries at {path}")
        cuda_include_cmds.append(f"--include-data-dir={path}=lib")

if not cuda_include_cmds:
    print("CUDA not found, build will be CPU-only.")

#  2. Include necessary `.dll/.so` files for PyTorch
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
torch_dlls = glob.glob(os.path.join(torch_lib_path, "*.dll")) + glob.glob(
    os.path.join(torch_lib_path, "*.so*")
)

torch_include_cmds = [f"--include-data-files={dll}=lib" for dll in torch_dlls]

command = [
    sys.executable,
    "-m",
    "nuitka",
    "--standalone",
    "--onefile",
    "--output-dir=" + OUTPUT_DIR,
    "--enable-plugin=pyqt6",
    "--include-package=torch",
    "--nofollow-import-to=" + ",".join(EXCLUDE_MODULES),
    "--follow-imports",
    "--assume-yes-for-downloads",
]

# Якщо потрібне включення CUDA бібліотек або бібліотек PyTorch, розкоментуй наступні рядки:
# command.extend(cuda_include_cmds)
# command.extend(torch_include_cmds)

for file in INCLUDE_DATA_FILES:
    command.append(f"--include-data-files={file}")

if os.path.exists(ICON_PATH):
    command.append(f"--windows-icon-from-ico={ICON_PATH}")

command.append(ENTRY_POINT)

print("Starting build with Nuitka...")
subprocess.run(command)

# Перейменування згенерованого файлу з main.exe на DeepHistoImageAI.exe
src_exe = os.path.join(OUTPUT_DIR, "main.exe")
dst_exe = os.path.join(OUTPUT_DIR, f"{APP_NAME}.exe")
if os.path.exists(src_exe):
    shutil.move(src_exe, dst_exe)
    print(f"Renamed {src_exe} to {dst_exe}")
else:
    print(f"File {src_exe} not found. Build might have failed or the entry point name has changed.")
