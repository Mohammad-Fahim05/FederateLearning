import sys
import os
import platform
import shutil

print("=== SYSTEM ENVIRONMENT INSPECTION ===")
print("OS:", platform.platform())
print("Python Executable:", sys.executable)
print("Python Version:", sys.version)

try:
    import torch
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
        print("Device Count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}:", torch.cuda.get_device_name(i))
            print(f"Device {i} Memory:", round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2), "GB")
    else:
        print("CUDA device not detected. PyTorch will run on CPU.")
except ImportError:
    print("PyTorch: NOT INSTALLED")

try:
    import torchvision
    print("torchvision Version:", torchvision.__version__)
except ImportError:
    print("torchvision: NOT INSTALLED")

try:
    import wilds
    print("WILDS Version:", wilds.__version__)
except ImportError:
    print("WILDS: NOT INSTALLED")

try:
    import opacus
    print("Opacus Version:", opacus.__version__)
except ImportError:
    print("Opacus: NOT INSTALLED")

current_dir = os.path.abspath(".")
total, used, free = shutil.disk_usage(current_dir)
print(f"Disk ({current_dir}): Total = {total // (2**30)} GB, Free = {free // (2**30)} GB")

print("=== END INSPECTION ===")
