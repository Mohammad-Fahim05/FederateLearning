import os
import sys

print("=== CHECKING WILDS DATASET INTEGRATION ===")
try:
    import wilds
    from wilds import get_dataset
    print("WILDS version:", wilds.__version__)
    print("Available WILDS datasets:", wilds.supported_datasets)
except Exception as e:
    print("Error importing WILDS:", e)
