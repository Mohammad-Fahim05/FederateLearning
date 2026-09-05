packages = ['torch', 'torchvision', 'wilds', 'opacus', 'scipy', 'sklearn', 'pandas', 'matplotlib', 'yaml', 'tqdm', 'h5py', 'PIL']

for p in packages:
    try:
        __import__(p)
        print(f"{p}: INSTALLED")
    except ImportError:
        print(f"{p}: NOT INSTALLED")
