import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from PIL import Image
from src.data.preprocessing import get_transforms

class SyntheticCamelyon17Dataset(Dataset):
    """
    Synthetic Camelyon17 benchmark dataset generator.
    Simulates 5 hospital centers (Centers 0, 1, 2, 3, 4), WSI slide IDs,
    and 96x96 tissue patches with realistic domain stain shifts and class imbalance.
    Used for reproducible pipeline verification, unit testing, and fast experiment iterations.
    """
    def __init__(self, num_samples_per_center=500, transform=None, seed=42):
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.transform = transform
        
        self.samples = []
        self.labels = []
        self.center_ids = []
        self.slide_ids = []
        
        # 5 Hospitals, 10 slides per hospital = 50 WSIs total
        for center in range(5):
            # Center-specific stain color shift (domain shift)
            stain_shift = torch.tensor([center * 0.08, (4 - center) * 0.05, center * 0.03]).view(3, 1, 1)
            
            for slide in range(10):
                slide_id = center * 10 + slide
                num_patches = num_samples_per_center // 10
                
                for p in range(num_patches):
                    label = 1 if (p % 2 == 0) else 0
                    
                    # Generate synthetic 96x96 tissue patch pattern
                    base_pattern = torch.randn(3, 96, 96) * 0.2 + 0.7
                    if label == 1:
                        # Tumor patch has higher central nuclear density
                        base_pattern[:, 32:64, 32:64] -= 0.35
                        
                    patch = torch.clamp(base_pattern + stain_shift, 0.0, 1.0)
                    patch_pil = T_to_pil(patch)
                    
                    self.samples.append(patch_pil)
                    self.labels.append(label)
                    self.center_ids.append(center)
                    self.slide_ids.append(slide_id)
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img = self.samples[idx]
        label = self.labels[idx]
        center_id = self.center_ids[idx]
        slide_id = self.slide_ids[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label, center_id, slide_id

def T_to_pil(tensor_img):
    img_np = (tensor_img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(img_np)

class Camelyon17HospitalDataset(Dataset):
    """
    Unified PyTorch Dataset for Camelyon17.
    Supports WILDS Camelyon17 dataset or Synthetic benchmark fallback.
    """
    def __init__(self, root_dir="./data", download=False, use_synthetic=False, transform=None, seed=42):
        if transform is None:
            transform = get_transforms(is_train=False)
        self.transform = transform
        self.use_synthetic = use_synthetic
        
        if not use_synthetic:
            # Smart root_dir resolution for Kaggle and local paths
            resolved_root = root_dir
            candidate_roots = [
                root_dir,
                "/kaggle/input/datasets/mohdfam/camelyon17-wilds",
                "/kaggle/input/camelyon17-wilds",
                os.environ.get("CAMELYON17_DATA_DIR", ""),
                "./data"
            ]
            for cand in candidate_roots:
                if cand and os.path.exists(cand):
                    if os.path.exists(os.path.join(cand, "camelyon17_v1.0", "metadata.csv")):
                        resolved_root = cand
                        break
                    elif os.path.exists(os.path.join(cand, "metadata.csv")):
                        # If cand directly contains metadata.csv, parent is root_dir if folder is camelyon17_v1.0
                        parent = os.path.dirname(os.path.abspath(cand))
                        if os.path.basename(os.path.abspath(cand)) == "camelyon17_v1.0":
                            resolved_root = parent
                            break
                        resolved_root = cand
                        break

            try:
                import wilds
                self.wilds_dataset = wilds.get_dataset(dataset='camelyon17', download=download, root_dir=resolved_root)
                self.data = self.wilds_dataset
                self.is_wilds = True
                # Identify center/hospital field name in WILDS metadata
                if 'center' in self.wilds_dataset.metadata_fields:
                    self.center_field = 'center'
                elif 'hospital' in self.wilds_dataset.metadata_fields:
                    self.center_field = 'hospital'
                else:
                    self.center_field = self.wilds_dataset.metadata_fields[0]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load real WILDS Camelyon17 dataset from root_dir='{resolved_root}' (original='{root_dir}') with download={download}. "
                    f"Original error: {e}"
                ) from e
        else:
            self.is_wilds = False
            self.synthetic_ds = SyntheticCamelyon17Dataset(num_samples_per_center=500, transform=transform, seed=seed)

        self.cached_images = None
        self.cached_labels = None
        self.cached_centers = None
        self.cached_slides = None
        self.idx_to_cache_pos = None

    def preload_train_cache(self, train_indices, max_ram_gb=10.0):
        """
        Preloads training patches into system RAM as compact uint8 NumPy arrays
        to eliminate disk I/O and PNG decoding bottlenecks during federated training.
        Only caches specified training indices (e.g. Centers 0, 1, 2).
        """
        n_samples = len(train_indices)
        if n_samples == 0:
            return

        est_ram_gb = (n_samples * 96 * 96 * 3) / (1024 ** 3)
        print(f"[RAM Cache] Preloading {n_samples:,} training samples into system RAM (Est. {est_ram_gb:.2f} GB)...")
        
        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / (1024 ** 3)
            print(f"[RAM Cache] Available System RAM: {avail_gb:.2f} GB (Threshold: {max_ram_gb:.2f} GB)")
            if est_ram_gb > avail_gb or est_ram_gb > max_ram_gb:
                print(f"[RAM Cache] WARNING: Insufficient RAM for full caching ({est_ram_gb:.2f} GB required > {avail_gb:.2f} GB available). Disabling RAM cache.")
                return False
        except Exception:
            pass  # Fallback to direct allocation with try/except

        try:
            cached_imgs = np.empty((n_samples, 96, 96, 3), dtype=np.uint8)
            cached_lbls = np.empty(n_samples, dtype=np.int64)
            cached_cnts = np.empty(n_samples, dtype=np.int64)
            cached_slds = np.empty(n_samples, dtype=np.int64)
            idx_map = {}

            center_col = self.wilds_dataset.metadata_fields.index(self.center_field) if self.is_wilds else 0
            slide_col = self.wilds_dataset.metadata_fields.index('slide') if (self.is_wilds and 'slide' in self.wilds_dataset.metadata_fields) else 1

            for pos, idx in enumerate(train_indices):
                if self.is_wilds:
                    pil_img, y, meta = self.wilds_dataset[idx]
                    cached_imgs[pos] = np.array(pil_img)
                    cached_lbls[pos] = y.item()
                    cached_cnts[pos] = meta[center_col].item()
                    cached_slds[pos] = meta[slide_col].item()
                else:
                    pil_img = self.synthetic_ds.samples[idx]
                    cached_imgs[pos] = np.array(pil_img)
                    cached_lbls[pos] = self.synthetic_ds.labels[idx]
                    cached_cnts[pos] = self.synthetic_ds.center_ids[idx]
                    cached_slds[pos] = self.synthetic_ds.slide_ids[idx]
                    
                idx_map[idx] = pos

            self.cached_images = cached_imgs
            self.cached_labels = cached_lbls
            self.cached_centers = cached_cnts
            self.cached_slides = cached_slds
            self.idx_to_cache_pos = idx_map
            print(f"[RAM Cache] Successfully cached {n_samples:,} training samples in RAM ({cached_imgs.nbytes / (1024**3):.2f} GB).")
            return True
        except MemoryError:
            print("[RAM Cache] MemoryError during preloading. Disabling RAM cache and falling back to disk loading.")
            self.cached_images = None
            self.idx_to_cache_pos = None
            return False
            
    def get_center_subsets(self, centers):
        """
        Extracts patch indices belonging to specific hospital center IDs.
        """
        if self.is_wilds:
            all_metadata = self.wilds_dataset.metadata_array
            center_col = self.wilds_dataset.metadata_fields.index(self.center_field)
            indices = [i for i, meta in enumerate(all_metadata) if meta[center_col].item() in centers]
        else:
            indices = [i for i, c in enumerate(self.synthetic_ds.center_ids) if c in centers]
            
        return indices
        
    def __len__(self):
        if self.is_wilds:
            return len(self.wilds_dataset)
        return len(self.synthetic_ds)
        
    def __getitem__(self, idx):
        # Fast RAM cache retrieval if sample was preloaded
        if self.cached_images is not None and self.idx_to_cache_pos is not None and idx in self.idx_to_cache_pos:
            pos = self.idx_to_cache_pos[idx]
            img_np = self.cached_images[pos]
            img = Image.fromarray(img_np)
            y = int(self.cached_labels[pos])
            center_id = int(self.cached_centers[pos])
            slide_id = int(self.cached_slides[pos])
            if self.transform:
                img = self.transform(img)
            return img, y, center_id, slide_id

        # Disk loading fallback
        if self.is_wilds:
            x, y, metadata = self.wilds_dataset[idx]
            center_col = self.wilds_dataset.metadata_fields.index(self.center_field)
            center_id = metadata[center_col].item()
            slide_col = self.wilds_dataset.metadata_fields.index('slide') if 'slide' in self.wilds_dataset.metadata_fields else 1
            slide_id = metadata[slide_col].item()
            if self.transform:
                x = self.transform(x)
            return x, y.item(), center_id, slide_id
        else:
            return self.synthetic_ds[idx]
