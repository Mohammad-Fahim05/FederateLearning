import os
import sys
import time
import zipfile
import shutil

source_dir = r"D:\FederateLearning\data\camelyon17_v1.0"
parent_dir = r"D:\FederateLearning\data"
zip_path = r"D:\FederateLearning\data\camelyon17_v1.0.zip"

print("==================================================")
print("   PHASE 7E: CREATING KAGGLE UPLOAD ARCHIVE")
print("==================================================")
print(f"Source Directory: {source_dir}")
print(f"Target ZIP Archive: {zip_path}")

total_d, used_d, free_d = shutil.disk_usage(parent_dir)
print(f"Initial Free Disk Space on D: {free_d / (1024**3):.2f} GB")

start_time = time.time()
files_added = 0
png_count = 0

print("\nArchiving files with ZIP_STORED (fast I/O without redundant PNG compression)...")

with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            full_path = os.path.join(root, f)
            arcname = os.path.relpath(full_path, parent_dir)
            zf.write(full_path, arcname=arcname)
            files_added += 1
            if f.endswith('.png'):
                png_count += 1
            if files_added % 50000 == 0:
                print(f"  Archived {files_added:,} files ({png_count:,} PNG patches)...")

elapsed = time.time() - start_time
zip_size = os.path.getsize(zip_path)

print("\n==================================================")
print("   ARCHIVE CREATION COMPLETED")
print("==================================================")
print(f"Time Elapsed: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
print(f"Archive Path: {zip_path}")
print(f"Archive Size: {zip_size / (1024**3):.3f} GB ({zip_size:,} bytes)")
print(f"Total Files in ZIP: {files_added:,}")
print(f"Total PNG Patches in ZIP: {png_count:,}")

# Verification Step
print("\nVerifying Archive Integrity & Contents...")
with zipfile.ZipFile(zip_path, 'r') as zf:
    infolist = zf.infolist()
    verified_files = len(infolist)
    verified_pngs = sum(1 for info in infolist if info.filename.endswith('.png'))
    has_metadata = any(info.filename.endswith('metadata.csv') for info in infolist)
    has_release = any(info.filename.endswith('RELEASE_v1.0.txt') for info in infolist)
    sample_entry = infolist[0].filename

print(f"Verified Archive File Count: {verified_files:,}")
print(f"Verified PNG Patch Count: {verified_pngs:,}")
print(f"Contains metadata.csv: {has_metadata}")
print(f"Contains RELEASE_v1.0.txt: {has_release}")
print(f"Sample Internal Path: {sample_entry}")

assert verified_files == 455956, f"Expected 455956 files, got {verified_files}"
assert verified_pngs == 455954, f"Expected 455954 PNG patches, got {verified_pngs}"
assert has_metadata, "metadata.csv is missing from the archive!"
assert has_release, "RELEASE_v1.0.txt is missing from the archive!"

total_d, used_d, final_free = shutil.disk_usage(parent_dir)
print(f"Remaining Free Disk Space on D: {final_free / (1024**3):.2f} GB")
print("\nARCHIVE VERIFICATION STATUS: 100% VERIFIED SUCCESSFUL!")
