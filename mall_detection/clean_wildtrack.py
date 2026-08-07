import os
import sys
import glob
from pathlib import Path

def print_banner():
    print("=" * 60)
    print("        WILDTRACK DATASET PRUNING & OPTIMIZATION TOOL")
    print("=" * 60)
    print("This script will help you reduce the 11GB Wildtrack dataset")
    print("down to under 1.5GB by:")
    print("  1. Deleting raw MP4 files (redundant since frames are extracted)")
    print("  2. Deleting unannotated frames (keeps only the 400 annotated frames)")
    print("  3. Optionally converting/compressing PNG frames to JPEGs (5-10x savings)")
    print("=" * 60)

def main():
    print_banner()
    
    # Ask for dataset directory path
    default_path = os.getcwd()
    user_path = input(f"Enter the path to your Wildtrack dataset folder\n[Default: {default_path}]: ").strip()
    dataset_dir = Path(user_path if user_path else default_path).resolve()
    
    if not dataset_dir.exists():
        print(f"Error: Directory '{dataset_dir}' does not exist.")
        return

    # Check key folders
    annotations_dir = dataset_dir / "annotations_positions"
    images_dir = dataset_dir / "Image_subsets"
    
    if not annotations_dir.exists() or not images_dir.exists():
        print("\nError: Could not find 'annotations_positions' or 'Image_subsets' in the specified directory.")
        print("Please make sure you point to the root directory of the Wildtrack dataset.")
        print(f"Looked in: {dataset_dir}")
        return

    # 1. Check for MP4 files
    mp4_files = list(dataset_dir.glob("cam*.mp4"))
    total_mp4_size = sum(f.stat().st_size for f in mp4_files)
    
    print("\n--- [Step 1] Raw MP4 Videos ---")
    if mp4_files:
        print(f"Found {len(mp4_files)} raw video files (cam1.mp4 - cam7.mp4).")
        print(f"Total video size: {total_mp4_size / (1024**3):.2f} GB")
        confirm_mp4 = input("These are redundant if frames are extracted. Delete them? (y/n): ").strip().lower()
        if confirm_mp4 == 'y':
            for f in mp4_files:
                try:
                    f.unlink()
                    print(f"Deleted: {f.name}")
                except Exception as e:
                    print(f"Error deleting {f.name}: {e}")
            print("Successfully deleted raw videos.")
        else:
            print("Skipped deleting video files.")
    else:
        print("No raw cam*.mp4 files found in the dataset root.")

    # 2. Get annotated frame numbers from annotations_positions
    print("\n--- [Step 2] Scanning Annotations ---")
    json_files = list(annotations_dir.glob("*.json"))
    if not json_files:
        print("Error: No JSON files found in annotations_positions directory.")
        return
        
    annotated_frames = {f.stem for f in json_files}
    print(f"Found {len(annotated_frames)} annotated frames (e.g., {list(annotated_frames)[:3]}...).")

    # 3. Clean up unannotated frames in Image_subsets
    print("\n--- [Step 3] Pruning Unannotated Frames ---")
    # Identify camera subdirectories in Image_subsets
    cam_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
    if not cam_dirs:
        print("Error: No camera subdirectories found inside Image_subsets.")
        return

    print(f"Found camera subdirectories: {[d.name for d in cam_dirs]}")
    
    # Calculate potential savings
    files_to_delete = []
    files_to_keep = []
    
    for cam_dir in cam_dirs:
        # Get all files (images) inside camera directory
        for img_file in cam_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                if img_file.stem not in annotated_frames:
                    files_to_delete.append(img_file)
                else:
                    files_to_keep.append(img_file)

    total_delete_size = sum(f.stat().st_size for f in files_to_delete)
    
    if files_to_delete:
        print(f"Found {len(files_to_delete)} unannotated frames to delete.")
        print(f"Found {len(files_to_keep)} annotated frames to keep.")
        print(f"This will free up approximately {total_delete_size / (1024**2):.2f} MB of space.")
        
        confirm_prune = input("Proceed with deleting unannotated frames? (y/n): ").strip().lower()
        if confirm_prune == 'y':
            deleted_count = 0
            for f in files_to_delete:
                try:
                    f.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {f}: {e}")
            print(f"Successfully deleted {deleted_count} unannotated frames.")
        else:
            print("Skipped deleting unannotated frames.")
    else:
        print("No unannotated frames found. Image_subsets contains only annotated frames.")

    # 4. Optional PNG-to-JPEG conversion
    print("\n--- [Step 4] Optional PNG-to-JPEG Compression ---")
    png_files_to_convert = [f for f in files_to_keep if f.suffix.lower() == '.png']
    if png_files_to_convert:
        print(f"Found {len(png_files_to_convert)} annotated frames stored as large PNG files.")
        print("PNGs can be converted to JPEGs to reduce file sizes by 80-90% with minimal quality loss.")
        
        confirm_convert = input("Would you like to compress PNG files to JPEGs (quality=95)? (y/n): ").strip().lower()
        if confirm_convert == 'y':
            try:
                from PIL import Image
            except ImportError:
                print("Error: The 'Pillow' library is required for image conversion.")
                print("Please install it first: pip install pillow")
                return

            print("Converting PNGs to JPEGs (this might take a minute)...")
            converted_count = 0
            for png_path in png_files_to_convert:
                try:
                    jpeg_path = png_path.with_suffix('.jpg')
                    with Image.open(png_path) as img:
                        img.convert("RGB").save(jpeg_path, "JPEG", quality=95)
                    png_path.unlink()  # Delete original PNG
                    converted_count += 1
                except Exception as e:
                    print(f"Error converting {png_path.name}: {e}")
            
            print(f"Successfully converted {converted_count} PNGs to JPEGs.")
            print("Important: Make sure you update any code loading these images to use '.jpg' instead of '.png'.")
        else:
            print("Skipped image conversion.")
    else:
        print("No PNG files found to convert (images might already be JPEG).")

    print("\n" + "=" * 60)
    print("                          DONE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
