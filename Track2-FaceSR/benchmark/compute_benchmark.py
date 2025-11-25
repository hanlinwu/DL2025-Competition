import os
import glob
import numpy as np
import cv2
import torch
import lpips
import csv
import base64
from skimage.metrics import peak_signal_noise_ratio as psnr_ski
# Note: SSIM has been removed and replaced by FID (a dataset-level metric)

# --- IQA Library Initialization (NIQE and FID) ---
try:
    import pyiqa
    _HAS_IQA = True
    # NIQE is a per-image metric
    niqe_metric = pyiqa.create_metric('niqe', device=torch.device('cpu'))
    # FID is a dataset-level metric. Note: FID is initialized on CPU.
    fid_metric = pyiqa.create_metric('fid', device=torch.device('cpu')) 
except Exception as e:
    _HAS_IQA = False
    print(f"WARNING: pyiqa initialization failed ({e}). Perceptual scores (NIQE, FID) will be set to NaN.")

# LPIPS Model Initialization
LPIPS_MODEL = None
try:
    # Initialize LPIPS on CPU for broad compatibility
    LPIPS_MODEL = lpips.LPIPS(net='alex', spatial=False).to('cpu')
    LPIPS_MODEL.eval()
    _HAS_LPIPS = True
except Exception as e:
    _HAS_LPIPS = False
    print(f"WARNING: LPIPS initialization failed ({e}). LPIPS scores will be set to NaN.")

# =================================================================
# Configuration Parameters (Participants MUST modify these)
# =================================================================

# 1. 🚨 参赛者必须修改为自己超分结果的路径！！！！！
SUBMISSION_DIR = './SR_results' 

# 2. 🚨 ground truth的目录（可根据自己的运行路径修改）
GT_DIR = './celeb-GT-512' 

# 3. Output file
OUTPUT_CSV_FILE = 'submission.csv'
IMAGE_EXTENSION = '*.jpg' # Image format is jpg

# =================================================================
# Helper Functions
# =================================================================

def encode_image_to_base64(img_path):
    """Encodes an image to a Base64 string (using PNG encoding)"""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    # Use PNG encoding (lossless)
    success, encoded_buffer = cv2.imencode('.png', img)
    if not success:
        raise ValueError("Image encoding failed")

    base64_bytes = base64.b64encode(encoded_buffer.tobytes())
    return base64_bytes.decode('utf-8')

def load_image_and_process(filepath):
    """Loads image and returns NumPy (RGB uint8) for PSNR/Base64."""
    img = cv2.imread(filepath, cv2.IMREAD_COLOR) 
    if img is None:
        raise FileNotFoundError(f"Image not found at: {filepath}")
    
    # Convert to RGB format
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img_rgb # Returns uint8 RGB array (0-255)

def calculate_per_image_metrics(img_true_rgb, img_test_rgb, id):
    """Calculates PSNR, LPIPS, and NIQE for a single image."""
    
    if img_true_rgb.shape != img_test_rgb.shape:
        print(f"Skipping {id} metrics: Shape mismatch.")
        return {'id': id, 'psnr': np.nan, 'lpips': np.nan, 'niqe': np.nan}

    # --- 1. PSNR (based on uint8, 0-255) ---
    try:
        psnr_val = psnr_ski(img_true_rgb, img_test_rgb, data_range=255)
    except Exception:
        psnr_val = np.nan
    
    # --- 2. LPIPS (Perceptual FR) ---
    lpips_val = np.nan
    if _HAS_LPIPS:
        try:
            # LPIPS requires input tensors normalized to [-1, 1] (NCHW)
            tensor_true = torch.from_numpy(img_true_rgb).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1
            tensor_test = torch.from_numpy(img_test_rgb).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1
            
            with torch.no_grad():
                lpips_val = LPIPS_MODEL(tensor_true, tensor_test).item()
        except Exception:
            lpips_val = np.nan

    # --- 3. NIQE (Perceptual NR) ---
    niqe_val = np.nan
    if _HAS_IQA:
        try:
            # NIQE requires input tensor, range [0, 1]
            img_tensor = torch.from_numpy(img_test_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            niqe_val = niqe_metric(img_tensor).item()
        except Exception:
            niqe_val = np.nan

    return {
        'id': id,
        'psnr': psnr_val,
        'lpips': lpips_val,
        'niqe': niqe_val,
    }

# =================================================================
# Main Program
# =================================================================

def main():
    print("--- Starting Super-Resolution Metric & Base64 Calculation ---")
    
    # Get all Ground Truth files, which serve as the baseline
    gt_files = sorted(glob.glob(os.path.join(GT_DIR, IMAGE_EXTENSION)))
    
    if not gt_files:
        print(f"Error: Ground Truth directory is empty. Check path: {GT_DIR}")
        return

    # Prepare data structure
    # Store final results, key is image_id, value is dict containing metrics and base64
    results_map = {} 
    
    # Store all valid image paths for FID dataset calculation
    valid_gt_paths = []
    valid_sub_paths = []

    for gt_path in gt_files:
        base_name = os.path.basename(gt_path)
        sub_path = os.path.join(SUBMISSION_DIR, base_name)
        id = base_name.split('.')[0] # Use the filename without extension as the ID
        
        # Check if the submission file exists
        if not os.path.exists(sub_path):
            print(f"Warning: Submission file not found for ID {id}. Metrics will be NaN.")
            results_map[id] = {'id': id, 'psnr': 'NaN', 'lpips': 'NaN', 'niqe': 'NaN', 'image_base64': ''}
            continue
        
        try:
            # --- A. Load Images (uint8 RGB) ---
            img_true = load_image_and_process(gt_path)
            img_test = load_image_and_process(sub_path)
            
            # --- B. Calculate Per-Image Metrics ---
            metrics = calculate_per_image_metrics(img_true, img_test, id)
            
            # --- C. Base64 Encoding ---
            encoded_str = encode_image_to_base64(sub_path)
            
            # --- D. Store Results and FID Paths ---
            
            # Store per-image results (excluding FID)
            row_data = {
                **metrics,
                'image_base64': encoded_str
            }
            results_map[id] = row_data

            # Prepare FID path lists
            valid_gt_paths.append(gt_path)
            valid_sub_paths.append(sub_path)
            
        except Exception as e:
            print(f"Critical error processing {id}: {e}. Skipping.")
            # Write a row with NaN and empty Base64
            results_map[id] = {'id': id, 'psnr': 'NaN', 'lpips': 'NaN', 'niqe': 'NaN', 'image_base64': ''}
            continue

    # --- E. Calculate Dataset-Level FID ---
    final_results_list = list(results_map.values())
    
    if _HAS_IQA and valid_sub_paths and valid_gt_paths:
        try:
            print(f"\nCalculating Dataset-Level FID...")
            
            # 🚨 Key Fix: Pass directories to the FID metric for reliable calculation
            # pyiqa's FID metric often expects the directory paths
            fid_score_tensor = fid_metric(SUBMISSION_DIR, GT_DIR)
            fid_score = fid_score_tensor.item()
            
            print(f"Calculated Dataset-Level FID Score: {fid_score:.4f}")
            
            # Inject FID result into all rows
            # Since FID is a single score for the entire dataset, it applies to all successful image rows
            for row in final_results_list:
                row['fid'] = fid_score
        
        except Exception as e:
            print(f"Critical error during FID calculation: {e}. Setting FID to NaN for all rows.")
            fid_score = 'NaN'
            for row in final_results_list:
                row['fid'] = fid_score
    else:
        fid_score = 'NaN'
        for row in final_results_list:
            row['fid'] = fid_score

    # --- F. Write to CSV File ---
    if final_results_list:
        # Note: CSV header updated from ssim to fid
        header = ['id', 'psnr', 'lpips', 'niqe', 'fid', 'image_base64'] 
        
        with open(OUTPUT_CSV_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in final_results_list:
                # Convert numpy.nan to string 'NaN'
                # Check if FID is string 'NaN' or number
                processed_row = {k: v if not (isinstance(v, float) and np.isnan(v)) else 'NaN' for k, v in row.items()}
                writer.writerow(processed_row)
        print(f"\n--- SUCCESS ---")
        print(f"Processed {len(final_results_list)} images. Results saved to {OUTPUT_CSV_FILE}")
    else:
        print("\n--- FAILURE ---")
        print("No images were successfully processed. Check your paths.")

if __name__ == "__main__":
    main()