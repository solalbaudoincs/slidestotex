import os
import argparse
import tempfile
import shutil
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set

# --- Required Libraries ---
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pdf2image import convert_from_path
from skimage.metrics import structural_similarity as ssim
import imagehash
import cv2 # OpenCV

# --- Constants ---
DPI = 150 # Resolution for PDF to PNG conversion
THUMBNAIL_WIDTH = 200 # Width for visualization thumbnails
VIZ_PADDING = 10
VIZ_TEXT_HEIGHT = 15
VIZ_METHOD_BOX_HEIGHT = VIZ_TEXT_HEIGHT * 3 + VIZ_PADDING * 2 # Space for 3 method indicators

# --- Helper Function: Load Image (Slightly modified for uint8 needed by some libs) ---
def load_image_helper(filepath: str, mode: str = 'L') -> Optional[Tuple[Image.Image, np.ndarray]]:
    """Loads an image using Pillow and converts to NumPy array."""
    try:
        img_pil = Image.open(filepath).convert(mode)
        # Use uint8 for OpenCV compatibility, float32 for calculations if needed later
        img_np = np.array(img_pil, dtype=np.uint8 if mode != 'F' else np.float32)
        return img_pil, img_np
    except FileNotFoundError:
        print(f"Warning: File not found {filepath}")
        return None
    except Exception as e:
        print(f"Warning: Could not load image {filepath}: {e}")
        return None

def count_non_white_pixels(img_array: np.ndarray, threshold: int = 240) -> int:
    """Counts pixels darker than a threshold (approximates content)."""
    if img_array.dtype != np.uint8: # Ensure correct type for comparison
        img_array = img_array.astype(np.uint8)
    return np.sum(img_array < threshold)

# --- Detection Function V1 (from previous response) ---
def detect_incremental_mse_pixel_count(
    filepaths: List[str],
    mse_threshold: float = 100.0,
    pixel_increase_factor: float = 1.001,
    content_threshold: int = 240
) -> List[List[str]]:
    if not filepaths: return []
    all_subsequences = []
    current_subsequence = [filepaths[0]]
    load_result = load_image_helper(filepaths[0], 'L')
    if load_result is None: return []
    _, img_prev = load_result
    img_prev_float = img_prev.astype(np.float32) # Use float for MSE
    pixels_prev = count_non_white_pixels(img_prev)

    for i in range(1, len(filepaths)):
        filepath_curr = filepaths[i]
        load_result = load_image_helper(filepath_curr, 'L')
        if load_result is None:
            if current_subsequence: all_subsequences.append(current_subsequence)
            current_subsequence = []
            img_prev = None
            pixels_prev = 0
            continue
        _, img_curr = load_result
        img_curr_float = img_curr.astype(np.float32)

        if img_prev is None:
             current_subsequence = [filepath_curr]
             img_prev = img_curr
             img_prev_float = img_curr_float
             pixels_prev = count_non_white_pixels(img_prev)
             continue

        is_incremental = False
        if img_curr.shape == img_prev.shape:
            # Create mask where either image has content (pixels below threshold)
            mask = np.logical_or(img_prev < content_threshold, img_curr < content_threshold)
            if np.sum(mask) > 0:  # Avoid division by zero if mask is empty
                # Calculate MSE only on masked pixels
                masked_diff = (img_curr_float - img_prev_float) * mask
                mse = np.sum(masked_diff**2) / np.sum(mask)  # Normalize by number of masked pixels
            else:
                mse = 0  # If no content pixels, images are effectively identical
                
            pixels_curr = count_non_white_pixels(img_curr)
            if mse < mse_threshold and pixels_curr >= pixels_prev * pixel_increase_factor:
                is_incremental = True
        else: print(f"V1 Warn: Shape mismatch {filepaths[i-1]} vs {filepath_curr}")

        if is_incremental: current_subsequence.append(filepath_curr)
        else:
            if current_subsequence: all_subsequences.append(current_subsequence)
            current_subsequence = [filepath_curr]
        img_prev, img_prev_float, pixels_prev = img_curr, img_curr_float, count_non_white_pixels(img_curr)

    if current_subsequence: all_subsequences.append(current_subsequence)
    return all_subsequences

# --- Detection Function V2 (from previous response) ---
def detect_incremental_ssim_diff(
    filepaths: List[str],
    ssim_threshold: float = 0.98,
    diff_pixel_max_ratio: float = 0.05,
    diff_change_threshold: int = 50
) -> List[List[str]]:
    if not filepaths: return []
    all_subsequences = []
    current_subsequence = [filepaths[0]]
    load_result = load_image_helper(filepaths[0], 'L')
    if load_result is None: return []
    _, img_prev = load_result # Needs uint8 for SSIM

    for i in range(1, len(filepaths)):
        filepath_curr = filepaths[i]
        load_result = load_image_helper(filepath_curr, 'L')
        if load_result is None:
            if current_subsequence: all_subsequences.append(current_subsequence)
            current_subsequence = []
            img_prev = None
            continue
        _, img_curr = load_result

        if img_prev is None:
             current_subsequence = [filepath_curr]
             img_prev = img_curr
             continue

        is_incremental = False
        if img_curr.shape == img_prev.shape:
            try:
                # SSIM uses uint8, data_range is 255
                score = ssim(img_prev, img_curr, data_range=img_prev.max() - img_prev.min())

                # Analyze absolute difference
                abs_diff = np.abs(img_curr.astype(np.float32) - img_prev.astype(np.float32))
                noise_threshold = 10.0
                significant_diff_pixels = np.sum(abs_diff > noise_threshold)
                total_pixels = img_curr.size
                diff_ratio = significant_diff_pixels / total_pixels

                if (score > ssim_threshold and
                    diff_ratio < diff_pixel_max_ratio and
                    significant_diff_pixels > diff_change_threshold):
                     is_incremental = True
            except Exception as e: # Catch potential SSIM errors (e.g., flat images)
                print(f"V2 Warn: SSIM error between {filepaths[i-1]} and {filepath_curr}: {e}")
                is_incremental = False # Treat as non-incremental on error
        else: print(f"V2 Warn: Shape mismatch {filepaths[i-1]} vs {filepath_curr}")

        if is_incremental: current_subsequence.append(filepath_curr)
        else:
            if current_subsequence: all_subsequences.append(current_subsequence)
            current_subsequence = [filepath_curr]
        img_prev = img_curr

    if current_subsequence: all_subsequences.append(current_subsequence)
    return all_subsequences

# --- Detection Function V3 (from previous response) ---
def detect_incremental_phash_template(
    filepaths: List[str],
    hash_diff_threshold: int = 3,
    template_match_threshold: float = 0.98,
    diff_change_threshold: int = 50,
    diff_pixel_max_ratio: float = 0.05
) -> List[List[str]]:
    if not filepaths: return []
    all_subsequences = []
    current_subsequence = [filepaths[0]]
    load_result = load_image_helper(filepaths[0], 'L') # Need PIL for hash, NP for CV
    if load_result is None: return []
    img_prev_pil, img_prev_np = load_result
    try:
        hash_prev = imagehash.phash(img_prev_pil)
    except Exception as e:
         print(f"V3 Warn: pHash error on {filepaths[0]}: {e}")
         return [] # Cannot start sequence

    for i in range(1, len(filepaths)):
        filepath_curr = filepaths[i]
        load_result = load_image_helper(filepath_curr, 'L')
        if load_result is None:
            if current_subsequence: all_subsequences.append(current_subsequence)
            current_subsequence = []
            img_prev_pil, img_prev_np, hash_prev = None, None, None
            continue
        img_curr_pil, img_curr_np = load_result

        if img_prev_np is None or hash_prev is None:
             current_subsequence = [filepath_curr]
             img_prev_pil, img_prev_np = img_curr_pil, img_curr_np
             try: hash_prev = imagehash.phash(img_prev_pil)
             except Exception as e:
                 print(f"V3 Warn: pHash error on {filepath_curr}: {e}. Resetting.")
                 hash_prev = None # Break sequence here if hash fails
                 if current_subsequence: all_subsequences.append(current_subsequence)
                 current_subsequence = []
             continue

        is_incremental = False
        try:
            hash_curr = imagehash.phash(img_curr_pil)
            if img_curr_np.shape == img_prev_np.shape:
                hash_distance = hash_curr - hash_prev
                if hash_distance <= hash_diff_threshold:
                    # Use TM_CCOEFF_NORMED
                    result = cv2.matchTemplate(img_curr_np, img_prev_np, cv2.TM_CCOEFF_NORMED)
                    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

                    if maxVal >= template_match_threshold and maxLoc == (0, 0):
                        abs_diff = np.abs(img_curr_np.astype(np.float32) - img_prev_np.astype(np.float32))
                        noise_threshold = 10.0
                        significant_diff_pixels = np.sum(abs_diff > noise_threshold)
                        total_pixels = img_curr_np.size
                        diff_ratio = significant_diff_pixels / total_pixels

                        if (significant_diff_pixels > diff_change_threshold and
                            diff_ratio < diff_pixel_max_ratio):
                            is_incremental = True
            else: print(f"V3 Warn: Shape mismatch {filepaths[i-1]} vs {filepath_curr}")
        except Exception as e:
            print(f"V3 Warn: Error during comparison {filepaths[i-1]} vs {filepath_curr}: {e}")
            is_incremental = False # Treat as non-incremental on error

        if is_incremental: current_subsequence.append(filepath_curr)
        else:
            if current_subsequence: all_subsequences.append(current_subsequence)
            current_subsequence = [filepath_curr]
        img_prev_pil, img_prev_np, hash_prev = img_curr_pil, img_curr_np, hash_curr

    if current_subsequence: all_subsequences.append(current_subsequence)
    return all_subsequences

# --- Main Orchestration & Visualization ---
def get_last_slides(subsequences: List[List[str]]) -> List[str]:
    """Helper to extract last slide from each subsequence."""
    return [sub[-1] for sub in subsequences if sub]

def pdf_to_images(pdf_path: str, output_folder: str, dpi: int) -> List[str]:
    """Converts PDF pages to PNG images."""
    print(f"Converting {pdf_path} to images in {output_folder}...")
    try:
        images = convert_from_path(pdf_path, dpi=dpi, output_folder=output_folder, fmt='png', thread_count=4, output_file='page')
        num_pages = len(images)
        filepaths = []
        for i in range(1, num_pages + 1):
             found = False
             for filename in os.listdir(output_folder):
                 if filename.startswith(f'page-{i}') and filename.endswith('.png'):
                     filepaths.append(os.path.join(output_folder, filename))
                     found = True
                     break
                 elif filename.startswith(f'page_{i:03d}') and filename.endswith('.png'):
                     filepaths.append(os.path.join(output_folder, filename))
                     found = True
                     break
             if not found:
                 print(f"Warning: Could not reliably find image for page {i}. Trying directory listing.")
                 all_pngs = sorted([os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.lower().endswith('.png')])
                 if len(all_pngs) == num_pages:
                     print("Using sorted directory listing.")
                     return all_pngs
                 else:
                     raise RuntimeError(f"Could not determine generated image path for page {i}. Found files: {os.listdir(output_folder)}")

        print(f"Successfully converted {len(filepaths)} pages.")
        filepaths.sort(key=lambda x: int(Path(x).stem.split('-')[-1].split('_')[-1]))
        return filepaths
    except Exception as e:
        print(f"Error during PDF conversion: {e}")
        print("Ensure poppler is installed and its bin directory is in your system PATH.")
        return []

def create_visual_demo(
    all_filepaths: List[str],
    last_slides_v1: List[str],
    last_slides_v2: List[str],
    last_slides_v3: List[str],
    output_png_path: str
):
    """Creates a single PNG visualizing the results of the three methods."""
    print(f"Generating visualization: {output_png_path}")
    if not all_filepaths:
        print("No images to visualize.")
        return

    num_slides = len(all_filepaths)
    try:
        first_img = Image.open(all_filepaths[0])
        aspect_ratio = first_img.height / first_img.width
        thumb_h = int(THUMBNAIL_WIDTH * aspect_ratio)
    except Exception as e:
        print(f"Warning: Could not read first image {all_filepaths[0]} for dimensions: {e}. Using default height.")
        thumb_h = int(THUMBNAIL_WIDTH * (3/4))

    total_thumb_height = thumb_h + VIZ_PADDING + VIZ_METHOD_BOX_HEIGHT + VIZ_PADDING + VIZ_TEXT_HEIGHT + VIZ_PADDING
    cols = max(1, int(math.sqrt(num_slides * 2)))
    rows = math.ceil(num_slides / cols)

    img_width = cols * (THUMBNAIL_WIDTH + VIZ_PADDING) + VIZ_PADDING
    img_height = rows * total_thumb_height + VIZ_PADDING

    viz_img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(viz_img)
    try:
        font_path = "arial.ttf"
        font_small = ImageFont.truetype(font_path, VIZ_TEXT_HEIGHT - 2)
        font_title = ImageFont.truetype(font_path, VIZ_TEXT_HEIGHT)
    except IOError:
        print("Warning: Arial font not found. Using default PIL font.")
        font_small = ImageFont.load_default()
        font_title = ImageFont.load_default()

    kept_color = (0, 180, 0)
    skipped_color = (200, 0, 0)
    text_color = (0, 0, 0)
    methods = ["V1", "V2", "V3"]
    results = [last_slides_v1, last_slides_v2, last_slides_v3]

    current_col = 0
    current_row = 0
    for i, filepath in enumerate(all_filepaths):
        x_offset = VIZ_PADDING + current_col * (THUMBNAIL_WIDTH + VIZ_PADDING)
        y_offset = VIZ_PADDING + current_row * total_thumb_height

        try:
            thumb = Image.open(filepath)
            thumb.thumbnail((THUMBNAIL_WIDTH, thumb_h))
            viz_img.paste(thumb, (x_offset, y_offset))
        except Exception as e:
             print(f"Warning: Could not load/paste thumbnail for {filepath}: {e}")
             draw.rectangle([x_offset, y_offset, x_offset+THUMBNAIL_WIDTH, y_offset+thumb_h], fill=(200,200,200), outline=(0,0,0))
             draw.text((x_offset+5, y_offset+5), "Load Error", fill=(255,0,0), font=font_small)

        title = f"Slide {i+1}"
        draw.text((x_offset, y_offset + thumb_h + VIZ_PADDING), title, fill=text_color, font=font_title)

        indicator_y_start = y_offset + thumb_h + VIZ_PADDING + VIZ_TEXT_HEIGHT + VIZ_PADDING
        for m_idx, method_name in enumerate(methods):
            is_kept = filepath in results[m_idx]
            color = kept_color if is_kept else skipped_color
            status_text = "Keep" if is_kept else "Skip"
            indicator_y = indicator_y_start + m_idx * (VIZ_TEXT_HEIGHT + VIZ_PADDING // 2)
            draw.text((x_offset, indicator_y), f"{method_name}: {status_text}", fill=color, font=font_small)

        current_col += 1
        if current_col >= cols:
            current_col = 0
            current_row += 1

    try:
        viz_img.save(output_png_path)
        print(f"Visualization saved to {output_png_path}")
    except Exception as e:
        print(f"Error saving visualization: {e}")

# --- Threshold Comparison Functions ---
def compare_thresholds(
    image_filepaths: List[str],
    output_dir: str,
    v1_mse_thresholds: List[float] = [50.0, 100.0, 150.0, 200.0],
    v1_pixel_factors: List[float] = [1.0001, 1.0005, 1.001, 1.005],
    v2_ssim_thresholds: List[float] = [0.95, 0.96, 0.97, 0.98, 0.99],
    v2_diff_ratios: List[float] = [0.03, 0.05, 0.08, 0.1],
    v3_hash_thresholds: List[int] = [2, 3, 4, 5, 6],
    v3_template_thresholds: List[float] = [0.95, 0.96, 0.97, 0.98, 0.99]
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    compare_v1_thresholds(image_filepaths, output_dir, v1_mse_thresholds, v1_pixel_factors)
    compare_v2_thresholds(image_filepaths, output_dir, v2_ssim_thresholds, v2_diff_ratios)
    compare_v3_thresholds(image_filepaths, output_dir, v3_hash_thresholds, v3_template_thresholds)

def create_threshold_comparison_viz(
    all_filepaths: List[str],
    results_dict: Dict[str, Set[str]],
    output_png_path: str,
    title: str
) -> None:
    print(f"Generating threshold comparison: {output_png_path}")
    if not all_filepaths:
        print("No images to visualize.")
        return

    num_slides = len(all_filepaths)
    num_thresholds = len(results_dict)
    try:
        first_img = Image.open(all_filepaths[0])
        aspect_ratio = first_img.height / first_img.width
        thumb_h = int(THUMBNAIL_WIDTH * aspect_ratio)
    except Exception as e:
        print(f"Warning: Could not read first image for dimensions: {e}. Using default height.")
        thumb_h = int(THUMBNAIL_WIDTH * (3/4))

    label_height = VIZ_TEXT_HEIGHT
    threshold_row_height = VIZ_TEXT_HEIGHT + VIZ_PADDING
    total_thumb_height = thumb_h + VIZ_PADDING + (num_thresholds * threshold_row_height) + VIZ_PADDING + label_height
    cols = max(1, int(math.sqrt(num_slides * 2)))
    rows = math.ceil(num_slides / cols)

    img_width = cols * (THUMBNAIL_WIDTH + VIZ_PADDING) + VIZ_PADDING
    img_height = rows * total_thumb_height + VIZ_PADDING

    viz_img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(viz_img)
    try:
        font_path = "arial.ttf"
        font_small = ImageFont.truetype(font_path, label_height - 2)
        font_title = ImageFont.truetype(font_path, label_height)
    except IOError:
        print("Warning: Arial font not found. Using default PIL font.")
        font_small = ImageFont.load_default()
        font_title = ImageFont.load_default()

    kept_color = (0, 180, 0)
    skipped_color = (200, 0, 0)
    text_color = (0, 0, 0)
    draw.text((VIZ_PADDING, VIZ_PADDING), title, fill=text_color, font=font_title)
    current_col = 0
    current_row = 0
    for i, filepath in enumerate(all_filepaths):
        x_offset = VIZ_PADDING + current_col * (THUMBNAIL_WIDTH + VIZ_PADDING)
        y_offset = VIZ_PADDING * 2 + label_height + current_row * total_thumb_height
        try:
            thumb = Image.open(filepath)
            thumb.thumbnail((THUMBNAIL_WIDTH, thumb_h))
            viz_img.paste(thumb, (x_offset, y_offset))
        except Exception as e:
            print(f"Warning: Could not load/paste thumbnail for {filepath}: {e}")
            draw.rectangle([x_offset, y_offset, x_offset+THUMBNAIL_WIDTH, y_offset+thumb_h], 
                           fill=(200,200,200), outline=(0,0,0))
            draw.text((x_offset+5, y_offset+5), "Load Error", fill=(255,0,0), font=font_small)

        slide_title = f"Slide {i+1}"
        draw.text((x_offset, y_offset + thumb_h + VIZ_PADDING), slide_title, fill=text_color, font=font_title)
        indicator_y_start = y_offset + thumb_h + VIZ_PADDING + label_height + VIZ_PADDING
        for t_idx, (threshold_label, kept_slides) in enumerate(results_dict.items()):
            is_kept = filepath in kept_slides
            color = kept_color if is_kept else skipped_color
            status_text = "Keep" if is_kept else "Skip"
            indicator_y = indicator_y_start + t_idx * threshold_row_height
            draw.text((x_offset, indicator_y), f"{threshold_label}: {status_text}", fill=color, font=font_small)

        current_col += 1
        if current_col >= cols:
            current_col = 0
            current_row += 1

    try:
        viz_img.save(output_png_path)
        print(f"Threshold comparison visualization saved to {output_png_path}")
    except Exception as e:
        print(f"Error saving threshold comparison: {e}")

def compare_v1_thresholds(
    image_filepaths: List[str],
    output_dir: str,
    mse_thresholds: List[float],
    pixel_factors: List[float]
) -> None:
    fixed_pixel_factor = 1.0005
    results = {}
    for mse in mse_thresholds:
        subsequences = detect_incremental_mse_pixel_count(
            image_filepaths,
            mse_threshold=mse,
            pixel_increase_factor=fixed_pixel_factor
        )
        kept_slides = set(get_last_slides(subsequences))
        results[f"MSE={mse}"] = kept_slides
    create_threshold_comparison_viz(
        image_filepaths,
        results,
        os.path.join(output_dir, "v1_mse_comparison.png"),
        "V1 MSE Threshold Comparison (fixed pixel factor)"
    )
    fixed_mse = 100.0
    results = {}
    for factor in pixel_factors:
        subsequences = detect_incremental_mse_pixel_count(
            image_filepaths,
            mse_threshold=fixed_mse,
            pixel_increase_factor=factor
        )
        kept_slides = set(get_last_slides(subsequences))
        results[f"Factor={factor}"] = kept_slides
    create_threshold_comparison_viz(
        image_filepaths,
        results,
        os.path.join(output_dir, "v1_pixfactor_comparison.png"),
        "V1 Pixel Factor Comparison (fixed MSE)"
    )

def compare_v2_thresholds(
    image_filepaths: List[str],
    output_dir: str,
    ssim_thresholds: List[float],
    diff_ratios: List[float]
) -> None:
    fixed_diff_ratio = 0.05
    fixed_diff_change = 50
    results = {}
    for ssim in ssim_thresholds:
        subsequences = detect_incremental_ssim_diff(
            image_filepaths,
            ssim_threshold=ssim,
            diff_pixel_max_ratio=fixed_diff_ratio,
            diff_change_threshold=fixed_diff_change
        )
        kept_slides = set(get_last_slides(subsequences))
        results[f"SSIM={ssim}"] = kept_slides
    create_threshold_comparison_viz(
        image_filepaths,
        results,
        os.path.join(output_dir, "v2_ssim_comparison.png"),
        "V2 SSIM Threshold Comparison (fixed diff ratio)"
    )
    fixed_ssim = 0.97
    results = {}
    for ratio in diff_ratios:
        subsequences = detect_incremental_ssim_diff(
            image_filepaths,
            ssim_threshold=fixed_ssim,
            diff_pixel_max_ratio=ratio,
            diff_change_threshold=fixed_diff_change
        )
        kept_slides = set(get_last_slides(subsequences))
        results[f"DiffRatio={ratio}"] = kept_slides
    create_threshold_comparison_viz(
        image_filepaths,
        results,
        os.path.join(output_dir, "v2_diffratio_comparison.png"),
        "V2 Diff Ratio Comparison (fixed SSIM)"
    )

def compare_v3_thresholds(
    image_filepaths: List[str],
    output_dir: str,
    hash_thresholds: List[int],
    template_thresholds: List[float]
) -> None:
    fixed_template = 0.97
    fixed_diff_ratio = 0.05
    fixed_diff_change = 50
    results = {}
    for hash_diff in hash_thresholds:
        subsequences = detect_incremental_phash_template(
            image_filepaths,
            hash_diff_threshold=hash_diff,
            template_match_threshold=fixed_template,
            diff_pixel_max_ratio=fixed_diff_ratio,
            diff_change_threshold=fixed_diff_change
        )
        kept_slides = set(get_last_slides(subsequences))
        results[f"HashDiff={hash_diff}"] = kept_slides
    create_threshold_comparison_viz(
        image_filepaths,
        results,
        os.path.join(output_dir, "v3_hash_comparison.png"),
        "V3 Hash Difference Threshold Comparison (fixed template match)"
    )
    fixed_hash_diff = 3
    results = {}
    for template in template_thresholds:
        subsequences = detect_incremental_phash_template(
            image_filepaths,
            hash_diff_threshold=fixed_hash_diff,
            template_match_threshold=template,
            diff_pixel_max_ratio=fixed_diff_ratio,
            diff_change_threshold=fixed_diff_change
        )
        kept_slides = set(get_last_slides(subsequences))
        results[f"Template={template}"] = kept_slides
    create_threshold_comparison_viz(
        image_filepaths,
        results,
        os.path.join(output_dir, "v3_template_comparison.png"),
        "V3 Template Match Threshold Comparison (fixed hash diff)"
    )

def main():
    parser = argparse.ArgumentParser(description="Detect incremental slides in a PDF and visualize comparison.")
    parser.add_argument("pdf_filepath", help="Path to the input PDF file.")
    parser.add_argument("-o", "--output", help="Filename for the output comparison PNG (default: demo_comparison_<pdf_name>.png).")
    parser.add_argument("--dpi", type=int, default=DPI, help=f"Resolution for PDF conversion (default: {DPI})")
    parser.add_argument("--v1-mse", type=float, default=150.0, help="V1: MSE threshold")
    parser.add_argument("--v1-pixinc", type=float, default=1.0005, help="V1: Pixel increase factor")
    parser.add_argument("--v2-ssim", type=float, default=0.97, help="V2: SSIM threshold")
    parser.add_argument("--v2-diffratio", type=float, default=0.08, help="V2: Max diff pixel ratio")
    parser.add_argument("--v2-diffchange", type=int, default=40, help="V2: Min diff pixel change")
    parser.add_argument("--v3-hash", type=int, default=4, help="V3: pHash difference threshold")
    parser.add_argument("--v3-template", type=float, default=0.97, help="V3: Template match threshold")
    parser.add_argument("--v3-diffratio", type=float, default=0.08, help="V3: Max diff pixel ratio")
    parser.add_argument("--v3-diffchange", type=int, default=40, help="V3: Min diff pixel change")
    parser.add_argument("--compare-thresholds", action="store_true", help="Run comparison with different threshold values")
    parser.add_argument("--threshold-viz-dir", help="Directory for threshold comparison visualizations (default: thresholds_<pdf_name>)")

    args = parser.parse_args()

    pdf_path = Path(args.pdf_filepath)
    if not pdf_path.is_file():
        print(f"Error: PDF file not found at {args.pdf_filepath}")
        return

    output_filename = args.output or f"demo_comparison_{pdf_path.stem}.png"

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        image_filepaths = pdf_to_images(str(pdf_path), temp_dir, args.dpi)

        if not image_filepaths:
            print("Failed to convert PDF to images. Exiting.")
            return

        print("\n--- Running Detection Algorithms ---")
        print("Running V1 (Simple MSE)...")
        subsequences_v1 = detect_incremental_mse_pixel_count(
            image_filepaths,
            mse_threshold=args.v1_mse,
            pixel_increase_factor=args.v1_pixinc
        )
        last_slides_v1 = get_last_slides(subsequences_v1)
        print(f"V1 Results: Found {len(subsequences_v1)} subsequences, keeping {len(last_slides_v1)} slides.")
        print("\nRunning V2 (SSIM & Diff)...")
        subsequences_v2 = detect_incremental_ssim_diff(
            image_filepaths,
            ssim_threshold=args.v2_ssim,
            diff_pixel_max_ratio=args.v2_diffratio,
            diff_change_threshold=args.v2_diffchange
        )
        last_slides_v2 = get_last_slides(subsequences_v2)
        print(f"V2 Results: Found {len(subsequences_v2)} subsequences, keeping {len(last_slides_v2)} slides.")
        print("\nRunning V3 (pHash & Template)...")
        subsequences_v3 = detect_incremental_phash_template(
            image_filepaths,
            hash_diff_threshold=args.v3_hash,
            template_match_threshold=args.v3_template,
            diff_change_threshold=args.v3_diffchange,
            diff_pixel_max_ratio=args.v3_diffratio
        )
        last_slides_v3 = get_last_slides(subsequences_v3)
        print(f"V3 Results: Found {len(subsequences_v3)} subsequences, keeping {len(last_slides_v3)} slides.")
        print("\n--- Generating Visual Comparison ---")
        create_visual_demo(
            image_filepaths,
            last_slides_v1,
            last_slides_v2,
            last_slides_v3,
            output_filename
        )
        if args.compare_thresholds:
            threshold_dir = args.threshold_viz_dir or f"thresholds_{pdf_path.stem}"
            print(f"\n--- Running Threshold Comparisons ---")
            print(f"Saving comparison visualizations to: {threshold_dir}")
            compare_thresholds(image_filepaths, threshold_dir)

    print(f"\nProcessing finished. Temporary files in {temp_dir} deleted.")

if __name__ == "__main__":
    main()