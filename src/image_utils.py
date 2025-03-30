import os
import tempfile
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from skimage.metrics import structural_similarity as ssim

from .config import IMAGE_BASE_WIDTH

FilePath = str
ImageMatrix = np.ndarray

logger = logging.getLogger(__name__)


def pdf_to_images(pdf_path: FilePath) -> Tuple[List[FilePath], FilePath]:
    """
    Convert PDF pages to PNG images in a temporary directory.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        A tuple containing:
            - List of paths to generated PNG images (resized).
            - The path to the temporary directory created (needs cleanup by caller).
    """
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Converting PDF to images in temporary directory: {temp_dir}")
    filepaths = []
    try:
       
        pages = convert_from_path(pdf_path, dpi=200) 

        for i, page in enumerate(pages):
            out_path = os.path.join(temp_dir, f"page_{i+1}.png")

            # Resize based on fixed width
            wpercent = IMAGE_BASE_WIDTH / float(page.size[0])
            hsize = int(float(page.size[1]) * wpercent)
            resized_page = page.resize((IMAGE_BASE_WIDTH, hsize), Image.Resampling.LANCZOS)

            # Save as PNG
            resized_page.convert('RGB').save(out_path, "PNG", quality=95) 
            filepaths.append(out_path)

        logger.info(f"Successfully converted PDF to {len(filepaths)} images in {temp_dir}")
        return filepaths, temp_dir

    except Exception as e:
        logger.error(f"Failed during PDF conversion: {e}")
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        return [], temp_dir

def load_grayscale_image_data(filepath: FilePath) -> Optional[ImageMatrix]:
    try:
        img_pil = Image.open(filepath).convert('L') # Convert to grayscale
        # Use uint8 as required by skimage.ssim
        img_np = np.array(img_pil, dtype=np.uint8)
        return img_np
    except FileNotFoundError:
        logger.warning(f"File not found during loading: {filepath}")
        return None
    except Exception as e:
        logger.warning(f"Could not load image {filepath}: {e}")
        return None

def detect_incremental_ssim_diff(
    filepaths: List[FilePath],
    ssim_threshold: float = 0.95,       # Min SSIM score (higher means more similar)
    diff_pixel_max_ratio: float = 0.05, # Max ratio of significantly different pixels
    diff_change_threshold: int = 50     # Min number of pixels that must change significantly
                                        # (absolute intensity diff > noise_threshold)
) -> List[List[FilePath]]:
    """
    Detects incremental subsequences using SSIM and difference analysis.
    SSIM: https://en.wikipedia.org/wiki/Structural_similarity_index_measure

    Args:
        filepaths: List of PNG file paths (grayscale recommended) in order.
        ssim_threshold: Minimum SSIM score for slides to be considered similar structure.
        diff_pixel_max_ratio: The ratio of significantly different pixels (absolute diff)
                              to total pixels should not exceed this value for incremental change.
        diff_change_threshold: The minimum number of significantly different pixels required
                               to register as a change (i.e., not identical).

    Returns:
        A list of lists, where each inner list contains filepaths of a subsequence.
    """
    if not filepaths:
        return []

    all_subsequences = []
    current_subsequence = [] 

    img_prev: Optional[ImageMatrix] = None

    for i, filepath_curr in enumerate(filepaths):
        img_curr = load_grayscale_image_data(filepath_curr)

        if img_curr is None:
            logger.warning(f"Skipping {filepath_curr} due to loading error.")
            # End the current subsequence if an image fails to load
            if current_subsequence:
                all_subsequences.append(current_subsequence)
            current_subsequence = [] 
            img_prev = None          
            continue                 # Move to next file path

        # Handle the very first image or image after a load failure
        if img_prev is None:
            current_subsequence = [filepath_curr]
            img_prev = img_curr
            continue # Go to the next iteration to compare with the second image

        
        is_incremental = False
        if img_curr.shape == img_prev.shape:
            try:
                # 1. Calculate SSIM
                score = ssim(img_prev, img_curr, data_range=img_prev.max() - img_prev.min())

                # 2. Analyze the absolute difference
                # Use float32 for accurate difference calculation, then threshold
                abs_diff = np.abs(img_curr.astype(np.float32) - img_prev.astype(np.float32))
                noise_threshold = 10.0 # Intensity difference threshold
                significant_diff_pixels = np.sum(abs_diff > noise_threshold)
                total_pixels = img_curr.size
                diff_ratio = significant_diff_pixels / total_pixels

                # Check conditions:
                # - High structural similarity (SSIM)
                # - Difference is small relative to image size (max ratio)
                # - Some significant difference exists (change threshold) - ensures not identical
                if (score > ssim_threshold and
                    diff_ratio < diff_pixel_max_ratio and
                    significant_diff_pixels > diff_change_threshold):
                     is_incremental = True

            except Exception as e: # Catch potential SSIM/calculation errors
                logger.warning(f"Comparison error between {filepaths[i-1]} and {filepath_curr}: {e}. Treating as new sequence.")
                is_incremental = False # Treat as non-incremental on error
        else:
             logger.warning(f"Shape mismatch between {Path(filepaths[i-1]).name} and {Path(filepath_curr).name}. Treating as new sequence.")
             is_incremental = False # Shape mismatch means not incremental

        if is_incremental:
            current_subsequence.append(filepath_curr)
        else:
            if current_subsequence:
                 all_subsequences.append(current_subsequence)
            current_subsequence = [filepath_curr]

        img_prev = img_curr

    if current_subsequence:
        all_subsequences.append(current_subsequence)

    logger.info(f"SSIM Diff Detection: Found {len(all_subsequences)} subsequences.")
    return all_subsequences


def get_relevant_slides_ssim_diff(
    all_slide_filepaths: List[FilePath],
    ssim_threshold: float = 0.97,
    diff_pixel_max_ratio: float = 0.06,
    diff_change_threshold: int = 50
) -> List[FilePath]:
    """
    Extract relevant slides 
    """
    logger.info("Starting SSIM & Difference based incremental slide detection...")
    subsequences = detect_incremental_ssim_diff(
        all_slide_filepaths,
        ssim_threshold=ssim_threshold,
        diff_pixel_max_ratio=diff_pixel_max_ratio,
        diff_change_threshold=diff_change_threshold
    )

    relevant_slides = [sub[-1] for sub in subsequences if sub]

    logger.info(f"Identified {len(relevant_slides)} relevant slides out of {len(all_slide_filepaths)} total.")
    return relevant_slides