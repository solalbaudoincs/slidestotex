import os
import hashlib
import shutil
import pickle
from typing import Any, List, Dict, Tuple, TypeAlias

FilePath: TypeAlias = str

from plyer import notification

from .config import CONVERTED_PDFS_DIR, logger

def generate_output_paths(pdf_path: FilePath, prompt: str) -> Tuple[str, str]:
    """
    Generate output directory and file paths based on input PDF and prompt.
    
    Args:
        pdf_path: Path to the PDF file
        prompt: Prompt used for processing
        
    Returns:
        Tuple of (output_dir, result_path)
    """
    pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
    pdf_prompt_hash = hashlib.md5((pdf_path + prompt).encode()).hexdigest()
    output_dir = f"{CONVERTED_PDFS_DIR}/{pdf_name}"
    result_path = os.path.join(output_dir, f"{pdf_prompt_hash}.pkl")
    
    return output_dir, result_path

def check_existing_results(result_path: FilePath, pdf_path: FilePath) -> bool:
    """
    Check if results already exist for the given PDF and prompt.
    
    Args:
        result_path: Path to expected results file
        pdf_path: Original PDF path for logging
        
    Returns:
        True if results exist, False otherwise
    """
    if os.path.exists(result_path):
        logger.info(f"Result for {pdf_path} with the given prompt already exists. Skipping processing.")
        return True
    return False

def save_results(
    pdf_path: FilePath,
    output_dir: str, 
    result_path: str, 
    responses: List[Any], 
    callback_data: List[Dict],
    image_paths: List[FilePath]
) -> None:
    """
    Save processing results to disk.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save results
        result_path: Path to save the pickle file
        responses: Model responses
        callback_data: Callback data with token usage and costs
        image_paths: List of paths to slide images
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        pdf_name = os.path.basename(pdf_path)
        
        # Copy the original PDF to output directory
        shutil.copy(pdf_path, os.path.join(output_dir, pdf_name))
        
        # Save responses and callback data
        with open(result_path, "wb") as out_fp:
            pickle.dump((responses, callback_data), out_fp)
        
        # Save slide images as numbered pages
        
        if image_paths:
            for i, img_path in enumerate(image_paths):
                page_name = f"page_{i+1}.jpg"
                output_img_path = os.path.join(output_dir, page_name)
                shutil.copy(img_path, output_img_path)
                logger.debug(f"Saved slide image: {output_img_path}")
        
        logger.info(f"Results saved to {result_path}")
        
        # Send notification
        try:
            notification.notify(
                title='PDF Processing Complete',
                message=f'Successfully processed {pdf_name}',
                app_icon=None,
                timeout=10,
            )
        except Exception:
            logger.warning("Desktop notification failed - continuing without notification")
            
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        # Fallback save method
        pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
        with open(f"{pdf_name}_responses.pkl", "wb") as out_fp:
            pickle.dump((responses, callback_data), out_fp)
        logger.info(f"Results saved to fallback location: {pdf_name}_responses.pkl")
