import os
import shutil
import argparse

from .config import logger, DEFAULT_PROMPT_PATH
from .image_utils import pdf_to_images, get_relevant_slides_ssim_diff
from .ai_processing import batch_process_images
from .storage import generate_output_paths, check_existing_results, save_results

def process_pdf(pdf_path: str, prompt_path: str, extract_relevant_slides: bool) -> None:
    """
    Process PDF file and convert to LaTeX.
    
    Args:
        pdf_path: Path to the PDF file
        prompt_path: Path to the prompt file
    """
    logger.info(f"Processing PDF file: {pdf_path}")
    
    # Read prompt
    with open(prompt_path, "r", encoding='utf-8') as file:
        prompt = file.read().strip()

    # Generate paths for output
    output_dir, result_path = generate_output_paths(pdf_path, prompt)
    
    # Check if already processed
    if check_existing_results(result_path, pdf_path):
        logger.info(f"Already processed: {result_path}")
        return
    
    # Convert PDF to images
    image_paths, temp_dir = pdf_to_images(pdf_path)
    initial_image_count = len(image_paths)
    
    # Get relevant slides using SSIM difference
    if extract_relevant_slides:
        image_paths = get_relevant_slides_ssim_diff(image_paths)
    
    # Send to VLLM for processing
    responses, callback_data = batch_process_images(image_paths, prompt)
    
    total_cost = sum(cb["total_cost"] for cb in callback_data)
    
    logger.info(f"Total processing cost: {total_cost} USD")
    
    cost_per_image = total_cost / len(image_paths) if image_paths else 0
    saved_images = initial_image_count - len(image_paths)
    cost_saving = cost_per_image * saved_images if saved_images > 0 else 0
    
    logger.info(f"Estimated cost saving with incremental subsequence detection: {cost_saving} USD")
    
    # Save results
    save_results(pdf_path, output_dir, result_path, responses, callback_data, image_paths)
    
    #delete temporary images
    shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Convert slides PDF to LaTeX")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT_PATH, 
                        help=f"Path to the prompt file (default: {DEFAULT_PROMPT_PATH})")
    parser.add_argument("--extract-relevant-slides", action="store_true",
                        default=False,
                        help="Extract relevant slides using SSIM difference")
    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.pdf_path):
        logger.error(f"PDF file not found: {args.pdf_path}")
        return
    
    if not os.path.exists(args.prompt):
        logger.error(f"Prompt file not found: {args.prompt}")
        return

    try:
        process_pdf(args.pdf_path, args.prompt, args.extract_relevant_slides)
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()