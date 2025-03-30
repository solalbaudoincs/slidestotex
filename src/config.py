import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMAGE_BASE_WIDTH = 1024
API_KEY = "insert_your_api_key_here" 
# You could replace with environment variable usage for better security
# API_KEY = os.environ.get("OPENAI_API_KEY", "")

MODEL_NAME = "gpt-4o" # use chatgpt api model name convention
CONVERTED_PDFS_DIR = "./converted_pdfs"
