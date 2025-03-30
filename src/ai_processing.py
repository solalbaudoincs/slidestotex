import base64
from typing import Any, List, Dict

from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from tqdm.auto import tqdm
import logging

from .config import API_KEY, MODEL_NAME, logger

def get_ai_model() -> BaseChatModel:
    """
    Create and return a configured AI model instance.
    
    Returns:
        Configured language model
    """
    if not API_KEY:
        raise ValueError("OpenAI API key not found")
    
    return ChatOpenAI(api_key=API_KEY, model_name=MODEL_NAME)

def image_to_tex(filepath: str, prompt: str, model: BaseChatModel) -> Any:
    """
    Convert image to LaTeX using AI model.
    
    Args:
        filepath: Path to the image file
        prompt: Instruction prompt for the AI
        model: Language model instance
        
    Returns:
        Model response
    """
    with open(filepath, "rb") as f:
        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode()
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                },
            ]
        )
    return model.invoke([message])

def batch_process_images(image_paths: List[str], prompt: str) -> tuple:
    """
    Process multiple images with the AI model.
    
    Args:
        image_paths: List of paths to images
        prompt: Instruction prompt for the AI
        
    Returns:
        Tuple of (responses, callback_data)
    """
    model = get_ai_model()
    responses = []
    callback_data = []
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        logging.disable(logging.CRITICAL)  # Disable all logging during API calls
        with get_openai_callback() as cb:
            response = image_to_tex(image_path, prompt, model)
            responses.append(response)
            callback_data.append({
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_cost": cb.total_cost
            })
        logging.disable(logging.NOTSET)  # Re-enable logging
        
    return responses, callback_data
