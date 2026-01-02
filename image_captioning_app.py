"""
BLIP Image Captioning Demo with Gradio

A simple, standalone web application that generates natural language captions for uploaded images
using Salesforce's BLIP model (blip-image-captioning-base) from Hugging Face Transformers.

Features:
- Clean Gradio interface
- Conditional captioning with prompt "the image of" for better results
- Single-file script — easy to run and share

Requirements:
    gradio
    transformers
    torch
    pillow
    numpy

Installation:
    pip install gradio transformers torch pillow numpy

Quick Start:
    1. Save this file as image_captioning_app.py (or app.py for Hugging Face Spaces)
    2. Run: python image_captioning_app.py
    3. Open the local URL shown in terminal (usually http://127.0.0.1:7860)
    4. Upload any image and get an instant caption!

Note:
    - The model (~1 GB) will download automatically on first run.
    - For better captions, try the larger model: "Salesforce/blip-image-captioning-large"

Author: Piyush Kumar Singh
GitHub: https://github.com/PiyushKumarsing/blip-image-captioning-gradio
"""

import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the BLIP processor and model (downloaded automatically on first run)
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):
    """
    Generate a caption for the input image using BLIP conditional generation.
    
    Args:
        input_image (np.ndarray): Image array from Gradio (RGB format)
    
    Returns:
        str: Generated caption
    """
    # Convert numpy array to PIL Image and ensure RGB mode
    raw_image = Image.fromarray(input_image).convert("RGB")
    
    # Use conditional prompt for more descriptive captions
    text = "the image of"
    
    # Prepare inputs for the model
    inputs = processor(images=raw_image, text=text, return_tensors="pt")
    
    # Generate caption
    outputs = model.generate(**inputs, max_length=50)
    
    # Decode and clean up the output
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    return caption

# Create the Gradio interface
iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(label="Upload an Image"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="BLIP Image Captioning",
    description="Upload any image to automatically generate a descriptive caption using Salesforce's BLIP model.",
    article="""
    **How it works**: This app uses conditional image-to-text generation with the prompt "the image of" 
    to produce more accurate and natural captions.
    
    Try uploading photos of people, animals, landscapes, objects, or scenes!
    """,
    allow_flagging="never"  # Optional: cleaner look
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
