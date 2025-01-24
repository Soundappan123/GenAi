import streamlit as st
import os
import base64
from PIL import Image
from vllm import LLM
from vllm.sampling_params import SamplingParams
from huggingface_hub import login
import io
import torch

# Set up Hugging Face token (replace with your token)
os.environ["HF_TOKEN"] = "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"  
login(token=os.environ['HF_TOKEN'])

# Specify device (CPU or CUDA)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define Pixtral model details
model_name = "mistralai/Pixtral-12B-2409"
sampling_params = SamplingParams(max_tokens=8192, temperature=0.7)

# Initialize Pixtral model with device configuration
llm = LLM(model=model_name, tokenizer_mode="mistral", max_model_len=10000, device=device)

def encode_image_base64(image):
    """Encodes an image to a base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Streamlit UI setup
st.set_page_config(page_title="Pixtral Image Text Extraction", layout="wide")
st.title("Upload Images to Extract Text Using Pixtral")

# File uploader for image(s)
uploaded_files = st.file_uploader("Choose an image...", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files:
    # Process uploaded images
    content = []
    for uploaded_file in uploaded_files:
        # Open and process the image
        image = Image.open(uploaded_file)
        img_base64 = encode_image_base64(image)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}})

    def llm_chat(single_message, sampling_params):
        # Simulating output as if Pixtral processed it
        return {"choices": [{"message": {"content": "Extracted text from image"}}]}

    outputs = []
    st.write("Processing images...")

    for i, image_content in enumerate(content):
        single_message = [{"role": "user", "content": [image_content]}]
        try:
            output = llm.chat(single_message, sampling_params=sampling_params)
            st.write(f"Debug: Raw output for Image {i + 1}:\n{output}")  # Debug: Print raw output
            outputs.append(output)
        except Exception as e:
            st.write(f"Error processing Image {i + 1}: {e}")
            outputs.append(None)

    # Display the outputs
    for i, output in enumerate(outputs):
        if output:
            extracted_text = output["choices"][0]["message"]["content"]
            st.write(f"Extracted text from Image {i + 1}:")
            st.write(extracted_text)
        else:
            st.write(f"Failed to extract text from Image {i + 1}.")
else:
    st.write("Please upload at least one image to start the processing.")
