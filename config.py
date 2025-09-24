# config.py

import torch

# --- Model Configurations ---
# GENERAL_CHAT_MODEL_NAME: Using distilgpt2 as it's lighter and faster for demonstration.
# VQA_MODEL_NAME: Salesforce/blip-vqa-base is used as it's a standard choice for Visual Question Answering.
GENERAL_CHAT_MODEL_NAME = "distilgpt2"
VQA_MODEL_NAME = "Salesforce/blip-vqa-base"

# Quantization settings for GPU (requires bitsandbytes and accelerate libraries)
# Set USE_4BIT_QUANTIZATION = False if you face installation issues or don't have a compatible NVIDIA GPU.
USE_4BIT_QUANTIZATION = True
BNB_4BIT_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16 # Recommended for better performance on modern GPUs
}

# --- Knowledge Base (Simplified for demonstration) ---
DOMAIN_KNOWLEDGE_BASES = {
    "general": {
        "greeting": "Hello! How can I assist you today?",
        "farewell": "Goodbye! Have a great day.",
        "capabilities": "I can help with product inquiries, technical support, and general questions. You can also upload images for visual questions and voice inputs.",
        "default": "I'm not sure how to respond to that. Could you please rephrase or ask about our products or services?",
        "joke": "Why don't scientists trust atoms? Because they make up everything!",
    },
    "product_support": {
        "product_info": "Please specify which product you are interested in. We have information on Product A, Product B, and Product C.",
        "product_A": "Product A is a high-performance gadget with features X, Y, and Z. It costs $100. It's great for gaming!",
        "product_B": "Product B is a budget-friendly option with features P, Q, and R. It costs $50. Perfect for everyday use.",
        "warranty": "Our products come with a one-year warranty. You can find more details on our website or contact us directly.",
        "shipping": "Shipping typically takes 3-5 business days within India. You'll receive a tracking number via email once your order ships."
    },
    "technical_support": {
        "troubleshoot_internet": "Please check your router and modem connections. If the issue persists, try restarting them. If still no luck, describe any error messages you see.",
        "reset_password": "To reset your password, visit our login page and click 'Forgot Password'.",
        "software_issue": "Can you describe the software issue in more detail? Which software are you using and what exactly is happening?",
        "common_issues": "I can help with common technical issues like internet troubleshooting, password resets, and software problems. What specific technical problem are you facing?"
    },
    "image_query": {
        "unidentified_image": "I received an image. What specific question do you have about it?",
        "simulate_vqa_response": lambda query, image_desc: f"Based on the image which seems to show {image_desc}, and your question about '{query}', I can tell you that this is a simulated visual information response. (Real VQA model not used or failed)."
    },
    "voice_query": {
        "unidentified_voice": "I received a voice input. What would you like to ask?",
        "simulate_voice_response": lambda transcribed_text: f"I understood your voice input as: '{transcribed_text}'. How can I assist further based on this?"
    }
}

# --- Response Generation Parameters for LLM (DistilGPT2) ---
GENERATION_PARAMS = {
    "max_new_tokens": 60, # Maximum number of tokens the LLM will generate
    "num_return_sequences": 1, # Generate only one sequence
    "do_sample": True, # Use sampling for more varied responses (less deterministic)
    "top_k": 50, # Consider only the top 50 most likely next tokens
    "top_p": 0.95, # Consider tokens whose cumulative probability sum up to 95%
    "temperature": 0.7 # Controls randomness: higher = more creative, lower = more focused
}
