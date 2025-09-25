import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from SmartAITool.core import *



MID = "apple/FastVLM-0.5B"
IMAGE_TOKEN_INDEX = -200  # what the model code looks for

# Modify load_model_and_tokenizer to ensure global reuse
tok, model = None, None  # Initialize global variables

def load_model_and_tokenizer():
    """Load the model and tokenizer if not already loaded."""
    global tok, model
    if tok is None or model is None:
        tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
    return tok, model

def describe_image(image_path, tok, model):
    """Describe the given image using the loaded model and tokenizer."""
    # Build chat -> render to string (not tokens) so we can place <image> exactly
    messages = [
        {"role": "user", "content": "<image>\nDescribe this image in detail."}
    ]

    rendered = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    pre, post = rendered.split("<image>", 1)
    # Tokenize the text *around* the image token (no extra specials!)
    pre_ids = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids
    post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids
    # Splice in the IMAGE token id (-200) at the placeholder position
    img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
    input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    # Preprocess image via the model's own processor
    img = Image.open(image_path).convert("RGB")
    px = model.get_vision_tower().image_processor(images=img, return_tensors="pt")["pixel_values"]
    px = px.to(model.device, dtype=model.dtype)
    # Generate
    with torch.no_grad():
        out = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=px,
            max_new_tokens=256,  # Increase the token limit
        )
    return tok.decode(out[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    tok, model = load_model_and_tokenizer()
    image_description = describe_image("/home/ubuntu7/m15kh/vllm/frame_1.png", tok, model)
    cprint(image_description)
