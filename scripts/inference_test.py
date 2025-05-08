import typer
from pathlib import Path
import torch
# Use the specific class with LM head, importing from its submodule
from transformers import AutoTokenizer, AutoProcessor
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
import logging
from PIL import Image
import requests
from io import BytesIO

# Suppress specific warnings if needed (like slow processor warning)
# warnings.filterwarnings("ignore", category=UserWarning, message=".*Using a slow image processor.*"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer()

def load_image(image_path_or_url: str) -> Image.Image:
    """Loads an image from a local path or URL."""
    try:
        if image_path_or_url.startswith(("http://", "https://")):
            logger.info(f"Downloading image from URL: {image_path_or_url}")
            response = requests.get(image_path_or_url, stream=True)
            response.raise_for_status() # Raise an error for bad status codes
            image = Image.open(BytesIO(response.content))
        else:
            image_path = Path(image_path_or_url)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found at path: {image_path}")
            logger.info(f"Loading image from local path: {image_path}")
            image = Image.open(image_path)
        # Convert to RGB if necessary (some models might require it)
        if image.mode != "RGB":
            logger.debug(f"Converting image from {image.mode} to RGB")
            image = image.convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Failed to load image '{image_path_or_url}': {e}", exc_info=True)
        raise

@app.command()
def run_inference(
    model_path: Path = typer.Argument(
        ...,
        help="Path to the directory containing the resized model, merged tokenizer, and processor.",
        exists=True, file_okay=False, dir_okay=True, readable=True
    ),
    prompt_text: str = typer.Argument(..., help="Text prompt for the model."),
    image_input: str = typer.Option(
        None, "--image", "-i",
        help="Optional path or URL to an input image."
    ),
    max_new_tokens: int = typer.Option(100, help="Maximum number of new tokens to generate."),
    temperature: float = typer.Option(0.7, help="Sampling temperature."),
    top_k: int = typer.Option(50, help="Top-k sampling."),
    print_arch: bool = typer.Option(False, "--print-arch", help="Print the model architecture after loading."),
    # Add more generation parameters as needed (e.g., top_p, do_sample)
):
    """
    Loads the resized Qwen-VL model and performs inference with a given prompt (and optional image).
    """
    logger.info(f"Starting inference test.")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Prompt: '{prompt_text}'")
    if image_input:
        logger.info(f"Image input: {image_input}")

    # --- 1. Load Model, Tokenizer, Processor ---
    try:
        logger.info(f"Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        logger.info(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")

        logger.info(f"Loading processor from {model_path}...")
        processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        logger.info("Processor loaded.")

        logger.info(f"Loading model from {model_path} using Qwen2_5_VLForConditionalGeneration...")
        # Use the specific Qwen VL class WITH LM HEAD
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16, # Adjust dtype if needed
            trust_remote_code=True
        )
        logger.info("Model loaded.")

        # --- Optional: Print Architecture ---
        if print_arch:
            logger.info("Printing model architecture...")
            print("\n" + "="*20 + " Model Architecture " + "="*20)
            print(model)
            print("="* (40 + len(" Model Architecture ")) + "\n")
            # Optionally exit after printing if that's the desired behavior
            # raise typer.Exit()

        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        model.to(device)
        model.eval() # Set model to evaluation mode

    except Exception as e:
        logger.error(f"Failed to load components from {model_path}: {e}", exc_info=True)
        raise typer.Exit(code=1)

    # --- 2. Prepare Input ---
    logger.info("Preparing input prompt...")
    image = None
    messages = []
    try:
        if image_input:
            image = load_image(image_input)
            messages.append({"role": "user", "content": [{'image': image}, {'text': prompt_text}]})
        else:
            messages.append({"role": "user", "content": prompt_text})
        text = processor.tokenizer.apply_chat_template(
             messages,
             tokenize=False,
             add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image] if image else None, return_tensors="pt").to(device)
        logger.info("Input prepared successfully.")
    except Exception as e:
        logger.error(f"Failed to prepare input: {e}", exc_info=True)
        raise typer.Exit(code=1)

    # --- 3. Generate Output ---
    logger.info(f"Generating output (max_new_tokens={max_new_tokens})...")
    try:
        with torch.no_grad():
             generated_ids = model.generate(
                 **inputs,
                 max_new_tokens=max_new_tokens,
                 do_sample=True,
                 temperature=temperature,
                 top_k=top_k,
                 pad_token_id=tokenizer.eos_token_id # Important for sampling
             )
        input_token_len = inputs['input_ids'].shape[1]
        generated_tokens = generated_ids[:, input_token_len:]
        logger.info("Decoding generated tokens...")
        outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        logger.info("Generation complete.")
    except Exception as e:
        logger.error(f"Failed during generation: {e}", exc_info=True)
        raise typer.Exit(code=1)

    # --- 4. Print Output ---
    print("\n" + "-" * 10 + " Generated Output " + "-" * 10)
    for i, output in enumerate(outputs):
        print(f"Output {i+1}:")
        print(output.strip())
    print("-" * 38)

    logger.info("Inference test finished.")


if __name__ == "__main__":
    app() 
