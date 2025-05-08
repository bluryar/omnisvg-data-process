import typer
from pathlib import Path
import torch # Typically needed, even if indirectly
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
import logging
import json
import shutil

# Import necessary functions from the merge_tokenizer script
from scripts.merge_tokenizer import (
    load_base_tokenizer_and_extract,
    build_svg_vocab,
    combine_vocabularies,
    create_new_tokenizer,
    create_hf_wrapper,
    # We might also need save_merged_tokenizer if we want its logic exactly,
    # but saving the final tokenizer happens naturally in this script.
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer()

# --- Main Command Function --- 

@app.command()
def merge_resize_save(
    base_model_path: Path = typer.Argument(
        ...,
        help="Path to the original base Hugging Face model directory (contains model, tokenizer, processor).",
        exists=True, file_okay=False, dir_okay=True, readable=True
    ),
    output_path: Path = typer.Argument(
        ...,
        help="Directory to save the merged tokenizer, resized model, and processor.",
        file_okay=False, dir_okay=True, writable=True
    )
):
    """
    Merges SVG tokens into the base model's tokenizer (using functions from merge_tokenizer.py),
    resizes the model embeddings accordingly, and saves the merged tokenizer, resized model,
    and original processor to the output path.
    """
    logger.info("Starting tokenizer merge and model resize process.")
    logger.info(f"Base model path: {base_model_path}")
    logger.info(f"Output path: {output_path}")

    # --- 1. Merge Tokenizer Logic (using imported functions) ---
    try:
        # Load base tokenizer and extract components
        logger.info("Loading base tokenizer and extracting components...")
        base_hf_tokenizer, backend_tokenizer, base_vocab, merges, bpe_params = load_base_tokenizer_and_extract(str(base_model_path))

        # Build SVG vocab
        logger.info("Building SVG vocabulary...")
        svg_vocab_set = build_svg_vocab()

        # Combine vocabularies
        logger.info("Combining vocabularies...")
        combined_vocab = combine_vocabularies(base_vocab, svg_vocab_set)
        new_vocab_size = len(combined_vocab)

        # Create new core tokenizer
        logger.info("Creating new core tokenizer...")
        new_core_tokenizer = create_new_tokenizer(combined_vocab, merges, bpe_params, backend_tokenizer)

        # Create HF wrapper for the new tokenizer
        logger.info("Creating HF wrapper for merged tokenizer...")
        merged_hf_tokenizer = create_hf_wrapper(new_core_tokenizer, base_hf_tokenizer)
        logger.info(f"Merged tokenizer created in memory. New vocab size: {new_vocab_size}")

    except Exception as e:
        logger.error(f"Failed during tokenizer merging phase: {e}", exc_info=True)
        raise typer.Exit(code=1)

    # --- 2. Load Original Model and Processor ---
    try:
        logger.info(f"Loading base model configuration from {base_model_path}...")
        config = AutoConfig.from_pretrained(str(base_model_path), trust_remote_code=True)
        original_vocab_size = config.vocab_size
        logger.info(f"Original model config vocabulary size: {original_vocab_size}")

        logger.info(f"Loading base model from {base_model_path} using specific class Qwen2_5_VLForConditionalGeneration...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(base_model_path),
            config=config, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
        )
        logger.info(f"Base model loaded: {model.config.model_type}")

        logger.info(f"Loading processor from {base_model_path}...")
        processor = AutoProcessor.from_pretrained(str(base_model_path), trust_remote_code=True)
        logger.info("Processor loaded.")

    except Exception as e:
        logger.error(f"Failed to load base model, processor, or config from {base_model_path}: {e}", exc_info=True)
        raise typer.Exit(code=1)

    # --- 3. Resize Token Embeddings if Necessary ---
    if new_vocab_size != original_vocab_size:
        logger.info(f"Vocabulary size mismatch: Model ({original_vocab_size}) vs Merged Tokenizer ({new_vocab_size}). Resizing model embeddings...")
        try:
            model.resize_token_embeddings(new_vocab_size)
            logger.info("Model embedding layer resized.")
            logger.info(f"Model config vocab size after resize: {model.config.vocab_size}")

            if model.config.vocab_size != new_vocab_size:
                 logger.warning(f"Model config vocab size ({model.config.vocab_size}) does not match new vocab size ({new_vocab_size}) after resize. Manually setting.")
                 model.config.vocab_size = new_vocab_size

            tie_word_embeddings = getattr(model.config, 'tie_word_embeddings', False)
            if tie_word_embeddings:
                logger.info("`tie_word_embeddings` is True. LM head was likely resized automatically.")
            else:
                logger.info("`tie_word_embeddings` is False. Manually resizing LM head.")
                old_lm_head = model.get_output_embeddings()
                if old_lm_head is not None:
                    hidden_size = model.config.hidden_size # Get hidden size from model config
                    logger.info(f"Creating new LM head with output size {new_vocab_size}")
                    new_lm_head = torch.nn.Linear(hidden_size, new_vocab_size, bias=getattr(old_lm_head, 'bias', None) is not None)
                    new_lm_head = new_lm_head.to(device=model.device, dtype=model.dtype)
                    logger.info("Copying existing weights to new LM head...")
                    new_lm_head.weight.data[:original_vocab_size, :] = old_lm_head.weight.data[:original_vocab_size, :]
                    logger.info("Initializing new LM head weights based on new input embeddings...")
                    input_embeddings_data = model.get_input_embeddings().weight.data
                    # Ensure shapes match before copying
                    if new_lm_head.weight.data[original_vocab_size:, :].shape == input_embeddings_data[original_vocab_size:, :].shape:
                        new_lm_head.weight.data[original_vocab_size:, :] = input_embeddings_data[original_vocab_size:, :]
                    else:
                         logger.warning(f"Shape mismatch initializing new LM head weights. Expected {new_lm_head.weight.data[original_vocab_size:, :].shape}, got {input_embeddings_data[original_vocab_size:, :].shape}. Using mean initialization instead.")
                         # Fallback: Initialize new weights with the mean of the original weights
                         mean_weights = old_lm_head.weight.data.mean(dim=0, keepdim=True)
                         new_lm_head.weight.data[original_vocab_size:, :] = mean_weights.repeat(new_vocab_size - original_vocab_size, 1)

                    if getattr(old_lm_head, 'bias', None) is not None:
                        logger.info("Copying existing bias to new LM head...")
                        new_lm_head.bias.data[:original_vocab_size] = old_lm_head.bias.data[:original_vocab_size]
                        logger.info("Initializing new LM head bias terms to zero...")
                        new_lm_head.bias.data[original_vocab_size:].zero_()
                    model.lm_head = new_lm_head
                    logger.info("Manual LM head resizing complete.")
                else:
                    logger.warning("Could not find output embeddings/LM head to resize manually.")
        except Exception as e:
            logger.error(f"Failed during model embedding or LM head resize: {e}", exc_info=True)
            raise typer.Exit(code=1)
    else:
        logger.info("Model vocabulary size already matches tokenizer vocabulary size. No resizing needed.")

    # --- 4. Save Resized Model, Merged Tokenizer, and Processor ---
    logger.info(f"Saving resized model, merged tokenizer, and processor to {output_path}...")
    try:
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Saving model...")
        model.save_pretrained(str(output_path))

        logger.info("Saving merged tokenizer...")
        merged_hf_tokenizer.save_pretrained(str(output_path))

        logger.info("Saving processor...")
        processor.save_pretrained(str(output_path))

        # Manually save the combined vocabulary to vocab.json for reference/compatibility
        # We need the combined_vocab dictionary from the merge step
        vocab_save_path = output_path / "vocab.json"
        logger.info(f"Manually saving combined vocabulary to: {vocab_save_path}")
        with open(vocab_save_path, "w", encoding="utf-8") as f:
            # Ensure combined_vocab is accessible here - it was created in step 1
            json.dump(combined_vocab, f, ensure_ascii=False, indent=2)

        # Copy merges.txt if it exists
        merges_file_path_src = base_model_path / "merges.txt"
        merges_file_path_dest = output_path / "merges.txt"
        if merges_file_path_src.exists():
            logger.info("Copying merges.txt...")
            shutil.copyfile(merges_file_path_src, merges_file_path_dest)

        logger.info("All components saved successfully.")

    except Exception as e:
        logger.error(f"Failed to save components to {output_path}: {e}", exc_info=True)
        raise typer.Exit(code=1)

    logger.info("Tokenizer merge, model resize, and saving process complete.")

if __name__ == "__main__":
    app() 
