import os
import json
import shutil
import typer
from pathlib import Path
# Use the core tokenizers library
from tokenizers import Tokenizer
from tokenizers.models import BPE
# Import the wrapper class
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from omnisvg_data_process.tokenizer import SVGTokenizer # Assuming SVGTokenizer is accessible

# Create a Typer app instance
app = typer.Typer()

# --- Configuration ---
# Qwen base model path (local path provided by user)
QWEN_MODEL_PATH = "/home/bluryar/code/omnisvg-data-process/model/Qwen/Qwen2.5-VL-3B-Instruct"
# Path to save the merged tokenizer files
MERGED_TOKENIZER_SAVE_DIR = "./model/merged_bpe_tokenizer"

def load_base_tokenizer_and_extract(model_path: str):
    """Loads the base Hugging Face tokenizer and extracts necessary components."""
    print(f"Loading base tokenizer from: {model_path}")
    try:
        qwen_tokenizer_hf = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        backend_tokenizer = qwen_tokenizer_hf.backend_tokenizer
        print(f"Original Qwen vocab size (reported by transformers): {qwen_tokenizer_hf.vocab_size}")

        if not isinstance(backend_tokenizer.model, BPE):
            print(f"Error: The base tokenizer's model is not BPE ({type(backend_tokenizer.model)}). This script requires a BPE model.")
            raise typer.Exit(code=1)

        # Extract Vocab
        qwen_vocab = backend_tokenizer.get_vocab(with_added_tokens=True)
        print(f"Extracted Qwen vocab size (from backend): {len(qwen_vocab)}")

        # Extract Merges
        qwen_merges = []
        if hasattr(backend_tokenizer.model, 'merges') and backend_tokenizer.model.merges:
             qwen_merges = backend_tokenizer.model.merges
             print(f"Extracted {len(qwen_merges)} merge rules from tokenizer model.")
        else:
            merges_file_path = Path(model_path) / "merges.txt"
            if merges_file_path.exists():
                print(f"Loading merges from: {merges_file_path}")
                with open(merges_file_path, "r", encoding="utf-8") as f:
                    first_line = f.readline()
                    if not first_line.startswith("#"):
                        f.seek(0)
                    for line in f:
                        line = line.strip()
                        if line:
                            qwen_merges.append(tuple(line.split()))
                print(f"Loaded {len(qwen_merges)} merge rules from file.")
            else:
                print("Error: Could not find merge rules in tokenizer model or merges.txt.")
                raise typer.Exit(code=1)

        # Extract BPE Params
        bpe_params = {
            "unk_token": backend_tokenizer.model.unk_token,
            "end_of_word_suffix": getattr(backend_tokenizer.model, 'end_of_word_suffix', None),
            "byte_fallback": getattr(backend_tokenizer.model, 'byte_fallback', False),
        }
        # Filter out None values explicitly for cleaner creation later
        bpe_params = {k: v for k, v in bpe_params.items() if v is not None}
        print(f"Extracted BPE params: {bpe_params}")

        return qwen_tokenizer_hf, backend_tokenizer, qwen_vocab, qwen_merges, bpe_params

    except Exception as e:
        print(f"Error loading Qwen tokenizer or extracting components: {e}")
        raise typer.Exit(code=1)

def build_svg_vocab():
    """Initializes SVGTokenizer and returns its vocabulary set."""
    print("Initializing SVGTokenizer...")
    try:
        svg_tokenizer_instance = SVGTokenizer()
        svg_vocab_set = svg_tokenizer_instance.vocabulary
        print(f"SVG vocabulary size: {len(svg_vocab_set)}")
        return svg_vocab_set
    except Exception as e:
        print(f"Error initializing SVGTokenizer or building vocab: {e}")
        raise typer.Exit(code=1)

def combine_vocabularies(base_vocab: dict, svg_vocab: set):
    """Combines the base vocabulary with the SVG vocabulary."""
    print("Combining vocabularies...")
    combined_vocab = base_vocab.copy()
    added_count = 0
    next_id = max(combined_vocab.values()) + 1 if combined_vocab else 0
    for token in sorted(list(svg_vocab)):
        if token not in combined_vocab:
            combined_vocab[token] = next_id
            next_id += 1
            added_count += 1
    print(f"Added {added_count} new SVG tokens. Total combined vocabulary size: {len(combined_vocab)}")
    return combined_vocab

def create_new_tokenizer(combined_vocab: dict, merges: list, bpe_params: dict, base_backend_tokenizer: Tokenizer):
    """Creates the new core BPE tokenizer model and instance, copying components."""
    print("Creating new BPE model instance using constructor...")
    try:
        new_bpe_model = BPE(
            vocab=combined_vocab,
            merges=merges,
            **bpe_params # Pass filtered params
        )
        print("New BPE model created.")
    except Exception as e:
        print(f"Error creating new BPE model: {e}")
        raise typer.Exit(code=1)

    print("Creating new Tokenizer instance...")
    new_tokenizer = Tokenizer(new_bpe_model)
    # Copy components
    print("Copying Normalizer, PreTokenizer, Decoder, PostProcessor...")
    if base_backend_tokenizer.normalizer:
        new_tokenizer.normalizer = base_backend_tokenizer.normalizer
        print(f"  Copied Normalizer: {type(new_tokenizer.normalizer)}")
    if base_backend_tokenizer.pre_tokenizer:
        new_tokenizer.pre_tokenizer = base_backend_tokenizer.pre_tokenizer
        print(f"  Copied PreTokenizer: {type(new_tokenizer.pre_tokenizer)}")
    if base_backend_tokenizer.decoder:
        new_tokenizer.decoder = base_backend_tokenizer.decoder
        print(f"  Copied Decoder: {type(new_tokenizer.decoder)}")
    if base_backend_tokenizer.post_processor:
        new_tokenizer.post_processor = base_backend_tokenizer.post_processor
        print(f"  Copied PostProcessor: {type(new_tokenizer.post_processor)}")
    print("Copied components to new Tokenizer.")
    return new_tokenizer

def create_hf_wrapper(new_core_tokenizer: Tokenizer, base_hf_tokenizer: PreTrainedTokenizerFast):
    """Creates the PreTrainedTokenizerFast wrapper."""
    print("Creating PreTrainedTokenizerFast wrapper...")
    explicit_args = {
        "tokenizer_object",
        "bos_token", "eos_token", "unk_token", "pad_token",
        "model_max_length", "padding_side", "truncation_side",
        "special_tokens_map"
    }
    filtered_kwargs = {
        k: v for k, v in base_hf_tokenizer.init_kwargs.items()
        if k not in explicit_args
    }
    print(f"Filtered kwargs for PreTrainedTokenizerFast: {filtered_kwargs.keys()}")

    new_hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=new_core_tokenizer,
        bos_token=base_hf_tokenizer.bos_token,
        eos_token=base_hf_tokenizer.eos_token,
        unk_token=base_hf_tokenizer.unk_token,
        pad_token=base_hf_tokenizer.pad_token,
        model_max_length=base_hf_tokenizer.model_max_length,
        padding_side=base_hf_tokenizer.padding_side,
        truncation_side=base_hf_tokenizer.truncation_side,
        special_tokens_map=base_hf_tokenizer.special_tokens_map,
        **filtered_kwargs
    )
    print("PreTrainedTokenizerFast wrapper created.")
    return new_hf_tokenizer

def save_merged_tokenizer(hf_wrapper: PreTrainedTokenizerFast, combined_vocab: dict, save_dir: Path, base_model_path: Path):
    """Saves the merged tokenizer files (wrapper, vocab.json, merges.txt)."""
    print(f"Saving merged tokenizer to: {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save using the wrapper's save_pretrained
        print("Saving tokenizer files using PreTrainedTokenizerFast.save_pretrained...")
        hf_wrapper.save_pretrained(str(save_dir)) # save_pretrained expects string path
        print(f"Saved tokenizer.json and config files to {save_dir}")

        # Manually save the combined vocabulary to vocab.json
        vocab_save_path = save_dir / "vocab.json"
        print(f"Manually saving combined vocabulary to: {vocab_save_path}")
        with open(vocab_save_path, "w", encoding="utf-8") as f:
            json.dump(combined_vocab, f, ensure_ascii=False, indent=2)
        print("vocab.json saved successfully.")

        # Copy merges.txt if it exists
        merges_file_path_src = base_model_path / "merges.txt"
        merges_file_path_dest = save_dir / "merges.txt"
        if merges_file_path_src.exists():
            print("Copying merges.txt...")
            shutil.copyfile(merges_file_path_src, merges_file_path_dest)

    except Exception as e:
        print(f"Error saving tokenizer files: {e}")
        raise typer.Exit(code=1)

def verify_tokenizer(save_dir: Path):
    """Loads and performs basic verification on the saved tokenizer."""
    print("Verifying the saved tokenizer...")
    try:
        loaded_hf_tokenizer = AutoTokenizer.from_pretrained(str(save_dir), trust_remote_code=True)
        print(f"Successfully loaded saved tokenizer with AutoTokenizer. Vocab size: {loaded_hf_tokenizer.vocab_size}")
        print(f"Length of tokenizer (includes added/special): {len(loaded_hf_tokenizer)}")

        # Verify SVG token
        svg_token_example = "<｜SVG_START｜>"
        token_id = loaded_hf_tokenizer.convert_tokens_to_ids(svg_token_example)
        # Check if the token exists in the vocab
        if token_id == loaded_hf_tokenizer.unk_token_id and svg_token_example != loaded_hf_tokenizer.unk_token:
             print(f"Verification Warning: SVG token '{svg_token_example}' was not found in the loaded tokenizer (ID: {token_id}).")
        else:
            reconstructed = loaded_hf_tokenizer.convert_ids_to_tokens(token_id)
            print(f"Test: SVG token '{svg_token_example}' -> ID {token_id} -> Reconstructed '{reconstructed}'")
            assert reconstructed == svg_token_example, f"Token mismatch: {reconstructed} != {svg_token_example}"
            print("SVG token verification successful.")

    except Exception as e:
        print(f"Error verifying tokenizer files: {e}")
        # Don't exit, just report the error

@app.command()
def merge_tokenizers(
    base_model_path: Path = typer.Argument(..., help="Path to the base Hugging Face tokenizer directory.", exists=True, file_okay=False, dir_okay=True, readable=True),
    output_dir: Path = typer.Argument(..., help="Directory to save the merged tokenizer files.", file_okay=False, dir_okay=True, writable=True)
):
    """Merges a base BPE tokenizer with SVG tokens and saves the result."""

    # --- Main Workflow --- 
    # 1. Load Base Tokenizer & Extract
    qwen_tokenizer_hf, backend_tokenizer, qwen_vocab, qwen_merges, bpe_params = load_base_tokenizer_and_extract(str(base_model_path))

    # 2. Build SVG Vocab
    svg_vocab_set = build_svg_vocab()

    # 3. Combine Vocabularies
    combined_vocab = combine_vocabularies(qwen_vocab, svg_vocab_set)

    # 4. Create New Core Tokenizer
    new_core_tokenizer = create_new_tokenizer(combined_vocab, qwen_merges, bpe_params, backend_tokenizer)

    # 5. Create HF Wrapper
    new_hf_tokenizer = create_hf_wrapper(new_core_tokenizer, qwen_tokenizer_hf)

    # 6. Save Merged Tokenizer
    save_merged_tokenizer(new_hf_tokenizer, combined_vocab, output_dir, base_model_path)

    # 7. Verify (Optional but recommended)
    verify_tokenizer(output_dir)

    print("Tokenizer merging process complete.")


if __name__ == "__main__":
    app() 
