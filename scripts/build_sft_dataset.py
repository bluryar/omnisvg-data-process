import csv
import json
import uuid
import logging
import argparse
from pathlib import Path
import cairosvg
from tqdm import tqdm
import sys
import os
import random
import multiprocessing
from functools import partial
from typing import Dict, Any, Optional, Tuple, List

# --- Configuration ---
# Add the project root to the Python path to find omnisvg_data_process
# Adjust this path if your script is located elsewhere relative to the package
project_root = Path(__file__).resolve().parent.parent # Assumes script is in 'scripts' or similar, adjust if needed
sys.path.insert(0, str(project_root))

try:
    from omnisvg_data_process.tokenizer import SVGTokenizer
    from omnisvg_data_process.constants import VIEWBOX_WIDTH, VIEWBOX_HEIGHT
except ImportError as e:
    print(f"Error: Could not import omnisvg_data_process modules.")
    print(f"Ensure the package is installed or the project root ({project_root}) is in PYTHONPATH.")
    print(f"Import error details: {e}")
    sys.exit(1)

# --- Predefined Instructions ---
TEXT_INSTRUCTIONS = [
    "Generate an SVG based on the following description:",
    "Create a vector graphic for:",
    "Produce an SVG illustration depicting:",
    "I need an SVG file that represents:",
    "Can you make an SVG of:",
    "Illustrate the following text as an SVG:",
    "Generate SVG code for this description:",
    "Render this concept visually in SVG:",
    "What would an SVG for this look like?",
    "Create an SVG graphic representing:",
    "根据以下描述生成一个SVG：",
    "为以下内容创建一个矢量图形：",
    "绘制一个SVG插画，描绘：",
    "我需要一个代表以下内容的SVG文件：",
    "你能制作一个关于...的SVG吗：",
    "将以下文字绘制成SVG图形：",
    "为这段描述生成SVG代码：",
    "用SVG视觉化呈现这个概念：",
    "这个描述对应的SVG应该是什么样子？",
    "Translate this description into an SVG:",
    "Vectorize the concept described as:",
    "将这个描述转换成SVG：",
    "把描述的概念矢量化为：",
    "为这段文字设计一个SVG图标：",
]

IMAGE_INSTRUCTIONS = [
    "Convert the provided image into an SVG format.",
    "Generate an SVG representation of the input image.",
    "Create a vector graphic based on the image.",
    "Vectorize the given image.",
    "I need an SVG version of this picture.",
    "Trace this image into an SVG file.",
    "Convert this picture to a vector SVG.",
    "Give me the SVG vector for this image.",
    "Analyze the image and produce its SVG equivalent.",
    "Recreate this image using SVG paths.",
    "将提供的图像转换为SVG格式。",
    "生成输入图像的SVG表示。",
    "根据这张图片创建一个矢量图形。",
    "矢量化给定的图像。",
    "我需要这张图片的SVG版本。",
    "将这张图描摹成SVG文件。",
    "把这张图片转换成矢量SVG。",
    "给我这张图片的SVG矢量图。",
    "分析图像并生成其SVG等价物。",
    "使用SVG路径重新创建这张图片。",
    "Produce an SVG from the following image.",
    "从以下图片生成SVG。",
    "What is the SVG equivalent of this image?",
    "这张图片的SVG等价物是什么？",
    "Turn this raster image into a vector SVG.",
    "把这个光栅图转换成矢量SVG。"
]

# --- Global variable for worker process data ---
worker_data = {}

# --- Logging Setup & Worker Initialization ---
def initialize_worker():
    """Initializes logging and tokenizer for each worker process."""
    global worker_data
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    # Initialize tokenizer for this worker
    worker_data['tokenizer'] = SVGTokenizer()
    logging.info("Worker process initialized (Tokenizer created).")

# --- Helper Functions ---
def ensure_dir(path: Path):
    """Creates a directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)

def rasterize_svg(svg_content: str, output_path: Path) -> bool:
    """Converts an SVG string to a PNG file using cairosvg."""
    try:
        svg_bytes = svg_content.encode('utf-8')
        cairosvg.svg2png(bytestring=svg_bytes, write_to=str(output_path), output_width=VIEWBOX_WIDTH, output_height=VIEWBOX_HEIGHT)
        return True
    except Exception as e:
        # Reduced logging severity for rasterization errors as we proceed without the image
        logging.debug(f"Failed to rasterize SVG to {output_path}: {e}", exc_info=False)
        return False

def get_file_encoding(input_path: Path) -> Optional[str]:
    """Tries to detect the encoding of a file."""
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1']
    for enc in encodings_to_try:
        try:
            with open(input_path, 'r', encoding=enc) as f:
                f.read(1024) # Try reading a small chunk
            # logging.info(f"Detected encoding '{enc}' for {input_path.name}")
            return enc
        except UnicodeDecodeError:
            continue
        except Exception as e:
             logging.warning(f"Error detecting encoding for {input_path.name} with {enc}: {e}")
             break # Stop trying encodings if another error occurs
    logging.error(f"Could not determine encoding for {input_path.name}")
    return None

# --- Single Item Processing Function (for multiprocessing) ---
def process_single_item(item_data: Dict[str, Any], images_dir: Path) -> Optional[Dict[str, Any]]:
    """Processes a single SVG data item.

    Uses the tokenizer initialized globally within the worker process.
    """
    # Get tokenizer from worker-specific global data
    tokenizer = worker_data.get('tokenizer')
    if not tokenizer:
        logging.error("Tokenizer not found in worker process data. Skipping item.")
        return None

    original_id = item_data.get('id', 'N/A')
    svg_content = item_data.get('svg')
    description = str(item_data.get('description', '')).replace('<image>', '').replace('<video>', '').replace('<audio>', '')

    if not svg_content or not description:
        logging.warning(f"Skipping item (ID: {original_id}) due to missing SVG or description.")
        return None # Indicate an error/skip for this item

    try:
        # 1. Generate UUID
        entry_uuid = str(uuid.uuid4())

        # 2. Normalize SVG
        normalized_svg = tokenizer.normalize_svg(svg_content)
        if not normalized_svg:
            logging.warning(f"Skipping item (UUID: {entry_uuid}, ID: {original_id}): Normalization failed or resulted in empty SVG.")
            return None

        # 3. Tokenize SVG
        svg_tokens = tokenizer.svg_to_tokens(normalized_svg)
        if svg_tokens is None:
            logging.warning(f"Skipping item (UUID: {entry_uuid}, ID: {original_id}): Tokenization failed.")
            return None # Indicate tokenization failure

        # 4. Rasterize SVG to PNG
        png_filename = f"{entry_uuid}.png"
        png_path_abs = images_dir / png_filename
        png_path_rel = Path("images") / png_filename # Relative path for the dataset field
        rasterization_successful = rasterize_svg(normalized_svg, png_path_abs)

        image_list = [str(png_path_rel)] if rasterization_successful else []

        # 5. Prepare output JSON object
        random_text_instruction = random.choice(TEXT_INSTRUCTIONS)
        random_image_instruction = random.choice(IMAGE_INSTRUCTIONS)

        output_data = {
            "uuid": entry_uuid,
            "svg": svg_content, # Keep original SVG
            "description": f"{description}",
            "text_instruction": random_text_instruction,
            "image_instruction": random_image_instruction,
            "text_input": f"{description}",
            "image_input": "<image>",
            "output": svg_tokens if svg_tokens else "",
            "empty_images": [],
            "images": image_list,
            "_rasterization_successful": rasterization_successful, # Internal flag
            "_tokenization_successful": True # Flag success
        }
        return output_data

    except Exception as e:
        logging.error(f"Unexpected error processing item (ID: {original_id}): {e}", exc_info=True)
        return None # Indicate general processing error

# --- Main Data Loading and Processing Function ---
def load_and_process_data(input_files: list[str], output_file: str, num_processes: Optional[int] = None):
    """
    Loads data from various file formats (CSV, JSON, JSONL), processes them in parallel,
    and saves the results to a JSON Lines file.
    """
    output_path = Path(output_file)
    images_dir = output_path.parent / "images"
    ensure_dir(images_dir)
    logging.info(f"Output directory: {output_path.parent}")
    logging.info(f"Images directory: {images_dir}")

    # --- Data Loading ---
    all_items_to_process: List[Dict[str, Any]] = []
    initial_item_count = 0
    loading_errors = 0

    for input_file in input_files:
        input_path = Path(input_file)
        if not input_path.exists():
            logging.warning(f"Input file not found: {input_file}, skipping.")
            continue

        logging.info(f"Loading data from: {input_file}")
        file_encoding = get_file_encoding(input_path)
        if not file_encoding:
            loading_errors += 1 # Count file as error if encoding fails
            continue

        try:
            with open(input_path, 'r', encoding=file_encoding, errors='ignore') as infile:
                file_extension = input_path.suffix.lower()
                current_file_items = 0

                if file_extension == ".csv":
                    # Handle potentially large fields
                    try:
                        # Try setting a large limit, fallback if it fails (e.g., in restricted envs)
                        csv.field_size_limit(min(2147483647, sys.maxsize))
                    except OverflowError:
                         logging.warning(f"Could not set max CSV field size limit on this system for {input_file}.")

                    reader = csv.reader(infile)
                    try:
                        header = next(reader) # Skip header row
                        header_lower = [h.lower().strip() for h in header]
                        logging.info(f"CSV Header in {input_path.name}: {header}")
                        # Find column indices dynamically (adjust expected names if needed)
                        try:
                            id_col_idx = header_lower.index('id')
                        except ValueError:
                            id_col_idx = -1 # Indicate 'id' is optional or missing
                            logging.warning(f"'id' column not found in {input_path.name}, will use N/A.")
                        svg_col_idx = header_lower.index('svg')
                        desc_col_idx = header_lower.index('description')
                    except StopIteration:
                         logging.warning(f"Input file {input_file} is empty or has no header, skipping.")
                         continue
                    except (ValueError, csv.Error) as e:
                         logging.error(f"Error reading header or finding required columns ('svg', 'description') in {input_file}: {e}. Skipping file.")
                         loading_errors += 1 # Count file as error
                         continue

                    for i, row in enumerate(reader):
                         initial_item_count += 1
                         try:
                             if len(row) <= max(svg_col_idx, desc_col_idx, (id_col_idx if id_col_idx != -1 else -1) ):
                                 logging.warning(f"Skipping row {i+1} in {input_file}: Insufficient columns. Row starts with: {row[:3]}")
                                 loading_errors += 1
                                 continue
                             item = {
                                 'id': row[id_col_idx] if id_col_idx != -1 else f"row_{i+1}",
                                 'svg': row[svg_col_idx],
                                 'description': row[desc_col_idx]
                             }
                             all_items_to_process.append(item)
                             current_file_items += 1
                         except csv.Error as e:
                            logging.error(f"CSV Error processing row {i+1} in {input_file}: {e}. Skipping row.")
                            loading_errors += 1
                         except IndexError:
                             logging.warning(f"Skipping row {i+1} in {input_file}: Index out of bounds accessing required columns. Row starts with: {row[:3]}")
                             loading_errors += 1


                elif file_extension == ".jsonl":
                    for i, line in enumerate(infile):
                        initial_item_count += 1
                        try:
                            item = json.loads(line)
                            if 'svg' not in item or 'description' not in item:
                                logging.warning(f"Skipping line {i+1} in {input_file}: Missing 'svg' or 'description' key. Keys found: {list(item.keys())}")
                                loading_errors += 1
                                continue
                            # Ensure 'id' exists, even if null/missing in source
                            if 'id' not in item: item['id'] = f"line_{i+1}"
                            all_items_to_process.append(item)
                            current_file_items += 1
                        except json.JSONDecodeError as e:
                            logging.error(f"JSONL Error decoding line {i+1} in {input_file}: {e}. Skipping line.")
                            loading_errors += 1
                        except Exception as e:
                             logging.error(f"Unexpected error processing line {i+1} in {input_file}: {e}", exc_info=True)
                             loading_errors += 1

                elif file_extension == ".json":
                    try:
                        data = json.load(infile)
                        if not isinstance(data, list):
                            logging.error(f"Error in {input_file}: JSON root is not a list. Skipping file.")
                            loading_errors += 1 # Count file as error
                            continue

                        for i, item in enumerate(data):
                            initial_item_count += 1
                            if not isinstance(item, dict):
                                logging.warning(f"Skipping item #{i+1} in {input_file}: Not a dictionary. Found type: {type(item)}")
                                loading_errors += 1
                                continue
                            if 'svg' not in item or 'description' not in item:
                                logging.warning(f"Skipping item #{i+1} in {input_file}: Missing 'svg' or 'description' key. Keys found: {list(item.keys())}")
                                loading_errors += 1
                                continue
                            # Ensure 'id' exists
                            if 'id' not in item: item['id'] = f"item_{i+1}"
                            all_items_to_process.append(item)
                            current_file_items += 1
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON Error decoding {input_file}: {e}. Skipping file.")
                        loading_errors += 1 # Count file as error
                        continue
                    except Exception as e:
                        logging.error(f"Unexpected error reading {input_file}: {e}", exc_info=True)
                        loading_errors += 1 # Count file as error
                        continue

                else:
                    logging.warning(f"Unsupported file extension '{file_extension}' for {input_file}. Skipping.")
                    continue

                logging.info(f"Loaded {current_file_items} items from {input_path.name}")

        except FileNotFoundError:
            logging.error(f"Input file not found during loading: {input_file}")
            loading_errors += 1
        except IOError as e:
            logging.error(f"I/O Error loading {input_file}: {e}")
            loading_errors += 1
        except Exception as e:
            logging.error(f"Failed to load or parse file {input_file}: {e}", exc_info=True)
            loading_errors += 1

    logging.info(f"Total items loaded for processing: {len(all_items_to_process)} (Initial items found: {initial_item_count}, Loading errors/skips: {loading_errors})")

    if not all_items_to_process:
        logging.warning("No valid items found to process. Exiting.")
        return

    # --- Parallel Processing ---
    processed_count = 0
    error_count = loading_errors # Start error count with loading errors
    skipped_rasterization = 0
    skipped_tokenization = 0 # Renamed from skipped_tokenization_failure for clarity

    # Determine number of processes
    if num_processes is None:
        num_processes = max(1, os.cpu_count() - 1 if os.cpu_count() else 1) # Default to N-1 CPUs
    logging.info(f"Starting parallel processing with {num_processes} processes.") # Log from main process


    try:
        # Use functools.partial to fix the images_dir argument (tokenizer removed)
        processing_function = partial(process_single_item, images_dir=images_dir)

        # Set initializer to configure logging and tokenizer in each worker
        with multiprocessing.Pool(processes=num_processes, initializer=initialize_worker) as pool, \
             open(output_path, 'w', encoding='utf-8') as outfile:

            # Use imap_unordered for potentially better performance with I/O or uneven task times
            results_iterator = pool.imap_unordered(processing_function, all_items_to_process)

            # Process results as they become available
            for result in tqdm(results_iterator, total=len(all_items_to_process), desc="Processing items", unit="item"):
                if result is not None:
                    # Check internal flags
                    if not result.get("_tokenization_successful", True): # Should always be true if result is not None based on process_single_item logic
                         skipped_tokenization += 1
                         error_count += 1 # Count tokenization failure as error
                         continue # Don't write if tokenization failed

                    if not result.get("_rasterization_successful", True):
                         skipped_rasterization += 1
                         # Note: We still write the item even if rasterization fails

                    # Remove internal flags before writing
                    result.pop("_rasterization_successful", None)
                    result.pop("_tokenization_successful", None)

                    # Write the valid processed item to the output file
                    outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                    processed_count += 1
                else:
                    # Result is None, indicating an error occurred in process_single_item or item was skipped early
                    error_count += 1 # Increment error count for items that failed processing

    except IOError as e:
        logging.error(f"Failed to open or write to output file {output_path}: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"An unexpected error occurred during parallel processing: {e}", exc_info=True)


    # --- Final Summary ---
    logging.info("=" * 30)
    logging.info("Processing Complete.")
    logging.info(f"Total initial items found: {initial_item_count}")
    logging.info(f"Items loaded for processing: {len(all_items_to_process)}")
    logging.info(f"Successfully processed and written: {processed_count}")
    logging.info(f"Items skipped or failed during loading/processing: {error_count}")
    logging.info(f"  - Skipped due to tokenization failure: {skipped_tokenization}") # This indicates an error state
    logging.info(f"  - Rasterization skipped/failed (item still processed): {skipped_rasterization}")
    logging.info(f"Output dataset saved to: {output_path}")
    logging.info(f"Images saved to: {images_dir}")
    logging.info("=" * 30)


# --- Command Line Argument Parser ---
if __name__ == "__main__":
    # Configure logging for the main process (optional, but good practice)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(
        description="Build an SFT dataset for OmniSVG from CSV, JSON, or JSONL files using multiple processes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values
    )
    parser.add_argument(
        "-i", "--input-files",
        nargs='+',
        required=True,
        help="Path(s) to the input file(s). Supported formats: .csv, .json, .jsonl. "
             "Expected keys/columns: 'svg', 'description'. 'id' is optional."
    )
    parser.add_argument(
        "-o", "--output-file",
        required=True,
        help="Path to the output JSON Lines (.jsonl) file."
    )
    parser.add_argument(
        "-n", "--num-processes",
        type=int,
        default=None, # Default calculation happens in the function
        help="Number of worker processes to use. Defaults to os.cpu_count() - 1."
    )

    args = parser.parse_args()

    # Make sure this check is here for multiprocessing safety
    # (Though Pool context manager usually handles this well)
    if os.name == 'posix':
         # Use 'spawn' instead of 'fork' for better compatibility and avoiding pickling issues
         multiprocessing.set_start_method('spawn', force=True) # Or 'forkserver' depending on needs/OS

    load_and_process_data(args.input_files, args.output_file, args.num_processes)
