import typer
from pathlib import Path
from loguru import logger
import json
from tqdm import tqdm

from .tokenizer import SVGTokenizer

app = typer.Typer()

@app.command()
def process_svgs(
    input_dir: Path = typer.Argument(..., help="Directory containing SVG files.", exists=True, file_okay=False, readable=True),
    output_file: Path = typer.Argument(..., help="Path to the output JSON Lines file.", dir_okay=False, writable=True),
    max_files: int = typer.Option(None, "--max-files", "-n", help="Maximum number of files to process."),
):
    """Processes SVG files in a directory and saves tokenized output to a JSON Lines file."""
    logger.info(f"Starting SVG processing.")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output file: {output_file}")

    tokenizer = SVGTokenizer()
    svg_files = list(input_dir.rglob("*.svg"))

    if not svg_files:
        logger.error(f"No SVG files found in {input_dir}")
        raise typer.Exit(code=1)

    logger.info(f"Found {len(svg_files)} SVG files.")

    if max_files is not None:
        logger.info(f"Processing a maximum of {max_files} files.")
        svg_files = svg_files[:max_files]

    processed_count = 0
    error_count = 0

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for svg_path in tqdm(svg_files, desc="Processing SVGs"):
            try:
                with open(svg_path, 'r', encoding='utf-8') as f:
                    svg_content = f.read()

                tokens = tokenizer.entokenize(svg_content)

                if tokens:
                    output_data = {
                        "file_path": str(svg_path.relative_to(input_dir)),
                        "tokens": tokens
                    }
                    outfile.write(json.dumps(output_data) + '\n')
                    processed_count += 1
                else:
                    logger.warning(f"Skipping file due to tokenization error or empty result: {svg_path}")
                    error_count += 1

            except Exception as e:
                logger.error(f"Failed to process file {svg_path}: {e}")
                error_count += 1

    logger.info(f"Processing finished.")
    logger.info(f"Successfully processed: {processed_count} files.")
    logger.info(f"Skipped/Errored: {error_count} files.")
    logger.info(f"Output saved to: {output_file}")

if __name__ == "__main__":
    app() 
