import re
from typing import Optional
from .constants import VIEWBOX_WIDTH, VIEWBOX_HEIGHT, SOP_TOKEN, EOS_TOKEN, PATH_START_TOKEN, PATH_END_TOKEN, CMD_TOKENS, FILL_URL_TOKEN, FILL_CURRENTCOLOR_TOKEN
from .tokenize import Tokenize
from .detokenize import Detokenize
from .normalizer import Normalizer
from .logger import Logger

class SVGTokenizer(Tokenize, Detokenize, Normalizer, Logger):
    def __init__(self):
        super().__init__(label='SVGTokenizer') # Initialize parent classes, including Logger
        self.logger.info("Initializing SVGTokenizer...")
        self.vocabulary = self._build_vocabulary()
        self.logger.info(f"Vocabulary built with {len(self.vocabulary)} unique tokens.")
        # Add more initialization if needed by parent classes via super()
        # super().__init__() # Example if parents had __init__

    def _build_vocabulary(self) -> set:
        """Generates the set of all possible token strings."""
        vocab = set()

        # 1. Fixed Tokens
        vocab.add(SOP_TOKEN)
        vocab.add(EOS_TOKEN)
        vocab.add(PATH_START_TOKEN)
        vocab.add(PATH_END_TOKEN)
        vocab.add(CMD_TOKENS['Z']) # Z has no params
        vocab.add(CMD_TOKENS['F']) # F marker
        vocab.add(FILL_URL_TOKEN)
        vocab.add(FILL_CURRENTCOLOR_TOKEN)

        # 2. Value-specific Coordinate Tokens (0 to VIEWBOX_DIM inclusive)
        for x in range(VIEWBOX_WIDTH + 1): # Include 200
            for y in range(VIEWBOX_HEIGHT + 1): # Include 200
                vocab.add(f"<｜coord_{x}_{y}｜>")

        # 3. Value-specific Color Tokens (0-255)
        for val in range(256):
             vocab.add(f"<｜color_rgb_{val}｜>")

        # 4. Value-specific Arc Parameter Tokens (for use within combined A token)
        # RX / RY (0 to Viewbox Dim)
        for r in range(max(VIEWBOX_WIDTH, VIEWBOX_HEIGHT) + 1):
            if r <= VIEWBOX_WIDTH: vocab.add(f"<｜arc_rx_{r}｜>")
            if r <= VIEWBOX_HEIGHT: vocab.add(f"<｜arc_ry_{r}｜>")
        # Angle (0-359)
        for angle in range(360):
             vocab.add(f"<｜arc_angle_{angle}｜>")
        # Flags (0 or 1)
        vocab.add("<｜arc_large_0｜>")
        vocab.add("<｜arc_large_1｜>")
        vocab.add("<｜arc_sweep_0｜>")
        vocab.add("<｜arc_sweep_1｜>")

        # 5. Combined Command Tokens (M, L, C, A)
        # Note: Generating ALL combinations is huge. We store the *patterns* conceptually.
        # We already added the command markers (e.g., <｜SVG_M｜>) via CMD_TOKENS.
        # The individual value tokens (<｜coord_x_y｜>, etc.) are also added.
        # For validation or model building, one might use the fixed tokens + generate value tokens on the fly,
        # or just use the fixed tokens + the set of generated value tokens above.
        # Let's ensure the command markers themselves are in the vocab explicitly.
        for cmd_token in CMD_TOKENS.values():
             if cmd_token != CMD_TOKENS['Z'] and cmd_token != CMD_TOKENS['F']: # Z and F handled
                 vocab.add(cmd_token)

        return vocab

    def svg_to_tokens(self, svg_content: str) -> Optional[str]:
        """Converts SVG content string to a single string of concatenated tokens."""
        # Calls the tokenize method inherited from the Tokenize class
        tokens_list = self.tokenize(svg_content)
        if tokens_list:
            return "".join(tokens_list)
        else:
            return None # Return None if tokenization failed

    def tokens_to_svg(self, tokens_str: str) -> Optional[str]:
        """Converts a concatenated token string back to an SVG content string."""
        # Use regex to split the concatenated string back into individual tokens
        tokens_list = re.findall(r'(<｜[^｜]+｜>)', tokens_str)

        if not tokens_list:
            if tokens_str is None or tokens_str.strip() == "":
                self.logger.info(f"Tokens to SVG: Input token string is empty.")
                # Ensure the returned empty SVG string includes width and height
                return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEWBOX_WIDTH} {VIEWBOX_HEIGHT}" width="{VIEWBOX_WIDTH}px" height="{VIEWBOX_HEIGHT}px"></svg>'
            else:
                self.logger.error(f"Tokens to SVG: Could not parse any tokens from string: '{tokens_str[:100]}...'")
                return None

        # Basic validation
        if tokens_list[0] != SOP_TOKEN:
             self.logger.warning(f"Tokens to SVG: Sequence does not start with {SOP_TOKEN}.")
        if tokens_list[-1] != EOS_TOKEN:
             self.logger.warning(f"Tokens to SVG: Sequence does not end with {EOS_TOKEN}.")

        try:
            # Calls the detokenize method inherited from the Detokenize class
            return self.detokenize(tokens_list)
        except Exception as e:
            self.logger.error(f"Tokens to SVG: Error during detokenization: {e}", exc_info=True)
            return None
