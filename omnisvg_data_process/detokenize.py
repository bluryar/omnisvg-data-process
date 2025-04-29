from typing import List, Tuple
import re
from .constants import VIEWBOX_WIDTH, VIEWBOX_HEIGHT, SOP_TOKEN, EOS_TOKEN, PATH_START_TOKEN, PATH_END_TOKEN, CMD_TOKENS, FILL_URL_TOKEN, FILL_CURRENTCOLOR_TOKEN
from lxml import etree
from .logger import Logger

class Detokenize(Logger):
  def _parse_coord_token(self, token: str) -> Tuple[str, str] | None:
      match = re.fullmatch(r"<｜coord_(\d+)_(\d+)｜>", token)
      return (match.group(1), match.group(2)) if match else None

  def _parse_color_rgb_token(self, token: str) -> int | None:
      match = re.fullmatch(r"<｜color_rgb_(\d+)｜>", token)
      return int(match.group(1)) if match and 0 <= int(match.group(1)) <= 255 else None

  def _parse_arc_param_token(self, token: str, param_type: str) -> str | None:
      # param_type should be one of: rx, ry, angle, large, sweep
      match = re.fullmatch(rf"<｜arc_{param_type}_(\d+)｜>", token)
      return match.group(1) if match else None

  def detokenize(self, tokens: List[str]) -> str:
      """Converts a list of sequential tokens back into an SVG string content."""
      svg_paths = []
      current_path_d_parts = []
      current_fill = None
      in_path = False
      i = 0

      while i < len(tokens):
          token = tokens[i]

          if token == SOP_TOKEN:
              i += 1
              continue
          if token == EOS_TOKEN:
              break # End of sequence

          if token == PATH_START_TOKEN:
              if in_path:
                    self.logger.warning("Found PATH_START while already in a path. Finishing previous path.")
                    if current_path_d_parts:
                      path_d = " ".join(current_path_d_parts)
                      path_attrs = {"d": path_d}
                      if current_fill:
                          path_attrs["fill"] = current_fill
                      svg_paths.append(etree.Element("path", attrib=path_attrs))
              current_path_d_parts = []
              current_fill = None
              in_path = True
              i += 1
              continue

          if token == PATH_END_TOKEN:
              if not in_path:
                  self.logger.warning("Found PATH_END without corresponding PATH_START.")
              else:
                    path_attrs = {}
                    if current_path_d_parts:
                        path_attrs["d"] = " ".join(current_path_d_parts)
                    
                    # Set fill attribute based on what was parsed
                    if current_fill: 
                        path_attrs["fill"] = current_fill
                    elif current_path_d_parts: # Only add fill=none if there are path commands
                        path_attrs["fill"] = "none" 
                        
                    if path_attrs: # Only append if we have d or fill
                        svg_paths.append(etree.Element("path", attrib=path_attrs))
                    else:
                        self.logger.warning("Path ended with no fill and no commands. Not adding empty path element.")

              in_path = False
              current_path_d_parts = []
              current_fill = None # Reset fill for next path
              i += 1
              continue

          if not in_path:
              self.logger.warning(f"Encountered token '{token}' outside of expected structure (SOP/PATH_START). Skipping.")
              i += 1 # Skip unexpected token
              continue

          # --- Process tokens within a path --- 
          
          # Check for Fill command marker
          if token == CMD_TOKENS['F']:
                i += 1 # Move past F marker
                if i < len(tokens):
                    fill_token = tokens[i]
                    if fill_token == FILL_URL_TOKEN:
                        current_fill = "url(#default_grad)" # Use a placeholder ID
                        i += 1
                    elif fill_token == FILL_CURRENTCOLOR_TOKEN:
                        current_fill = "currentColor"
                        i += 1
                    else:
                        # Expecting 3 color_rgb tokens
                        if i + 2 < len(tokens):
                            r_val = self._parse_color_rgb_token(tokens[i])
                            g_val = self._parse_color_rgb_token(tokens[i+1])
                            b_val = self._parse_color_rgb_token(tokens[i+2])
                            if r_val is not None and g_val is not None and b_val is not None:
                                try:
                                    current_fill = "#{:02x}{:02x}{:02x}".format(r_val, g_val, b_val)
                                except ValueError:
                                    self.logger.warning(f"Invalid RGB values ({r_val},{g_val},{b_val}) for hex conversion. Using black.")
                                    current_fill = "#000000"
                                i += 3 # Consume 3 color tokens
                            else:
                                self.logger.warning(f"Expected 3 valid color_rgb tokens after SVG_F, got: {tokens[i:i+3]}. Fill not set.")
                                # Don't consume the invalid color tokens, let outer loop handle them
                        else:
                            self.logger.warning(f"Found SVG_F marker but not enough subsequent tokens for RGB color. Fill not set.")
                            # Don't consume tokens
                else:
                    self.logger.warning(f"Found SVG_F marker at the end of the token list. Fill not set.")
                continue # Continue loop after processing fill attempt

          # Check for Path Commands
          elif token == CMD_TOKENS['M']:
                i += 1 # Move past M marker
                if i < len(tokens):
                    coords = self._parse_coord_token(tokens[i])
                    if coords:
                        current_path_d_parts.append(f"M {coords[0]} {coords[1]}")
                        i += 1 # Consume coord token
                    else:
                        self.logger.warning(f"Expected coord token after M, got '{tokens[i]}'.")
                else:
                    self.logger.warning("Expected coord token after M, but reached end of list.")
                continue

          elif token == CMD_TOKENS['L']:
                i += 1 # Move past L marker
                if i < len(tokens):
                    coords = self._parse_coord_token(tokens[i])
                    if coords:
                        current_path_d_parts.append(f"L {coords[0]} {coords[1]}")
                        i += 1 # Consume coord token
                    else:
                        self.logger.warning(f"Expected coord token after L, got '{tokens[i]}'.")
                else:
                    self.logger.warning("Expected coord token after L, but reached end of list.")
                continue

          elif token == CMD_TOKENS['C']:
                i += 1 # Move past C marker
                if i + 2 < len(tokens):
                    c1 = self._parse_coord_token(tokens[i])
                    c2 = self._parse_coord_token(tokens[i+1])
                    c3 = self._parse_coord_token(tokens[i+2])
                    if c1 and c2 and c3:
                        current_path_d_parts.append(f"C {c1[0]} {c1[1]} {c2[0]} {c2[1]} {c3[0]} {c3[1]}")
                        i += 3 # Consume 3 coord tokens
                    else:
                        self.logger.warning(f"Expected 3 coord tokens after C, got invalid sequence: {tokens[i:i+3]}. Skipping command.")
                        # Don't consume invalid tokens
                else:
                    self.logger.warning("Expected 3 coord tokens after C, but reached end of list.")
                continue

          elif token == CMD_TOKENS['A']:
                i += 1 # Move past A marker
                # Expect 5 arc param tokens + 1 coord token
                if i + 5 < len(tokens):
                    rx = self._parse_arc_param_token(tokens[i], 'rx')
                    ry = self._parse_arc_param_token(tokens[i+1], 'ry')
                    angle = self._parse_arc_param_token(tokens[i+2], 'angle')
                    large = self._parse_arc_param_token(tokens[i+3], 'large')
                    sweep = self._parse_arc_param_token(tokens[i+4], 'sweep')
                    coords = self._parse_coord_token(tokens[i+5])
                    if all([rx, ry, angle, large, sweep, coords]):
                        current_path_d_parts.append(f"A {rx} {ry} {angle} {large} {sweep} {coords[0]} {coords[1]}")
                        i += 6 # Consume 6 tokens
                    else:
                        self.logger.warning(f"Expected valid arc parameters and coord after A, got invalid sequence: {tokens[i:i+6]}. Skipping command.")
                        # Don't consume invalid tokens
                else:
                    self.logger.warning("Expected 6 tokens (arc params + coord) after A, but reached end of list.")
                continue

          elif token == CMD_TOKENS['Z']:
                current_path_d_parts.append("Z")
                i += 1 # Consume Z token
                continue

          else:
                # Handle unrecognized token within a path
                self.logger.warning(f"Unrecognized token inside path: '{token}'. Skipping.")
                i += 1 # Skip unrecognized token
                continue

      # Construct the final SVG
      svg_root = etree.Element("svg", attrib={
          "xmlns": "http://www.w3.org/2000/svg",
          "viewBox": f"0 0 {VIEWBOX_WIDTH} {VIEWBOX_HEIGHT}",
          # Add width/height attributes to match the normalizer output format
          "width": f"{VIEWBOX_WIDTH}px",
          "height": f"{VIEWBOX_HEIGHT}px"
      })
      for path in svg_paths:
          svg_root.append(path)

      # Use unicode encoding, no XML declaration, pretty_print=False
      try:
          svg_string = etree.tostring(svg_root, encoding='unicode', pretty_print=False)
          return svg_string
      except Exception as e:
          self.logger.error(f"Failed to serialize final SVG element: {e}", exc_info=True)
          # Return a minimal valid SVG as fallback
          return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEWBOX_WIDTH} {VIEWBOX_HEIGHT}"></svg>'
