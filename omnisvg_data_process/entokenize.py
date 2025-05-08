from typing import List, Optional
from .constants import VIEWBOX_WIDTH, VIEWBOX_HEIGHT, SOP_TOKEN, EOS_TOKEN, PATH_START_TOKEN, PATH_END_TOKEN, CMD_TOKENS, FILL_URL_TOKEN, FILL_CURRENTCOLOR_TOKEN
from lxml import etree
from .utils import parse_color, parse_path_d, discretize_arc_params, process_coords
from .logger import Logger

class Entokenize(Logger):
  def _format_coord_token(self, x: int, y: int) -> str:
      # New format with value inside
      return f"<｜coord_{x}_{y}｜>"

  def entokenize(self, svg_content: str) -> Optional[List[str]]:
      """Tokenizes SVG content with path delimiters, value-specific tokens, and no color quantization."""
      try:
          parser = etree.XMLParser(remove_comments=True, recover=True)
          root = etree.fromstring(svg_content.encode('utf-8'), parser=parser)
          if root is None:
                self.logger.error(f"Failed to parse SVG: Root is None after recovery.")
                return None
      except etree.XMLSyntaxError as e:
          self.logger.error(f"Invalid XML syntax: {e}")
          return None
      except Exception as e:
            self.logger.error(f"Unexpected error during XML parsing: {e}")
            return None

      viewbox = root.get('viewBox')
      expected_viewbox = f"0 0 {VIEWBOX_WIDTH} {VIEWBOX_HEIGHT}"
      expected_viewbox_float = f"0.0 0.0 {float(VIEWBOX_WIDTH)} {float(VIEWBOX_HEIGHT)}"
      if not viewbox or viewbox.strip() not in [expected_viewbox, expected_viewbox_float]:
            self.logger.warning(f"SVG does not have expected viewBox '{expected_viewbox}'. Found: '{viewbox}'. Proceeding anyway.")

      tokens: List[str] = [SOP_TOKEN]
      current_x, current_y = 0.0, 0.0
      path_count = 0

      for path_elem in root.xpath('.//*[local-name()="path"]'):
          path_count += 1
          d_string = path_elem.get('d')
          fill_color_attr = path_elem.get('fill')
          style_attr = path_elem.get('style')
          fill_color_str = fill_color_attr

          if style_attr:
              try:
                  style_rules = {k.strip(): v.strip() for k, v in (rule.split(':', 1) for rule in style_attr.split(';') if ':' in rule.strip())}
                  fill_color_str = style_rules.get('fill', fill_color_str)
              except ValueError:
                  self.logger.warning(f"Could not parse style attribute: {style_attr}")

          # Treat fill="" or fill=None or fill="none" as no fill
          if fill_color_str is None or fill_color_str.lower() == 'none' or fill_color_str == '':
                fill_color_str = None # This will result in no fill tokens being added
          # elif fill_color_str == '': # Removed: Old logic treated empty as black
          #       fill_color_str = 'black' 
          # elif fill_color_str is None: # Already handled above
          #       fill_color_str = 'black'

          if not d_string:
              self.logger.warning(f"Path element {path_count} has empty 'd' attribute. Skipping path.")
              continue

          tokens.append(PATH_START_TOKEN)

          # Add Fill tokens based on parse result
          color_result = parse_color(fill_color_str)
          if isinstance(color_result, tuple): # RGB tuple (r, g, b)
              tokens.append(CMD_TOKENS['F']) # Add F command marker first
              r, g, b = color_result
              # Append RGB values as separate tokens
              tokens.append(f"<｜color_rgb_{r}｜>")
              tokens.append(f"<｜color_rgb_{g}｜>")
              tokens.append(f"<｜color_rgb_{b}｜>")
          elif isinstance(color_result, str) and color_result.startswith('url('): # Correct check for url()
              tokens.append(CMD_TOKENS['F']) # Add F command marker
              tokens.append(FILL_URL_TOKEN)  # Add special token separately
          elif color_result == 'currentcolor':
              tokens.append(CMD_TOKENS['F']) # Add F command marker
              tokens.append(FILL_CURRENTCOLOR_TOKEN) # Add special token separately
          # No token added if color_result is None (implies fill='none')

          path_tokens_internal: List[str] = []

          try:
              raw_commands = parse_path_d(d_string)
              if not raw_commands:
                    self.logger.warning(f"Path {path_count} 'd' attribute parsed into zero commands: {d_string}")
                    tokens.append(PATH_END_TOKEN)
                    continue

              subpath_start_x, subpath_start_y = 0.0, 0.0

              for cmd_idx, (cmd, params) in enumerate(raw_commands):
                  cmd_abs = cmd.upper()
                  is_relative = cmd.islower()
                  processed_params = []
                  coords_for_update = []

                  # --- Coordinate/Parameter Processing ---
                  idx = 0
                  try:
                      if cmd_abs == 'M' or cmd_abs == 'L' or cmd_abs == 'T':
                          if len(params) < 2: raise ValueError("Need at least 2 params")
                          while idx + 1 < len(params):
                              x, y = params[idx], params[idx+1]
                              if is_relative: x += current_x; y += current_y
                              processed_params.extend(process_coords(x, y))
                              coords_for_update.append((x, y))
                              idx += 2
                      elif cmd_abs == 'H':
                          if len(params) < 1: raise ValueError("Need at least 1 param")
                          while idx < len(params):
                              x = params[idx]
                              if is_relative: x += current_x
                              processed_params.extend(process_coords(x, current_y))
                              coords_for_update.append((x, current_y))
                              idx += 1
                      elif cmd_abs == 'V':
                          if len(params) < 1: raise ValueError("Need at least 1 param")
                          while idx < len(params):
                              y = params[idx]
                              if is_relative: y += current_y
                              processed_params.extend(process_coords(current_x, y))
                              coords_for_update.append((current_x, y))
                              idx += 1
                      elif cmd_abs == 'C' or cmd_abs == 'S':
                          num_coord_pairs = 3 if cmd_abs == 'C' else 2
                          required_params = num_coord_pairs * 2
                          if len(params) < required_params: raise ValueError(f"Need at least {required_params} params")
                          while idx + required_params - 1 < len(params):
                              coords_batch_params = []
                              endpoint_x, endpoint_y = current_x, current_y
                              for i in range(num_coord_pairs):
                                  x, y = params[idx + i*2], params[idx + i*2 + 1]
                                  if is_relative: x += current_x; y += current_y
                                  coords_batch_params.extend(process_coords(x, y))
                                  if i == num_coord_pairs - 1: endpoint_x, endpoint_y = x, y
                              processed_params.extend(coords_batch_params)
                              coords_for_update.append((endpoint_x, endpoint_y))
                              idx += required_params
                      elif cmd_abs == 'Q' or cmd_abs == 'T':
                          num_coord_pairs = 2
                          required_params = num_coord_pairs * 2
                          if len(params) < required_params: raise ValueError(f"Need at least {required_params} params")
                          while idx + required_params - 1 < len(params):
                              coords_batch_params = []
                              endpoint_x, endpoint_y = current_x, current_y
                              for i in range(num_coord_pairs):
                                  x, y = params[idx + i*2], params[idx + i*2 + 1]
                                  if is_relative: x += current_x; y += current_y
                                  coords_batch_params.extend(process_coords(x, y))
                                  if i == num_coord_pairs - 1: endpoint_x, endpoint_y = x, y
                              processed_params.extend(coords_batch_params)
                              coords_for_update.append((endpoint_x, endpoint_y))
                              idx += required_params
                      elif cmd_abs == 'A':
                          required_params = 7
                          if len(params) < required_params: raise ValueError(f"Need at least {required_params} params")
                          while idx + required_params - 1 < len(params):
                              rx, ry, angle, large_arc_flag, sweep_flag, x, y = params[idx:idx+required_params]
                              final_x, final_y = x, y
                              if is_relative: final_x += current_x; final_y += current_y
                              processed_params.extend(discretize_arc_params(rx, ry, angle, large_arc_flag, sweep_flag))
                              processed_params.extend(process_coords(final_x, final_y))
                              coords_for_update.append((final_x, final_y))
                              idx += required_params
                      elif cmd_abs == 'Z':
                          coords_for_update.append((subpath_start_x, subpath_start_y))
                      else:
                          self.logger.warning(f"Path {path_count}, Cmd {cmd_idx}: Unsupported command '{cmd}'. Skipping.")
                          continue

                  except ValueError as ve:
                        self.logger.warning(f"Path {path_count}, Cmd {cmd_idx} ('{cmd}'): Invalid params: {params}. Error: {ve}. Skipping command.")
                        continue
                  except Exception as e:
                        self.logger.error(f"Path {path_count}, Cmd {cmd_idx} ('{cmd}'): Unexpected error during param processing: {e}", exc_info=True)
                        continue

                  if not coords_for_update and cmd_abs != 'Z':
                        self.logger.warning(f"Path {path_count}, Cmd {cmd_idx} ('{cmd}'): No coordinates processed.")
                        continue

                  # Update current_x, current_y
                  if coords_for_update:
                      new_current_x, new_current_y = coords_for_update[-1]
                      if cmd_abs == 'M' and cmd_idx == 0:
                          subpath_start_x, subpath_start_y = new_current_x, new_current_y
                      current_x, current_y = new_current_x, new_current_y

                  # --- Token Generation (Value-specific coordinate tokens, combined) ---
                  param_group_size = 0

                  if cmd_abs == 'M':
                      param_group_size = 2
                      base_cmd_token = CMD_TOKENS['M']
                      implicit_cmd_token = CMD_TOKENS['L']
                      if len(processed_params) >= param_group_size:
                          coord_str = self._format_coord_token(processed_params[0], processed_params[1])
                          path_tokens_internal.append(base_cmd_token) # Separate command
                          path_tokens_internal.append(coord_str)      # Separate coord
                          for i in range(param_group_size, len(processed_params), param_group_size):
                              if i + 1 < len(processed_params):
                                  coord_str_impl = self._format_coord_token(processed_params[i], processed_params[i+1])
                                  path_tokens_internal.append(implicit_cmd_token) # Separate command
                                  path_tokens_internal.append(coord_str_impl)     # Separate coord

                  elif cmd_abs == 'L' or cmd_abs == 'H' or cmd_abs == 'V':
                      param_group_size = 2
                      base_cmd_token = CMD_TOKENS['L']
                      if len(processed_params) >= param_group_size:
                          for i in range(0, len(processed_params), param_group_size):
                              if i + 1 < len(processed_params):
                                  coord_str = self._format_coord_token(processed_params[i], processed_params[i+1])
                                  path_tokens_internal.append(base_cmd_token) # Separate command
                                  path_tokens_internal.append(coord_str)      # Separate coord

                  elif cmd_abs == 'C' or cmd_abs == 'S':
                      param_group_size = 6 # 3 coord pairs
                      base_cmd_token = CMD_TOKENS['C']
                      if cmd_abs == 'S': self.logger.warning(f"Path {path_count}, Cmd {cmd_idx}: S command is approximated as C.")
                      if len(processed_params) >= param_group_size:
                          for i in range(0, len(processed_params), param_group_size):
                              if i + 5 < len(processed_params):
                                  coord1_str = self._format_coord_token(processed_params[i], processed_params[i+1])
                                  coord2_str = self._format_coord_token(processed_params[i+2], processed_params[i+3])
                                  coord3_str = self._format_coord_token(processed_params[i+4], processed_params[i+5])
                                  path_tokens_internal.append(base_cmd_token) # Separate command
                                  path_tokens_internal.append(coord1_str)     # Separate coord 1
                                  path_tokens_internal.append(coord2_str)     # Separate coord 2
                                  path_tokens_internal.append(coord3_str)     # Separate coord 3

                  elif cmd_abs == 'Q' or cmd_abs == 'T':
                        param_group_size = 4 # 2 coord pairs, output as C
                        base_cmd_token = CMD_TOKENS['C'] # Approx Q/T as C
                        if cmd_abs == 'T': self.logger.warning(f"Path {path_count}, Cmd {cmd_idx}: T command is approximated as C.")
                        if len(processed_params) >= param_group_size:
                            for i in range(0, len(processed_params), param_group_size):
                                if i + 3 < len(processed_params):
                                    # Treat Q/T as C for tokenization output
                                    coord1_str = self._format_coord_token(processed_params[i], processed_params[i+1])
                                    # For Q: control point is coord1. For T: control point is implied (reflection of previous). We use coord1 as approximation for both.
                                    # Endpoint is coord2
                                    coord3_str = self._format_coord_token(processed_params[i+2], processed_params[i+3])
                                    path_tokens_internal.append(base_cmd_token) # Separate command (C)
                                    path_tokens_internal.append(coord1_str)     # Separate coord 1 (control)
                                    path_tokens_internal.append(coord1_str)     # Separate coord 2 (control approx)
                                    path_tokens_internal.append(coord3_str)     # Separate coord 3 (end)

                  elif cmd_abs == 'A':
                      num_arc_params = 5
                      num_coord_params = 2
                      param_group_size = num_arc_params + num_coord_params
                      base_cmd_token = CMD_TOKENS['A']
                      if len(processed_params) >= param_group_size:
                          for i in range(0, len(processed_params), param_group_size):
                              if i + param_group_size - 1 < len(processed_params):
                                  arc_params_values = processed_params[i : i + num_arc_params]
                                  coord_params = processed_params[i + num_arc_params : i + param_group_size]
                                  # Construct value-specific tokens for arc params
                                  rx_d, ry_d, angle_d, large_f, sweep_f = arc_params_values
                                  arc_rx_token = f"<｜arc_rx_{rx_d}｜>"
                                  arc_ry_token = f"<｜arc_ry_{ry_d}｜>"
                                  arc_angle_token = f"<｜arc_angle_{angle_d}｜>"
                                  arc_large_token = f"<｜arc_large_{large_f}｜>"
                                  arc_sweep_token = f"<｜arc_sweep_{sweep_f}｜>"
                                  # Construct value-specific token for end coordinate
                                  coord_token_str = self._format_coord_token(*coord_params)
                                  # Append command and all parameters as separate tokens
                                  path_tokens_internal.append(base_cmd_token)
                                  path_tokens_internal.append(arc_rx_token)
                                  path_tokens_internal.append(arc_ry_token)
                                  path_tokens_internal.append(arc_angle_token)
                                  path_tokens_internal.append(arc_large_token)
                                  path_tokens_internal.append(arc_sweep_token)
                                  path_tokens_internal.append(coord_token_str)
                              else:
                                    self.logger.warning(f"Path {path_count}, Cmd {cmd_idx} ('{cmd}'): Insufficient params for A group at index {i}")

                  elif cmd_abs == 'Z':
                      path_tokens_internal.append(CMD_TOKENS['Z'])
                      current_x, current_y = subpath_start_x, subpath_start_y

                  if param_group_size > 0 and len(processed_params) % param_group_size != 0:
                        self.logger.warning(f"Path {path_count}, Cmd {cmd_idx} ('{cmd}'): Mismatch between processed params ({len(processed_params)}) and expected group size ({param_group_size}).")

          except Exception as e:
                self.logger.error(f"Path {path_count}: Unexpected error processing commands in d='{d_string}': {e}", exc_info=True)
                path_tokens_internal = []

          tokens.extend(path_tokens_internal)
          tokens.append(PATH_END_TOKEN)

      tokens.append(EOS_TOKEN)
      if len(tokens) <= 2:
          self.logger.warning(f"SVG resulted in empty token sequence (only SOP/EOS), returning minimal list.")
          # Return [SOP, EOS] instead of None for empty but valid SVG.
          return [SOP_TOKEN, EOS_TOKEN]

      return tokens
