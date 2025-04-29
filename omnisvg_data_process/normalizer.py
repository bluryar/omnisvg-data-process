from typing import Optional
from lxml import etree
from .utils import parse_path_d, discretize_arc_params, process_coords, parse_color
from .logger import Logger
from .constants import VIEWBOX_WIDTH, VIEWBOX_HEIGHT # Ensure constants are imported

class Normalizer(Logger):
  def normalize_svg(self, svg_content: str, file_path_for_logging: str = "(unknown file)") -> Optional[str]:
    """Normalizes and simplifies SVG content to a standard format."""
    try:
        parser = etree.XMLParser(remove_comments=True, recover=True)
        original_root = etree.fromstring(svg_content.encode('utf-8'), parser=parser)
        if original_root is None:
            self.logger.error(f"[{file_path_for_logging}] Normalize SVG: Failed to parse - Root is None.")
            return None
    except Exception as e:
        self.logger.error(f"[{file_path_for_logging}] Normalize SVG: Failed to parse - {e}")
        return None

    # Normalize colors first (before path processing which might add/remove elements)
    # Process relevant elements and their fill/stroke attributes
    elements_to_process = original_root.xpath(".//*[local-name()='path' or local-name()='rect' or local-name()='circle' or local-name()='ellipse' or local-name()='line' or local-name()='polygon' or local-name()='polyline']")
    for elem in elements_to_process:
        for attr_name in ["fill", "stroke"]:
            # 1. Get values from attribute and style
            val_from_attr = elem.get(attr_name)
            val_from_style = None
            style_attr = elem.get('style')
            style_dict = {}
            if style_attr:
                try:
                    # Parse style into a dictionary for easier lookup
                    style_dict = {k.strip().lower(): v.strip() for k, v in (rule.split(':', 1) for rule in style_attr.split(';') if ':' in rule.strip())}
                    val_from_style = style_dict.get(attr_name)
                except ValueError:
                    self.logger.warning(f"[{file_path_for_logging}] Normalize Style: Malformed style '{style_attr}'. Ignoring for {attr_name}.")
                    style_attr = None # Treat as if no style

            # 2. Determine effective original value (Style > Attribute)
            effective_original_color = None
            if val_from_style: # Prioritize style
                effective_original_color = val_from_style
            elif val_from_attr: # Fallback to attribute
                effective_original_color = val_from_attr

            # 3. Parse the effective value and normalize it
            normalized_color_val = "none" # Default if no color specified or invalid
            if effective_original_color:
                parsed = parse_color(effective_original_color)
                if isinstance(parsed, tuple): # RGB
                    r, g, b = parsed
                    try:
                        normalized_color_val = "#{:02x}{:02x}{:02x}".format(r, g, b).lower()
                    except ValueError:
                        self.logger.warning(f"[{file_path_for_logging}] Normalize Color: Invalid RGB {parsed} from effective '{effective_original_color}'.")
                        # Keep default "none"
                elif isinstance(parsed, str) and parsed.startswith('url('):
                    normalized_color_val = "url(#default_grad)" # Standardize URL
                elif parsed == 'currentcolor':
                    normalized_color_val = "currentColor" # Correct case!
                elif parsed is None: # Explicit 'none' found
                    normalized_color_val = "none"
                # else: Unhandled parse result, keep default "none"
            # 4. Set the DIRECT attribute on the element to the final normalized value
            # This ensures the attribute reflects the style override if applicable
            elem.set(attr_name, normalized_color_val)

            # 5. Clean up the style attribute by removing the handled key
            if style_attr and attr_name in style_dict:
                del style_dict[attr_name] # Remove the key we processed
                # Reconstruct style attribute without the handled key
                remaining_style = ";".join([f"{k}:{v}" for k, v in style_dict.items() if v.strip()])
                if remaining_style:
                    elem.set('style', remaining_style)
                elif 'style' in elem.attrib: # Remove empty style attribute
                    del elem.attrib['style']

    current_x, current_y = 0.0, 0.0
    path_count = 0

    for path_elem in original_root.xpath('.//*[local-name()="path"]'):
        path_count += 1
        d_string = path_elem.get('d')
        if not d_string:
            continue

        normalized_d_parts = []
        try:
            raw_commands = parse_path_d(d_string)
            if not raw_commands:
                continue

            abs_subpath_start_x, abs_subpath_start_y = 0.0, 0.0 # Track absolute for Z reset

            for cmd_idx, (cmd, params) in enumerate(raw_commands):
                cmd_abs = cmd.upper()
                is_relative = cmd.islower()
                normalized_params_str = []
                coords_for_update = []

                idx = 0
                try:
                    # Coordinate/Parameter Processing & Normalization
                    if cmd_abs == 'M' or cmd_abs == 'L' or cmd_abs == 'T':
                        if len(params) < 2: raise ValueError("Need at least 2 params")
                        while idx + 1 < len(params):
                            x, y = params[idx], params[idx+1]
                            abs_x, abs_y = (x + current_x, y + current_y) if is_relative else (x, y)
                            norm_x, norm_y = process_coords(abs_x, abs_y)
                            normalized_params_str.extend([str(norm_x), str(norm_y)])
                            coords_for_update.append((abs_x, abs_y))
                            idx += 2
                    elif cmd_abs == 'H':
                        if len(params) < 1: raise ValueError("Need at least 1 param")
                        while idx < len(params):
                            x = params[idx]
                            abs_x = x + current_x if is_relative else x
                            norm_x, _ = process_coords(abs_x, current_y) # Only norm_x matters
                            normalized_params_str.append(str(norm_x))
                            coords_for_update.append((abs_x, current_y))
                            idx += 1
                    elif cmd_abs == 'V':
                          if len(params) < 1: raise ValueError("Need at least 1 param")
                          while idx < len(params):
                              y = params[idx]
                              abs_y = y + current_y if is_relative else y
                              _, norm_y = process_coords(current_x, abs_y) # Only norm_y matters
                              normalized_params_str.append(str(norm_y))
                              coords_for_update.append((current_x, abs_y))
                              idx += 1
                    elif cmd_abs == 'C' or cmd_abs == 'S':
                          num_coord_pairs = 3 if cmd_abs == 'C' else 2
                          required_params = num_coord_pairs * 2
                          if len(params) < required_params: raise ValueError(f"Need {required_params} params")
                          while idx + required_params - 1 < len(params):
                              endpoint_x, endpoint_y = current_x, current_y
                              for i in range(num_coord_pairs):
                                  x, y = params[idx + i*2], params[idx + i*2 + 1]
                                  abs_x, abs_y = (x + current_x, y + current_y) if is_relative else (x, y)
                                  norm_x, norm_y = process_coords(abs_x, abs_y)
                                  normalized_params_str.extend([str(norm_x), str(norm_y)])
                                  if i == num_coord_pairs - 1: endpoint_x, endpoint_y = abs_x, abs_y
                              coords_for_update.append((endpoint_x, endpoint_y))
                              idx += required_params
                    elif cmd_abs == 'Q' or cmd_abs == 'T':
                          num_coord_pairs = 2
                          required_params = num_coord_pairs * 2
                          if len(params) < required_params: raise ValueError(f"Need {required_params} params")
                          while idx + required_params - 1 < len(params):
                              endpoint_x, endpoint_y = current_x, current_y
                              for i in range(num_coord_pairs):
                                  x, y = params[idx + i*2], params[idx + i*2 + 1]
                                  abs_x, abs_y = (x + current_x, y + current_y) if is_relative else (x, y)
                                  norm_x, norm_y = process_coords(abs_x, abs_y)
                                  normalized_params_str.extend([str(norm_x), str(norm_y)])
                                  if i == num_coord_pairs - 1: endpoint_x, endpoint_y = abs_x, abs_y
                              coords_for_update.append((endpoint_x, endpoint_y))
                              idx += required_params
                    elif cmd_abs == 'A':
                        required_params = 7
                        if len(params) < required_params: raise ValueError(f"Need {required_params} params")
                        while idx + required_params - 1 < len(params):
                            rx, ry, angle, large_arc_flag, sweep_flag, x, y = params[idx:idx+required_params]
                            final_x, final_y = x, y
                            if is_relative: final_x += current_x; final_y += current_y
                            rx_d, ry_d, angle_d, large_f, sweep_f = discretize_arc_params(rx, ry, angle, large_arc_flag, sweep_flag)
                            norm_x, norm_y = process_coords(final_x, final_y)
                            normalized_params_str.extend([str(rx_d), str(ry_d), str(angle_d), str(large_f), str(sweep_f), str(norm_x), str(norm_y)])
                            coords_for_update.append((final_x, final_y))
                            idx += required_params
                    elif cmd_abs == 'Z':
                        coords_for_update.append((abs_subpath_start_x, abs_subpath_start_y))
                    else:
                        self.logger.warning(f"[{file_path_for_logging}] Normalize SVG: Unsupported command '{cmd}'. Keeping original params.")
                        normalized_params_str = [str(p) for p in params]

                except ValueError as ve:
                      self.logger.warning(f"[{file_path_for_logging}] Normalize SVG Path {path_count}, Cmd {cmd_idx} ('{cmd}'): Invalid params: {params}. Error: {ve}. Skipping command.")
                      continue
                except Exception as e:
                      self.logger.error(f"[{file_path_for_logging}] Normalize SVG Path {path_count}, Cmd {cmd_idx} ('{cmd}'): Unexpected error: {e}", exc_info=True)
                      continue

                # Update current position using absolute float values
                if coords_for_update:
                    new_current_x, new_current_y = coords_for_update[-1]
                    if cmd_abs == 'M' and cmd_idx == 0:
                          abs_subpath_start_x, abs_subpath_start_y = new_current_x, new_current_y
                    current_x, current_y = new_current_x, new_current_y
                    if cmd_abs == 'Z':
                        current_x, current_y = abs_subpath_start_x, abs_subpath_start_y

                # Append the original command letter and normalized parameters
                if cmd_abs == 'Z':
                    # Use uppercase Z for consistency, although case doesn't matter for Z
                    normalized_d_parts.append(cmd.upper())
                elif normalized_params_str:
                      # Always use the uppercase command letter for normalized output
                      normalized_d_parts.append(cmd.upper() + " " + " ".join(normalized_params_str))

        except Exception as e:
              self.logger.error(f"[{file_path_for_logging}] Normalize SVG Path {path_count}: Error processing d='{d_string}': {e}", exc_info=True)
              continue

        if normalized_d_parts:
            path_elem.set("d", " ".join(normalized_d_parts))
        else:
            if 'd' in path_elem.attrib:
                  del path_elem.attrib['d']
                  self.logger.warning(f"[{file_path_for_logging}] Normalize SVG Path {path_count}: Removed 'd' attribute due to normalization errors.")

    # Create the new simplified SVG structure
    new_root = etree.Element("svg", attrib={
        "xmlns": "http://www.w3.org/2000/svg",
        "viewBox": f"0 0 {VIEWBOX_WIDTH} {VIEWBOX_HEIGHT}",
        "width": f"{VIEWBOX_WIDTH}px",
        "height": f"{VIEWBOX_HEIGHT}px"
    })

    # Iterate over the *modified* original paths again to extract final state
    final_path_elements = original_root.xpath('.//*[local-name()="path"]')
    for path_elem in final_path_elements:
        final_d = path_elem.get('d')
        if not final_d: # Skip paths with no valid 'd' attribute after normalization
            continue
            
        # Get the fill attribute which was correctly set in Step 1 (reflecting style priority)
        # Default to 'none' only if the attribute is actually missing after step 1
        # Determine the final fill color, considering style and attributes
        final_fill = "none" # Initialize default
        fill_from_style = None
        fill_from_attr = path_elem.get('fill') # Get potentially normalized attribute

        path_style = path_elem.get('style')
        if path_style:
            try:
                style_dict = dict(rule.split(':', 1) for rule in path_style.split(';') if ':' in rule.strip())
                fill_from_style = style_dict.get('fill') # This should be normalized
            except ValueError:
                pass # Ignore malformed style

        # Apply priority: Style > Attribute
        if fill_from_style and fill_from_style.lower() != 'none' and fill_from_style.strip() != '':
            # Use valid style value (can be #rrggbb, url, currentColor)
            final_fill = fill_from_style
        elif fill_from_attr and fill_from_attr.lower() != 'none' and fill_from_attr.strip() != '':
            # Use valid attribute value if style was not valid or not present
            final_fill = fill_from_attr
        # else: final_fill remains "none"

        # Create the new simplified path element
        new_path = etree.Element("path")
        new_path.set("d", final_d)
        # Always set fill, ensuring it captures currentColor correctly or defaults to none
        new_path.set("fill", final_fill) 
        
        # Append the simplified path to the new root
        new_root.append(new_path)

    # Serialize the new simplified tree
    try:
        # Set pretty_print=False to avoid extra whitespace and trailing newline
        simplified_svg_string = etree.tostring(new_root, encoding='unicode', pretty_print=False)
        return simplified_svg_string
    except Exception as e:
        self.logger.error(f"[{file_path_for_logging}] Normalize SVG: Error serializing simplified tree: {e}")
        return None
