from loguru import logger
from typing import List, Tuple, Union, Optional
import re
from .constants import NAMEED_COLOR as NAMED_COLORS_RGB, VIEWBOX_WIDTH, VIEWBOX_HEIGHT, COORD_PRECISION

def parse_path_d(d_string: str) -> List[Tuple[str, List[float]]]:
    """Parses the 'd' attribute of an SVG path into commands and parameters."""
    pattern = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])([^MmLlHhVvCcSsQqTtAaZz]*)")
    commands = []
    for match in pattern.finditer(d_string):
        cmd = match.group(1)
        params_str = match.group(2).strip()
        try:
            params = [float(p) for p in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?|[-+]?\d+", params_str)]
        except ValueError:
            logger.warning(f"Could not parse parameters for command '{cmd}': {params_str}")
            params = []
        commands.append((cmd, params))
    return commands

def hex_to_rgb(hex_color: str) -> Optional[Tuple[int, int, int]]:
    """Converts a hex color string (#rgb or #rrggbb) to an RGB tuple (0-255)."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        r = int(hex_color[0]*2, 16)
        g = int(hex_color[1]*2, 16)
        b = int(hex_color[2]*2, 16)
        return (r, g, b)
    elif len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)
    else:
        logger.warning(f"Unsupported hex color format: #{hex_color}")
        return None

# Regex for hex colors
hex_pattern = re.compile(r"^#([a-fA-F0-9]{3}|[a-fA-F0-9]{6})$")

# Regex for rgb/rgba formats - Allow floats in percentages
rgb_pattern = re.compile(r"rgb\(\s*((?:\d*\.\d+|\d+)\%?)\s*,\s*((?:\d*\.\d+|\d+)\%?)\s*,\s*((?:\d*\.\d+|\d+)\%?)\s*\)", re.IGNORECASE)
rgba_pattern = re.compile(r"rgba\(\s*((?:\d*\.\d+|\d+)\%?)\s*,\s*((?:\d*\.\d+|\d+)\%?)\s*,\s*((?:\d*\.\d+|\d+)\%?)\s*,\s*([\d.]+)\s*\)", re.IGNORECASE)

def _parse_rgb_value(value_str: str) -> Optional[int]:
    """Parses a single RGB value (number or percentage) string to int 0-255."""
    value_str = value_str.strip()
    if value_str.endswith('%'):
        try:
            percent = float(value_str[:-1])
            # Clamp percentage before scaling
            val = max(0.0, min(100.0, percent)) * 2.55
            # Round half up and clamp to 0-255
            return max(0, min(255, int(val + 0.5)))
        except ValueError:
            return None
    else:
        try:
            num = float(value_str)
            # Round half up and clamp to 0-255
            return max(0, min(255, int(num + 0.5)))
        except ValueError:
            return None

def parse_color(color_str: Optional[str]) -> Union[Tuple[int, int, int], str, None]:
    """
    Parses a CSS color string to raw integer RGB tuple (r, g, b) [0-255],
    or a special string ('url', 'currentcolor'), or None if invalid/none.
    Uses NAMED_COLORS_RGB dictionary from constants and manual parsing for hex/rgb/rgba.
    """
    if not color_str or color_str.strip() == '':
        return None

    color_str_processed = color_str.strip()
    color_str_lower = color_str_processed.lower()

    # 1. Handle 'none'
    if color_str_lower == 'none':
        return None

    # 2. Check for url() - Return the full string
    if color_str_lower.startswith('url('):
        return color_str_processed # Return the original (stripped) url string

    # 3. Check for currentColor or inherit
    if color_str_lower in ['currentcolor', 'inherit']:
         return 'currentcolor'

    # 4. Explicitly check for hex format (#rgb or #rrggbb)
    hex_match = hex_pattern.match(color_str_processed)
    if hex_match:
        rgb_tuple = hex_to_rgb(color_str_processed)
        if rgb_tuple:
             return rgb_tuple
        else:
             logger.warning(f"Regex matched hex '{color_str_processed}' but hex_to_rgb failed. Treating as currentColor.")
             return 'currentcolor'

    # 5. Check for named colors (using imported dict)
    if color_str_lower in NAMED_COLORS_RGB:
        hex_val = NAMED_COLORS_RGB[color_str_lower]
        rgb_tuple = hex_to_rgb(hex_val)
        if rgb_tuple:
             return rgb_tuple
        else:
             logger.warning(f"Invalid hex '{hex_val}' found for named color '{color_str_lower}'. Treating as currentColor.")
             return 'currentcolor'

    # 6. Manually parse rgb()
    rgb_match = rgb_pattern.match(color_str_processed)
    if rgb_match:
        try:
            r_str, g_str, b_str = rgb_match.groups()
            r = _parse_rgb_value(r_str)
            g = _parse_rgb_value(g_str)
            b = _parse_rgb_value(b_str)
            if r is not None and g is not None and b is not None:
                return (r, g, b)
            else:
                 logger.warning(f"Invalid value within rgb() string: '{color_str_processed}'")
        except Exception as e:
             logger.warning(f"Error parsing rgb() string '{color_str_processed}': {e}")
        # Fall through if parsing fails or values are invalid

    # 7. Manually parse rgba() (ignore alpha for now)
    rgba_match = rgba_pattern.match(color_str_processed)
    if rgba_match:
        try:
            r_str, g_str, b_str, a_str = rgba_match.groups()
            r = _parse_rgb_value(r_str)
            g = _parse_rgb_value(g_str)
            b = _parse_rgb_value(b_str)
            if r is not None and g is not None and b is not None:
                try:
                    alpha = float(a_str)
                    if alpha < 1.0:
                        logger.debug(f"RGBA alpha value {alpha} found in '{color_str_processed}'. Alpha is ignored for tokenization, using only RGB.")
                except ValueError:
                    logger.warning(f"Invalid alpha value '{a_str}' in rgba() string '{color_str_processed}'. Alpha ignored.")
                    return (r, g, b) # Return RGB tuple, ignoring alpha
            else:
                 logger.warning(f"Invalid RGB value within rgba() string: '{color_str_processed}'")
        except Exception as e:
            logger.warning(f"Error parsing rgba() string '{color_str_processed}': {e}")
        # Fall through if parsing fails or values are invalid

    # 8. Treat as currentColor as a final fallback
    logger.warning(f"Treating unhandled fill value as currentColor: '{color_str_processed}'")
    return 'currentcolor'

def round_coord(value: float) -> float:
     """Rounds coordinate based on precision."""
     if COORD_PRECISION == 0:
         return round(value)
     return round(value, COORD_PRECISION)

def process_coords(x: float, y: float) -> Tuple[int, int]:
    """Rounds coordinates (half up), clamps to viewbox integers [0, VIEWBOX_DIM]."""
    # Round half up for non-negative values
    x_rounded = int(x + 0.5)
    y_rounded = int(y + 0.5)

    # Clamp to viewbox boundaries [0, WIDTH] and [0, HEIGHT]
    max_x = VIEWBOX_WIDTH  # Use 200 as max value
    max_y = VIEWBOX_HEIGHT # Use 200 as max value

    x_clamped = max(0, min(max_x, x_rounded)) # Ensure x in [0, 200]
    y_clamped = max(0, min(max_y, y_rounded)) # Ensure y in [0, 200]

    # Ensure integer type
    x_int = int(x_clamped)
    y_int = int(y_clamped)

    return (x_int, y_int)

def discretize_arc_params(rx: float, ry: float, angle: float, large_arc_flag: float, sweep_flag: float) -> Tuple[int, int, int, int, int]:
    """Discretizes elliptical arc parameters to integer representations."""
    rx_disc = int(max(0, min(VIEWBOX_WIDTH, round(abs(rx)))))
    ry_disc = int(max(0, min(VIEWBOX_HEIGHT, round(abs(ry)))))
    angle_disc = int(round(angle % 360))
    if angle_disc < 0: angle_disc += 360
    try:
        large_arc_flag_int = int(large_arc_flag)
        sweep_flag_int = int(sweep_flag)
        if large_arc_flag_int not in [0, 1]:
            logger.warning(f"Invalid large-arc-flag value: {large_arc_flag}. Using 0.")
            large_arc_flag_int = 0
        if sweep_flag_int not in [0, 1]:
            logger.warning(f"Invalid sweep-flag value: {sweep_flag}. Using 0.")
            sweep_flag_int = 0
    except (ValueError, TypeError):
         logger.warning(f"Invalid flag values: large_arc={large_arc_flag}, sweep={sweep_flag}. Using 0 for both.")
         large_arc_flag_int = 0
         sweep_flag_int = 0
    return (rx_disc, ry_disc, angle_disc, large_arc_flag_int, sweep_flag_int)
