import pytest
from omnisvg_data_process.tokenizer import SVGTokenizer
from lxml import etree
from .fixture import ACTUALLY_SVG

# Helper to compare SVG structures ignoring whitespace and minor format differences
def svgs_are_equivalent(svg_str1: str, svg_str2: str) -> bool:
    try:
        parser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
        root1 = etree.fromstring(svg_str1.encode('utf-8'), parser=parser)
        root2 = etree.fromstring(svg_str2.encode('utf-8'), parser=parser)
        # Basic check: compare canonicalized XML strings
        # More robust: compare element tags, attributes (order insensitive), text content
        # Using canonical form (c14n) is a good way
        c14n_1 = etree.tostring(root1, method='c14n')
        c14n_2 = etree.tostring(root2, method='c14n')
        return c14n_1 == c14n_2
    except Exception as e:
        print(f"Error comparing SVGs: {e}")
        print("--- SVG 1 ---")
        print(svg_str1)
        print("--- SVG 2 ---")
        print(svg_str2)
        return False

@pytest.fixture(scope="module")
def tokenizer_instance():
    return SVGTokenizer()

# --- Test Cases --- 

# Basic SVG with M, L, Z and simple fill
SVG_BASIC = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <path fill="#ff8000" d="M 10.2 10.8 L 90 10 L 90 90.5 Z" />
</svg>'''
EXPECTED_TOKENS_BASIC = (
    "<｜SVG_START｜>" +
    "<｜PATH_START｜>" +
    "<｜SVG_F｜><｜color_rgb_255｜><｜color_rgb_128｜><｜color_rgb_0｜>" +
    "<｜SVG_M｜><｜coord_10_11｜>" +
    "<｜SVG_L｜><｜coord_90_10｜>" +
    "<｜SVG_L｜><｜coord_90_91｜>" +
    "<｜SVG_Z｜>" +
    "<｜PATH_END｜>" +
    "<｜SVG_END｜>"
)
NORMALIZED_SVG_BASIC = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200px" height="200px">
  <path d="M 10 11 L 90 10 L 90 91 Z" fill="#ff8000"/>
</svg>'''

# SVG with C command, named color, clamping
SVG_CURVE_CLAMP = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <path fill="lime" d="M 50 100 C 20 -30, 180 250, 200.7 100.3" />
</svg>'''
EXPECTED_TOKENS_CURVE_CLAMP = (
    "<｜SVG_START｜>" +
    "<｜PATH_START｜>" +
    "<｜SVG_F｜><｜color_rgb_0｜><｜color_rgb_255｜><｜color_rgb_0｜>" +
    "<｜SVG_M｜><｜coord_50_100｜>" +
    "<｜SVG_C｜><｜coord_20_0｜><｜coord_180_200｜><｜coord_200_100｜>" +
    "<｜PATH_END｜>" +
    "<｜SVG_END｜>"
)
NORMALIZED_SVG_CURVE_CLAMP = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200px" height="200px">
  <path d="M 50 100 C 20 0 180 200 200 100" fill="#00ff00"/>
</svg>'''

# SVG with A command, relative coords, url fill
SVG_ARC_REL = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <path fill="url(#grad)" d="m 10 150 a 30.4 40.6 45 1 0 80 0" />
</svg>'''
EXPECTED_TOKENS_ARC_REL = (
    "<｜SVG_START｜>" +
    "<｜PATH_START｜>" +
    "<｜SVG_F｜><｜fill_url｜>" +
    "<｜SVG_M｜><｜coord_10_150｜>" +
    "<｜SVG_A｜><｜arc_rx_30｜><｜arc_ry_41｜><｜arc_angle_45｜><｜arc_large_1｜><｜arc_sweep_0｜><｜coord_90_150｜>" +
    "<｜PATH_END｜>" +
    "<｜SVG_END｜>"
)
NORMALIZED_SVG_ARC_REL = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200px" height="200px">
  <path d="M 10 150 A 30 41 45 1 0 90 150" fill="url(#default_grad)"/>
</svg>'''

# SVG with fill="none" and fill="currentColor"
SVG_SPECIAL_FILL = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <path fill="none" d="M 10 10 L 20 20"/>
  <path fill="currentColor" d="M 30 30 L 40 40"/>
</svg>'''
EXPECTED_TOKENS_SPECIAL_FILL = (
    "<｜SVG_START｜>" +
    "<｜PATH_START｜>" +
    # No fill token for none
    "<｜SVG_M｜><｜coord_10_10｜>" +
    "<｜SVG_L｜><｜coord_20_20｜>" +
    "<｜PATH_END｜>" +
    "<｜PATH_START｜>" +
    "<｜SVG_F｜><｜fill_currentcolor｜>" +
    "<｜SVG_M｜><｜coord_30_30｜>" +
    "<｜SVG_L｜><｜coord_40_40｜>" +
    "<｜PATH_END｜>" +
    "<｜SVG_END｜>"
)
NORMALIZED_SVG_SPECIAL_FILL = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200px" height="200px">
  <path d="M 10 10 L 20 20" fill="none"/>
  <path d="M 30 30 L 40 40" fill="currentColor"/>
</svg>'''



# --- Test Functions --- 

def test_svg_to_tokens_basic(tokenizer_instance):
    tokens = tokenizer_instance.svg_to_tokens(SVG_BASIC)
    assert tokens == EXPECTED_TOKENS_BASIC

def test_svg_to_tokens_curve_clamp(tokenizer_instance):
    tokens = tokenizer_instance.svg_to_tokens(SVG_CURVE_CLAMP)
    assert tokens == EXPECTED_TOKENS_CURVE_CLAMP

def test_svg_to_tokens_arc_rel(tokenizer_instance):
    tokens = tokenizer_instance.svg_to_tokens(SVG_ARC_REL)
    assert tokens == EXPECTED_TOKENS_ARC_REL

def test_svg_to_tokens_special_fill(tokenizer_instance):
    tokens = tokenizer_instance.svg_to_tokens(SVG_SPECIAL_FILL)
    assert tokens == EXPECTED_TOKENS_SPECIAL_FILL

def test_tokens_to_svg_basic(tokenizer_instance):
    reconstructed_svg = tokenizer_instance.tokens_to_svg(EXPECTED_TOKENS_BASIC)
    assert reconstructed_svg is not None
    # Compare with normalized version for consistency
    normalized_original = tokenizer_instance.normalize_svg(SVG_BASIC)
    assert svgs_are_equivalent(reconstructed_svg, normalized_original)
    # Also check against pre-defined normalized string
    assert svgs_are_equivalent(reconstructed_svg, NORMALIZED_SVG_BASIC)

def test_tokens_to_svg_curve_clamp(tokenizer_instance):
    reconstructed_svg = tokenizer_instance.tokens_to_svg(EXPECTED_TOKENS_CURVE_CLAMP)
    assert reconstructed_svg is not None
    normalized_original = tokenizer_instance.normalize_svg(SVG_CURVE_CLAMP)
    assert svgs_are_equivalent(reconstructed_svg, normalized_original)
    assert svgs_are_equivalent(reconstructed_svg, NORMALIZED_SVG_CURVE_CLAMP)

def test_tokens_to_svg_arc_rel(tokenizer_instance):
    reconstructed_svg = tokenizer_instance.tokens_to_svg(EXPECTED_TOKENS_ARC_REL)
    assert reconstructed_svg is not None
    normalized_original = tokenizer_instance.normalize_svg(SVG_ARC_REL)
    assert svgs_are_equivalent(reconstructed_svg, normalized_original)
    assert svgs_are_equivalent(reconstructed_svg, NORMALIZED_SVG_ARC_REL)

def test_tokens_to_svg_special_fill(tokenizer_instance):
    reconstructed_svg = tokenizer_instance.tokens_to_svg(EXPECTED_TOKENS_SPECIAL_FILL)
    assert reconstructed_svg is not None
    normalized_original = tokenizer_instance.normalize_svg(SVG_SPECIAL_FILL)
    assert svgs_are_equivalent(reconstructed_svg, normalized_original)
    assert svgs_are_equivalent(reconstructed_svg, NORMALIZED_SVG_SPECIAL_FILL)

# Test edge case: empty SVG content
def test_empty_svg(tokenizer_instance):
    empty_svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200px" height="200px"></svg>'
    original_empty_svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"></svg>'
    tokens = tokenizer_instance.svg_to_tokens(original_empty_svg)
    expected = "<｜SVG_START｜><｜SVG_END｜>"
    assert tokens == expected
    reconstructed = tokenizer_instance.tokens_to_svg(expected)
    assert svgs_are_equivalent(reconstructed, empty_svg)

# Test edge case: invalid token string
def test_invalid_tokens(tokenizer_instance):
    invalid_tokens_str = "<｜SVG_START｜>invalid<｜SVG_END｜>"
    reconstructed = tokenizer_instance.tokens_to_svg(invalid_tokens_str)
    # Depending on error handling, might return None or empty SVG
    assert reconstructed is not None # Assuming detokenize handles errors gracefully
    # Check it's at least a valid SVG root (standard or self-closing)
    assert "<svg" in reconstructed and ("</svg>" in reconstructed or reconstructed.strip().endswith(">"))

def test_empty_token_string(tokenizer_instance):
     reconstructed = tokenizer_instance.tokens_to_svg("")
     expected_empty_svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200px" height="200px"></svg>'
     assert svgs_are_equivalent(reconstructed, expected_empty_svg)

# --- Tests for Real SVGs from Fixture --- 

# Use the imported list directly
try:
    real_svg_data = ACTUALLY_SVG
    # Extract just the SVG strings, creating IDs for parameterization
    real_svgs_parametrized = [(entry['svg'], entry.get('id', f"real_svg_{i}")) for i, entry in enumerate(real_svg_data)]
except (TypeError, KeyError, AttributeError) as e:
    print(f"Error: ACTUALLY_SVG structure incorrect or not found ({e})")
    real_svgs_parametrized = []

@pytest.mark.parametrize("svg_string, svg_id", real_svgs_parametrized, ids=[p[1] for p in real_svgs_parametrized])
def test_real_svg_full_cycle(tokenizer_instance, svg_string, svg_id):
    """Tests the full normalize -> tokenize -> detokenize cycle for real SVGs."""
    # 1. Normalize the original SVG
    # Using svg_id for logging within normalize_svg if needed
    normalized_svg = tokenizer_instance.normalize_svg(svg_string, file_path_for_logging=svg_id)
    assert normalized_svg is not None, f"Normalization failed for {svg_id}"

    # 2. Tokenize the original SVG (using the same tokenizer instance)
    tokens = tokenizer_instance.svg_to_tokens(svg_string)
    assert tokens is not None, f"Tokenization failed for {svg_id}"
    assert len(tokens) > 0, f"Tokenization produced empty tokens for {svg_id}"

    # 3. Detokenize the tokens back to SVG
    reconstructed_svg = tokenizer_instance.tokens_to_svg(tokens)
    assert reconstructed_svg is not None, f"Detokenization failed for {svg_id}"

    # 4. Assert that the detokenized SVG is equivalent to the normalized SVG
    assert svgs_are_equivalent(reconstructed_svg, normalized_svg), f"Reconstructed SVG does not match normalized SVG for {svg_id}" 
