"""
AgroPulse — Assets & Mascot Helper
===================================
Handles automatic detection, slicing, transparency processing, and caching
of robot mascot poses and official NASA Space Apps Challenge branding assets.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent


def clean_html(html_str: str) -> str:
    """
    Strips all leading and trailing whitespace from each line of an HTML string,
    eliminating blank lines. This prevents Markdown / CommonMark parsers from
    interpreting lines with >=4 spaces as indented <pre><code> code blocks.
    """
    if not html_str:
        return ""
    return "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())


# Official SVG Logo for AgroPulse (Blue-Green globe, white sprout, golden orbit)
AGROPULSE_LOGO_SVG = clean_html("""
<svg width="46" height="46" viewBox="0 0 64 64" role="img" aria-label="AgroPulse Logo">
<defs>
  <linearGradient id="gl" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="#2a6fb5"/>
    <stop offset="100%" stop-color="#2f9e63"/>
  </linearGradient>
  <clipPath id="cp">
    <circle cx="32" cy="32" r="22"/>
  </clipPath>
</defs>
<circle cx="32" cy="32" r="22" fill="url(#gl)"/>
<g clip-path="url(#cp)" stroke="#f4e3a6" stroke-width="1.6" fill="none" opacity=".75">
  <path d="M6 46Q32 38 58 46"/>
  <path d="M6 52Q32 44 58 52"/>
  <path d="M6 58Q32 50 58 58"/>
</g>
<path d="M32 44V28" stroke="#fff" stroke-width="2.6" stroke-linecap="round"/>
<path d="M32 33C25 33 22 28 22 23c6 0 10 3 10 10z" fill="#c9f2b5"/>
<path d="M32 29c0-6 3-10 10-10 0 6-4 10-10 10z" fill="#fff"/>
<ellipse cx="32" cy="32" rx="30" ry="9" transform="rotate(-24 32 32)" fill="none" stroke="#e6a92c" stroke-width="2.6"/>
<circle cx="56" cy="21" r="3.6" fill="#e6a92c" stroke="#fff" stroke-width="1.2"/>
</svg>
""")


def _find_asset(filename_keyword: str) -> Optional[Path]:
    """Search for asset files in RUTA_FINAL or assets directories."""
    candidates = [
        BASE_DIR / "RUTA_FINAL" / "3_ENTREGABLES" / "Prototipo",
        BASE_DIR / "RUTA_FINAL",
        BASE_DIR / "assets",
        BASE_DIR,
    ]
    for folder in candidates:
        if folder.exists():
            for root, _, files in os.walk(folder):
                for f in files:
                    if filename_keyword.lower() in f.lower():
                        return Path(root) / f
    return None


@st.cache_data(show_spinner=False)
def load_robot_mascot_poses() -> dict[str, str]:
    """
    Finds the 6-robot collage, slices it into a 2x3 grid, converts pure white
    background (R>240, G>240, B>240) to clean RGBA transparency, and encodes to base64 data-URIs.

    Grid Mapping:
      (row 0, col 0) -> 'wave' (welcoming farmer)
      (row 0, col 1) -> 'thumbs_up' (approval, success)
      (row 0, col 2) -> 'point' (directing focus to sliders)
      (row 1, col 0) -> 'run' (stress test / simulation in action)
      (row 1, col 1) -> 'salute' (scientific rigor / NASA data)
      (row 1, col 2) -> 'celebrate' (recommended winning plan)
    """
    collage_path = _find_asset("Collage de robots")
    if not collage_path or not collage_path.exists():
        return {}

    im = Image.open(collage_path)
    w, h = im.size
    cell_w = w // 3
    cell_h = h // 2

    pose_coords = {
        "wave": (0, 0),
        "thumbs_up": (0, 1),
        "point": (0, 2),
        "run": (1, 0),
        "salute": (1, 1),
        "celebrate": (1, 2),
    }

    result = {}
    for name, (row, col) in pose_coords.items():
        box = (col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h)
        crop = im.crop(box).convert("RGBA")
        arr = np.array(crop)

        # Make white background transparent
        white_mask = (arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240)
        arr[white_mask, 3] = 0

        # Also soften semi-white anti-aliased edge pixels
        near_white = (arr[:, :, 0] > 220) & (arr[:, :, 1] > 220) & (arr[:, :, 2] > 220) & (~white_mask)
        arr[near_white, 3] = (255 - ((arr[near_white, :3].mean(axis=-1) - 220) * (255 / 20))).astype(np.uint8)

        cleaned = Image.fromarray(arr)
        # Resize to crisp web display size (256x256)
        cleaned = cleaned.resize((256, 256), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        cleaned.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        result[name] = f"data:image/png;base64,{b64}"

    return result


@st.cache_data(show_spinner=False)
def load_nasa_challenge_banner_uri() -> Optional[str]:
    """Loads and returns the base64 data-URI of the NASA Space Apps Challenge robot banner."""
    banner_path = _find_asset("seedream") or _find_asset("ElevenLabs")
    if not banner_path or not banner_path.exists():
        return None

    im = Image.open(banner_path)
    # Resize slightly for crisp fast web delivery (1024 width max)
    im.thumbnail((1024, 768), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def render_mascot_bubble(
    pose: str,
    speech_text: str,
    bubble_title: str = "",
    side: str = "left",
    mascot_size: int = 86,
    theme_color: str = "#10B981",
) -> str:
    """
    Renders a friendly Glassmorphism speech bubble with the floating transparent robot mascot.
    All internal HTML is cleaned of leading whitespace to prevent Markdown code block wrapping.
    """
    poses = load_robot_mascot_poses()
    mascot_src = poses.get(pose, "")

    img_html = ""
    if mascot_src:
        img_html = (
            f'<div style="flex-shrink: 0; display: flex; align-items: center; justify-content: center;">'
            f'<img src="{mascot_src}" alt="AgroPulse Robot ({pose})" '
            f'style="width: {mascot_size}px; height: {mascot_size}px; object-fit: contain; '
            f'filter: drop-shadow(0 8px 16px rgba(0,0,0,0.45));" />'
            f'</div>'
        )

    title_html = (
        f"<div style='font-weight:700; font-size:1.0rem; color:#FFFFFF; margin-bottom:4px;'>{bubble_title}</div>"
        if bubble_title
        else ""
    )

    flex_dir = "row" if side == "left" else "row-reverse"

    bubble_raw = f"""
<div style="display: flex; flex-direction: {flex_dir}; align-items: center; gap: 16px; 
background: rgba(19, 35, 27, 0.65); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
border: 1px solid rgba(38, 77, 56, 0.6); border-radius: 20px; padding: 14px 18px; margin: 12px 0 16px 0;
box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);">
{img_html}
<div style="flex-grow: 1; min-width: 0;">
{title_html}
<div style="font-size: 0.92rem; line-height: 1.5; color: #E2E8F0;">
{speech_text}
</div>
</div>
</div>
"""
    return clean_html(bubble_raw)
