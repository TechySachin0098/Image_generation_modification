'''
import os
import sys
import argparse
import requests
import numpy as np
import cv2
from PIL import Image
import io
import urllib.request


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

API_KEY = "" # Get from platform.stability.ai
API_URL = "https://api.stability.ai/v2beta/stable-image/edit/inpaint"

INPAINT_STRENGTH = 0.70   # 0.60–0.75 recommended for face preservation
OUTPUT_FORMAT    = "png"

# ─────────────────────────────────────────────────────────────
# FIX A CONSTANT:
# Haar frontal face cascade typically starts the bounding box at the eyebrows,
# not the top of the forehead. We compensate by shifting the detected top of
# the face box upward by this fraction of the detected face height.
# 0.30 recovers roughly the forehead on most portrait crops.
# If the forehead is STILL being regenerated, try increasing this to 0.35–0.40.
HAAR_FOREHEAD_COMPENSATION = 0.30
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# PROMPT PRESETS
# ─────────────────────────────────────────────────────────────

PRESETS = {
    1: {
        "name": "Short Black Beard + Dark Hair",
        "prompt": "short neat black beard, short dark brown hair, natural lighting, photorealistic, high quality",
        "negative": "long beard, grey hair, bald, distorted face, blurry, cartoon"
    },
    2: {
        "name": "Long Red Beard + Blonde Hair",
        "prompt": "long full red beard, medium length blonde wavy hair, natural lighting, photorealistic, high quality",
        "negative": "short beard, dark hair, bald, distorted face, blurry, cartoon"
    },
    3: {
        "name": "Clean Shaven + Black Curly Hair",
        "prompt": "no beard, clean shaven face, short curly black hair, natural lighting, photorealistic, high quality",
        "negative": "beard, stubble, distorted face, blurry, cartoon"
    },
    4: {
        "name": "Grey Beard + Salt & Pepper Hair",
        "prompt": "medium grey beard, short salt and pepper hair, distinguished look, natural lighting, photorealistic",
        "negative": "dark beard, colored hair, bald, distorted face, blurry, cartoon"
    },
    5: {
        "name": "White Hair + Large Fluffy Beard",
        "prompt": "large fluffy white beard, long white hair, natural lighting, photorealistic, high quality",
        "negative": "dark beard, short beard, bald, distorted face, blurry, cartoon"
    },
    6: {
        "name": "Bald + Short Stubble",
        "prompt": "bald head, short dark stubble, natural lighting, photorealistic, high quality",
        "negative": "hair, long beard, distorted face, blurry, cartoon"
    },
    7: {
        "name": "Afro + Full Beard",
        "prompt": "large natural afro hair, full thick dark beard, natural lighting, photorealistic, high quality",
        "negative": "straight hair, no beard, bald, distorted face, blurry, cartoon"
    },
}


# ─────────────────────────────────────────────────────────────
# HAAR CASCADE DOWNLOADER
# ─────────────────────────────────────────────────────────────

def ensure_cascade(filename: str) -> str:
    """Download OpenCV Haar cascade XML if not present locally."""
    if os.path.exists(filename):
        return filename
    base_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
    url = base_url + filename
    print(f"  ↳ Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"  ✅ Downloaded: {filename}")
    except Exception as e:
        print(f"  ⚠️  Could not download {filename}: {e}")
        sys.exit(1)
    return filename


# ─────────────────────────────────────────────────────────────
# STEP 1: IMAGE LOADING
# ─────────────────────────────────────────────────────────────

def load_and_resize(image_path: str, max_size: int = 1024) -> np.ndarray:
    """Load image and resize to fit within max_size."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        print(f"  ↳ Resized to {image.shape[1]}x{image.shape[0]}")
    else:
        print(f"  ↳ Image size: {w}x{h} (no resize needed)")
    return image


# ─────────────────────────────────────────────────────────────
# STEP 2: FACE DETECTION  (with forehead compensation)
# ─────────────────────────────────────────────────────────────

def detect_face(image_bgr: np.ndarray) -> tuple:
    """
    Detect the main face using Haar cascade and apply forehead compensation.

    FIX A — WHY THIS IS NEEDED:
    OpenCV's haarcascade_frontalface_default starts its bounding box at or just
    below the eyebrows in most close-crop portraits. If we use fy as-is, then
    hair_bottom = fy + 5%*fh lands at eyebrow level, and the entire forehead
    gets painted into the hair mask and regenerated by the API.

    HOW WE FIX IT:
    After detection, we shift fy upward by HAAR_FOREHEAD_COMPENSATION * fh
    (default 30%). This moves the "top of face" up by ~30% of the face height,
    recovering the forehead region so it stays OUTSIDE the hair mask.

    The face box returned by this function is the CORRECTED box used for all
    subsequent mask calculations.
    """
    cascade_file = ensure_cascade("haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_file)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(60, 60)
    )

    h_img, w_img = image_bgr.shape[:2]

    if len(faces) == 0:
        print("  ⚠️  No face detected — using center-image fallback")
        fx = int(w_img * 0.20)
        fy = int(h_img * 0.10)
        fw = int(w_img * 0.60)
        fh = int(h_img * 0.65)
    else:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        fx, fy, fw, fh = faces[0]
        print(f"  ↳ Raw Haar box:  x={fx} y={fy} w={fw} h={fh}")

    # FIX A: Shift the top of the face box upward to recover the forehead.
    # The Haar detector typically starts at eyebrow level; we move it up by
    # 30% of the detected face height to approximately reach the true hairline.
    forehead_offset = int(fh * HAAR_FOREHEAD_COMPENSATION)
    fy_corrected    = max(0, fy - forehead_offset)
    fh_corrected    = fh + forehead_offset   # keep the bottom edge where it was

    print(f"  ↳ Corrected box: x={fx} y={fy_corrected} w={fw} h={fh_corrected}  "
          f"(forehead +{forehead_offset}px)")

    return (fx, fy_corrected, fw, fh_corrected)


# ─────────────────────────────────────────────────────────────
# STEP 3: MASK GENERATION
# ─────────────────────────────────────────────────────────────

def generate_hair_mask(image_bgr: np.ndarray, face: tuple) -> np.ndarray:
    """
    Hair region: strictly ABOVE the corrected face bounding box.

    With FIX A applied, fy now represents the true top of the face (hairline),
    so hair_bottom = fy + 5%*fh correctly stays just at the hairline edge
    without dipping into the forehead.

    FIX B — Side coverage is now constrained:
    Old code painted two full-height rectangles spanning the full image width.
    On a portrait with a non-white background, this masked the entire room.
    We now limit side panels to a tight corridor beside the face and cap their
    height to the ear level so the background stays untouched.

    Visual:
        [side]  [── top hair ──────────────────]  [side]
                │  stays above fy + 5%           │
        ────────┼────── fy (corrected hairline) ─┼────
                │   FACE PROTECTED               │
    """
    h_img, w_img = image_bgr.shape[:2]
    fx, fy, fw, fh = face
    mask = np.zeros((h_img, w_img), dtype=np.uint8)

    # Main top-hair rectangle: above and just touching the hairline
    hair_bottom = fy + int(fh * 0.05)
    hair_left   = max(0, fx - int(fw * 0.40))
    hair_right  = min(w_img, fx + fw + int(fw * 0.40))
    cv2.rectangle(mask, (hair_left, 0), (hair_right, hair_bottom), 255, -1)

    # FIX B — Narrow side panels for temples / sideburns only
    side_bottom = fy + int(fh * 0.55)       # ear mid-level — cap here
    side_width  = int(fw * 0.30)            # tight corridor, not full image width

    left_panel_right = max(0, fx - int(fw * 0.05))
    left_panel_left  = max(0, left_panel_right - side_width)
    cv2.rectangle(mask, (left_panel_left, 0), (left_panel_right, side_bottom), 255, -1)

    right_panel_left  = min(w_img, fx + fw + int(fw * 0.05))
    right_panel_right = min(w_img, right_panel_left + side_width)
    cv2.rectangle(mask, (right_panel_left, 0), (right_panel_right, side_bottom), 255, -1)

    return mask


def generate_beard_mask(image_bgr: np.ndarray, face: tuple) -> np.ndarray:
    """
    Beard region: from just below the mouth down to just below the chin.

    FIX C — Beard bottom reduced from 40% → 20% below the face box.
    On a close-crop selfie (face filling most of the frame), 40% below the
    face box reaches the collarbone/shirt. 20% gives enough room for a large
    beard while staying off the clothing.

    The top boundary (82% down the face box) is unchanged from v2 —
    this correctly starts below the lower lip.

    Visual:
        fy + 82% fh  ← beard top (below lips)
        ─────────────────────────────────────
        [  BEARD MASK AREA  ]
        ─────────────────────────────────────
        fy + fh + 20% fh  ← beard bottom (was +40%)
    """
    h_img, w_img = image_bgr.shape[:2]
    fx, fy, fw, fh = face
    mask = np.zeros((h_img, w_img), dtype=np.uint8)

    beard_top    = fy + int(fh * 0.82)
    beard_bottom = min(h_img, fy + fh + int(fh * 0.20))   # FIX C: was 0.40

    beard_left  = max(0, fx - int(fw * 0.10))
    beard_right = min(w_img, fx + fw + int(fw * 0.10))

    cv2.rectangle(mask, (beard_left, beard_top), (beard_right, beard_bottom), 255, -1)

    return mask


def cut_face_protection_zone(mask: np.ndarray, face: tuple) -> np.ndarray:
    """
    Hard-erase any mask that leaked into the core face region after dilation/blur.
    Guarantees eyes, nose, cheeks, and mouth are never regenerated.
    """
    fx, fy, fw, fh = face
    prot_top    = fy + int(fh * 0.05)
    prot_bottom = fy + int(fh * 0.80)
    prot_left   = fx + int(fw * 0.05)
    prot_right  = fx + fw - int(fw * 0.05)
    mask[prot_top:prot_bottom, prot_left:prot_right] = 0
    return mask


def combine_and_refine_mask(
    hair_mask: np.ndarray,
    beard_mask: np.ndarray,
    face: tuple,
    dilate_px: int = 8,
    blur_r: int = 11
) -> np.ndarray:
    """Combine masks, apply conservative smoothing, then hard-cut face zone."""
    combined = cv2.bitwise_or(hair_mask, beard_mask)

    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    combined = cv2.dilate(combined, kernel, iterations=1)

    if blur_r % 2 == 0:
        blur_r += 1
    combined = cv2.GaussianBlur(combined, (blur_r, blur_r), 0)
    _, combined = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)

    combined = cut_face_protection_zone(combined, face)
    return combined


def generate_mask(image_path: str, save_path: str = "mask.png"):
    """Full mask pipeline. Returns (mask_path, image_bgr, face)."""
    print("\n📐 STEP 1: Generating hair & beard mask...")
    image_bgr = load_and_resize(image_path)

    print("  ↳ Detecting face (with forehead compensation)...")
    face = detect_face(image_bgr)

    print("  ↳ Building hair mask (above corrected hairline)...")
    hair_mask = generate_hair_mask(image_bgr, face)

    print("  ↳ Building beard mask (below lips, above shirt)...")
    beard_mask = generate_beard_mask(image_bgr, face)

    print("  ↳ Combining, refining, protecting face zone...")
    final_mask = combine_and_refine_mask(hair_mask, beard_mask, face)

    Image.fromarray(final_mask).save(save_path)
    print(f"  ✅ Mask saved: {save_path}")

    return save_path, image_bgr, face


# ─────────────────────────────────────────────────────────────
# STEP 4: MASK PREVIEW
# ─────────────────────────────────────────────────────────────

def save_mask_preview(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    face: tuple,
    save_path: str
):
    """
    Side-by-side: Original | Red overlay + diagnostic boxes.

    Boxes drawn on the overlay:
      Yellow — Corrected face bounding box (after forehead compensation)
      Green  — Hard protection zone (eyes → lips; NO red should be inside this)
      Cyan   — Where the raw Haar box started (before forehead correction)
    """
    image_rgb    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = image_rgb.shape[:2]
    fx, fy, fw, fh = face

    overlay   = image_rgb.copy()
    red_layer = np.zeros_like(image_rgb)
    red_layer[:, :] = [220, 50, 50]
    mask_bool = mask > 127
    overlay[mask_bool] = (0.45 * overlay[mask_bool] + 0.55 * red_layer[mask_bool]).astype(np.uint8)

    # Corrected face box — yellow
    cv2.rectangle(overlay, (fx, fy), (fx + fw, fy + fh), (255, 220, 0), 2)

    # Protection zone — green
    prot_top    = fy + int(fh * 0.05)
    prot_bottom = fy + int(fh * 0.80)
    prot_left   = fx + int(fw * 0.05)
    prot_right  = fx + fw - int(fw * 0.05)
    cv2.rectangle(overlay, (prot_left, prot_top), (prot_right, prot_bottom), (0, 255, 80), 2)

    # Raw Haar top line — cyan (where eyebrows are, before correction)
    raw_fy = fy + int(fh * HAAR_FOREHEAD_COMPENSATION)
    cv2.line(overlay, (fx, raw_fy), (fx + fw, raw_fy), (0, 220, 255), 1)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    orig_copy = image_rgb.copy()
    cv2.putText(orig_copy, "Original",             (10, 28), font, 0.8, (255, 255, 255), 2)
    cv2.putText(overlay,   "RED = Will Change",    (10, 28), font, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay,   "Yellow = Face Box",    (10, 52), font, 0.55, (255, 220, 0),  1)
    cv2.putText(overlay,   "Green  = Protected",   (10, 72), font, 0.55, (0, 255, 80),   1)
    cv2.putText(overlay,   "Cyan   = Raw Haar",    (10, 92), font, 0.55, (0, 220, 255),  1)

    divider   = np.ones((h_img, 5, 3), dtype=np.uint8) * 200
    composite = np.hstack([orig_copy, divider, overlay])
    Image.fromarray(composite).save(save_path)
    print(f"  ✅ Preview saved: {save_path}")
    print(f"     CHECKLIST:")
    print(f"       ✔  RED on top of head (hair) — should not touch the eyebrows")
    print(f"       ✔  RED on chin/jaw (beard) — should not reach the shirt collar")
    print(f"       ✔  GREEN box interior has NO red (eyes, nose, cheeks safe)")
    print(f"       ✔  Background has minimal red — only narrow side corridors")


# ─────────────────────────────────────────────────────────────
# STEP 5: STABILITY AI INPAINTING
# ─────────────────────────────────────────────────────────────

def call_stability_inpaint(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    prompt: str,
    negative_prompt: str = "",
    strength: float = INPAINT_STRENGTH,
    seed: int = 42,
    output_path: str = "output.png"
) -> str:
    """Send image + mask + prompt to Stability AI. Only masked area changes."""
    if not API_KEY:
        raise ValueError(
            "\n❌ API key not set!\n"
            "   Set API_KEY at the top of this file.\n"
            "   Get one from: https://platform.stability.ai/account/keys"
        )

    print("\n🚀 STEP 3: Calling Stability AI Inpainting API...")
    print(f"  ↳ Prompt:   {prompt}")
    print(f"  ↳ Strength: {strength}  |  Seed: {seed}")

    def to_png_bytes(arr):
        pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if len(arr.shape) == 3 else arr)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()

    base_neg = (
        "distorted face, changed face, different person, different identity, "
        "different nose, different eyes, different skin tone, different age, "
        "morphed features, blurry face, cartoon, anime, painting, illustration"
    )
    full_neg = f"{negative_prompt}, {base_neg}".strip(", ")

    response = requests.post(
        API_URL,
        headers={"authorization": f"Bearer {API_KEY}", "accept": "image/*"},
        files={
            "image": ("image.png", to_png_bytes(image_bgr), "image/png"),
            "mask":  ("mask.png",  to_png_bytes(mask),      "image/png"),
        },
        data={
            "prompt":          prompt,
            "negative_prompt": full_neg,
            "output_format":   OUTPUT_FORMAT,
            "strength":        str(strength),
            "seed":            str(seed),
            "grow_mask":       "0",   # do NOT let the API expand the mask further
        }
    )

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"  ✅ Output saved: {output_path}")
        return output_path
    else:
        msg = response.text
        print(f"\n  ❌ API Error {response.status_code}: {msg}")
        if response.status_code == 403:
            print("     → Invalid API key or insufficient credits")
        elif response.status_code == 413:
            print("     → Image too large — reduce max_size in load_and_resize()")
        elif response.status_code == 422:
            print("     → Invalid parameters — check prompt length < 10,000 chars")
        raise RuntimeError(f"API call failed {response.status_code}: {msg}")


# ─────────────────────────────────────────────────────────────
# STEP 6: BEFORE / AFTER COMPARISON
# ─────────────────────────────────────────────────────────────

def save_comparison(original_path: str, output_path: str, save_path: str):
    """Side-by-side BEFORE / AFTER image."""
    orig = np.array(Image.open(original_path).convert("RGB"))
    out  = np.array(Image.open(output_path).convert("RGB"))
    if orig.shape != out.shape:
        out = cv2.resize(out, (orig.shape[1], orig.shape[0]))
    font   = cv2.FONT_HERSHEY_SIMPLEX
    orig_c = orig.copy()
    out_c  = out.copy()
    cv2.putText(orig_c, "BEFORE", (10, 32), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out_c,  "AFTER",  (10, 32), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    divider    = np.ones((orig.shape[0], 6, 3), dtype=np.uint8) * 230
    comparison = np.hstack([orig_c, divider, out_c])
    Image.fromarray(comparison).save(save_path)
    print(f"  ✅ Comparison saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    image_path: str,
    prompt: str,
    negative_prompt: str = "",
    preview_only: bool = False,
    strength: float = INPAINT_STRENGTH,
    seed: int = 42
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    base         = os.path.splitext(image_path)[0]
    mask_path    = f"{base}_mask.png"
    preview_path = f"{base}_mask_preview.png"
    output_path  = f"{base}_modified.png"
    compare_path = f"{base}_comparison.png"

    print(f"\n{'='*62}")
    print(f"  HAIR & BEARD MODIFIER  v3 (face-safe + background-safe)")
    print(f"{'='*62}")
    print(f"  Input:    {image_path}")
    print(f"  Prompt:   {prompt}")
    print(f"  Strength: {strength}")
    print(f"{'='*62}")

    mask_path_saved, image_bgr, face = generate_mask(image_path, mask_path)
    mask = cv2.imread(mask_path_saved, cv2.IMREAD_GRAYSCALE)

    print("\n👁️  STEP 2: Saving annotated mask preview...")
    save_mask_preview(image_bgr, mask, face, preview_path)

    if preview_only:
        print(f"\n⏸️  Preview-only mode.")
        print(f"   Open '{preview_path}' and check the checklist printed above.")
        print(f"   Rerun without --preview-only when satisfied.")
        return

    call_stability_inpaint(
        image_bgr=image_bgr, mask=mask, prompt=prompt,
        negative_prompt=negative_prompt, strength=strength,
        seed=seed, output_path=output_path
    )

    print("\n🖼️  STEP 4: Creating before/after comparison...")
    save_comparison(image_path, output_path, compare_path)

    print(f"\n{'='*62}")
    print(f"  ✅ DONE!  Files generated:")
    print(f"     • Mask:       {mask_path}")
    print(f"     • Preview:    {preview_path}")
    print(f"     • Result:     {output_path}")
    print(f"     • Comparison: {compare_path}")
    print(f"{'='*62}\n")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Modify hair and beard using AI inpainting (v3 — face & background safe)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python hair_beard_modifier.py --image photo.jpg --prompt "long red beard, blonde wavy hair"
  python hair_beard_modifier.py --image photo.jpg --preset 5
  python hair_beard_modifier.py --image photo.jpg --preview-only
  python hair_beard_modifier.py --image photo.jpg --preset 2 --strength 0.65 --seed 99

PRESETS:
  1: Short Black Beard + Dark Hair
  2: Long Red Beard + Blonde Hair
  3: Clean Shaven + Black Curly Hair
  4: Grey Beard + Salt & Pepper Hair
  5: White Hair + Large Fluffy Beard
  6: Bald + Short Stubble
  7: Afro + Full Beard

STRENGTH GUIDE:
  0.60  Very subtle — maximum face safety
  0.70  Recommended (default)
  0.80  Stronger — slight risk of edge drift
  0.90  Aggressive — may alter face edges

TUNING TIPS:
  • Forehead still regenerating?
    Increase HAAR_FOREHEAD_COMPENSATION (default 0.30) to 0.35 or 0.40.
  • Beard reaching the shirt?
    Decrease beard_bottom in generate_beard_mask() below 0.20.
  • Background still going red?
    Decrease side_width in generate_hair_mask() below 0.30.
        """
    )

    parser.add_argument("--image",        required=True,        help="Path to input image (jpg/png)")
    parser.add_argument("--prompt",       default=None,         help="Describe the desired hair/beard")
    parser.add_argument("--negative",     default="",           help="What to avoid in the output")
    parser.add_argument("--preset",       type=int,             help="Use a preset style (1-7)")
    parser.add_argument("--preview-only", action="store_true",  help="Generate mask preview only, no API call")
    parser.add_argument("--strength",     type=float, default=INPAINT_STRENGTH, help="0.0–1.0 (default 0.70)")
    parser.add_argument("--seed",         type=int,   default=42,               help="Seed for reproducibility")

    args = parser.parse_args()

    if args.preset:
        if args.preset not in PRESETS:
            print(f"❌ Invalid preset. Choose 1–{len(PRESETS)}")
            sys.exit(1)
        p = PRESETS[args.preset]
        prompt, negative = p["prompt"], p["negative"]
        print(f"\n🎨 Using preset {args.preset}: {p['name']}")
    elif args.prompt:
        prompt   = args.prompt
        negative = args.negative
    elif args.preview_only:
        prompt   = "photorealistic portrait"
        negative = ""
    else:
        parser.print_help()
        print("\n❌ Provide --prompt or --preset (or --preview-only to check the mask)")
        sys.exit(1)

    run_pipeline(
        image_path=args.image, prompt=prompt, negative_prompt=negative,
        preview_only=args.preview_only, strength=args.strength, seed=args.seed
    )


if __name__ == "__main__":
    main()     '''


import os
import sys
import argparse
import requests
import numpy as np
import cv2
from PIL import Image
import io
import urllib.request
import base64


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

API_KEY = "5a4ebdb6-f633-4487-b3f2-a67f1007abba:14951172760f0fc6fef81da5f4f2e610" # Get from fal.ai/dashboard/keys
API_URL = "https://fal.run/fal-ai/flux-pro/v1/fill"

OUTPUT_FORMAT    = "png"

# ─────────────────────────────────────────────────────────────
# FIX A CONSTANT:
# Haar frontal face cascade typically starts the bounding box at the eyebrows,
# not the top of the forehead. We compensate by shifting the detected top of
# the face box upward by this fraction of the detected face height.
# 0.30 recovers roughly the forehead on most portrait crops.
# If the forehead is STILL being regenerated, try increasing this to 0.35–0.40.
HAAR_FOREHEAD_COMPENSATION = 0.30
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# HAAR CASCADE DOWNLOADER
# ─────────────────────────────────────────────────────────────

def ensure_cascade(filename: str) -> str:
    """Download OpenCV Haar cascade XML if not present locally."""
    if os.path.exists(filename):
        return filename
    base_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
    url = base_url + filename
    print(f"  ↳ Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"  ✅ Downloaded: {filename}")
    except Exception as e:
        print(f"  ⚠️  Could not download {filename}: {e}")
        sys.exit(1)
    return filename


# ─────────────────────────────────────────────────────────────
# STEP 1: IMAGE LOADING
# ─────────────────────────────────────────────────────────────

def load_and_resize(image_path: str, max_size: int = 1024) -> np.ndarray:
    """Load image and resize to fit within max_size."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        print(f"  ↳ Resized to {image.shape[1]}x{image.shape[0]}")
    else:
        print(f"  ↳ Image size: {w}x{h} (no resize needed)")
    return image


# ─────────────────────────────────────────────────────────────
# STEP 2: FACE DETECTION  (with forehead compensation)
# ─────────────────────────────────────────────────────────────

def detect_face(image_bgr: np.ndarray) -> tuple:
    """Detect the main face using Haar cascade and apply forehead compensation."""
    cascade_file = ensure_cascade("haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_file)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(60, 60)
    )

    h_img, w_img = image_bgr.shape[:2]

    if len(faces) == 0:
        print("  ⚠️  No face detected — using center-image fallback")
        fx = int(w_img * 0.20)
        fy = int(h_img * 0.10)
        fw = int(w_img * 0.60)
        fh = int(h_img * 0.65)
    else:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        fx, fy, fw, fh = faces[0]
        print(f"  ↳ Raw Haar box:  x={fx} y={fy} w={fw} h={fh}")

    forehead_offset = int(fh * HAAR_FOREHEAD_COMPENSATION)
    fy_corrected    = max(0, fy - forehead_offset)
    fh_corrected    = fh + forehead_offset

    print(f"  ↳ Corrected box: x={fx} y={fy_corrected} w={fw} h={fh_corrected}  "
          f"(forehead +{forehead_offset}px)")

    return (fx, fy_corrected, fw, fh_corrected)


# ─────────────────────────────────────────────────────────────
# STEP 3: MASK GENERATION
# ─────────────────────────────────────────────────────────────

def generate_hair_mask(image_bgr: np.ndarray, face: tuple) -> np.ndarray:
    """Hair region: strictly ABOVE the corrected face bounding box."""
    h_img, w_img = image_bgr.shape[:2]
    fx, fy, fw, fh = face
    mask = np.zeros((h_img, w_img), dtype=np.uint8)

    hair_bottom = fy + int(fh * 0.05)
    hair_left   = max(0, fx - int(fw * 0.40))
    hair_right  = min(w_img, fx + fw + int(fw * 0.40))
    cv2.rectangle(mask, (hair_left, 0), (hair_right, hair_bottom), 255, -1)

    side_bottom = fy + int(fh * 0.55)
    side_width  = int(fw * 0.30)

    left_panel_right = max(0, fx - int(fw * 0.05))
    left_panel_left  = max(0, left_panel_right - side_width)
    cv2.rectangle(mask, (left_panel_left, 0), (left_panel_right, side_bottom), 255, -1)

    right_panel_left  = min(w_img, fx + fw + int(fw * 0.05))
    right_panel_right = min(w_img, right_panel_left + side_width)
    cv2.rectangle(mask, (right_panel_left, 0), (right_panel_right, side_bottom), 255, -1)

    return mask


def generate_beard_mask(image_bgr: np.ndarray, face: tuple) -> np.ndarray:
    """Beard region: from just below the mouth down to just below the chin."""
    h_img, w_img = image_bgr.shape[:2]
    fx, fy, fw, fh = face
    mask = np.zeros((h_img, w_img), dtype=np.uint8)

    beard_top    = fy + int(fh * 0.82)
    beard_bottom = min(h_img, fy + fh + int(fh * 0.20))

    beard_left  = max(0, fx - int(fw * 0.10))
    beard_right = min(w_img, fx + fw + int(fw * 0.10))

    cv2.rectangle(mask, (beard_left, beard_top), (beard_right, beard_bottom), 255, -1)

    return mask


def cut_face_protection_zone(mask: np.ndarray, face: tuple) -> np.ndarray:
    """Hard-erase any mask that leaked into the core face region after dilation/blur."""
    fx, fy, fw, fh = face
    prot_top    = fy + int(fh * 0.05)
    prot_bottom = fy + int(fh * 0.80)
    prot_left   = fx + int(fw * 0.05)
    prot_right  = fx + fw - int(fw * 0.05)
    mask[prot_top:prot_bottom, prot_left:prot_right] = 0
    return mask


def combine_and_refine_mask(
    hair_mask: np.ndarray,
    beard_mask: np.ndarray,
    face: tuple,
    dilate_px: int = 8,
    blur_r: int = 11
) -> np.ndarray:
    """Combine masks, apply conservative smoothing, then hard-cut face zone."""
    combined = cv2.bitwise_or(hair_mask, beard_mask)

    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    combined = cv2.dilate(combined, kernel, iterations=1)

    if blur_r % 2 == 0:
        blur_r += 1
    combined = cv2.GaussianBlur(combined, (blur_r, blur_r), 0)
    _, combined = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)

    combined = cut_face_protection_zone(combined, face)
    return combined


def generate_mask(image_path: str, save_path: str = "mask.png"):
    """Full mask pipeline. Returns (mask_path, image_bgr, face)."""
    print("\n📐 STEP 1: Generating hair & beard mask...")
    image_bgr = load_and_resize(image_path)

    print("  ↳ Detecting face (with forehead compensation)...")
    face = detect_face(image_bgr)

    print("  ↳ Building hair mask (above corrected hairline)...")
    hair_mask = generate_hair_mask(image_bgr, face)

    print("  ↳ Building beard mask (below lips, above shirt)...")
    beard_mask = generate_beard_mask(image_bgr, face)

    print("  ↳ Combining, refining, protecting face zone...")
    final_mask = combine_and_refine_mask(hair_mask, beard_mask, face)

    Image.fromarray(final_mask).save(save_path)
    print(f"  ✅ Mask saved: {save_path}")

    return save_path, image_bgr, face


# ─────────────────────────────────────────────────────────────
# STEP 4: MASK PREVIEW
# ─────────────────────────────────────────────────────────────

def save_mask_preview(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    face: tuple,
    save_path: str
):
    """Side-by-side: Original | Red overlay + diagnostic boxes."""
    image_rgb    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = image_rgb.shape[:2]
    fx, fy, fw, fh = face

    overlay   = image_rgb.copy()
    red_layer = np.zeros_like(image_rgb)
    red_layer[:, :] = [220, 50, 50]
    mask_bool = mask > 127
    overlay[mask_bool] = (0.45 * overlay[mask_bool] + 0.55 * red_layer[mask_bool]).astype(np.uint8)

    # Corrected face box — yellow
    cv2.rectangle(overlay, (fx, fy), (fx + fw, fy + fh), (255, 220, 0), 2)

    # Protection zone — green
    prot_top    = fy + int(fh * 0.05)
    prot_bottom = fy + int(fh * 0.80)
    prot_left   = fx + int(fw * 0.05)
    prot_right  = fx + fw - int(fw * 0.05)
    cv2.rectangle(overlay, (prot_left, prot_top), (prot_right, prot_bottom), (0, 255, 80), 2)

    # Raw Haar top line — cyan (where eyebrows are, before correction)
    raw_fy = fy + int(fh * HAAR_FOREHEAD_COMPENSATION)
    cv2.line(overlay, (fx, raw_fy), (fx + fw, raw_fy), (0, 220, 255), 1)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    orig_copy = image_rgb.copy()
    cv2.putText(orig_copy, "Original",             (10, 28), font, 0.8, (255, 255, 255), 2)
    cv2.putText(overlay,   "RED = Will Change",    (10, 28), font, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay,   "Yellow = Face Box",    (10, 52), font, 0.55, (255, 220, 0),  1)
    cv2.putText(overlay,   "Green  = Protected",   (10, 72), font, 0.55, (0, 255, 80),   1)
    cv2.putText(overlay,   "Cyan   = Raw Haar",    (10, 92), font, 0.55, (0, 220, 255),  1)

    divider   = np.ones((h_img, 5, 3), dtype=np.uint8) * 200
    composite = np.hstack([orig_copy, divider, overlay])
    Image.fromarray(composite).save(save_path)
    print(f"  ✅ Preview saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# STEP 5: FAL AI FLUX INPAINTING
# ─────────────────────────────────────────────────────────────

def call_fal_flux_inpaint(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    prompt: str,
    seed: int = 42,
    output_path: str = "output.png"
) -> str:
    """Send image + mask + prompt to Fal AI FLUX Pro Fill API."""
    if not API_KEY:
        raise ValueError(
            "\n❌ API key not set!\n"
            "   Set API_KEY at the top of this file.\n"
            "   Get one from: https://fal.ai/dashboard/keys"
        )

    print("\n🚀 STEP 3: Calling Fal AI Flux Inpainting API...")
    print(f"  ↳ Prompt:   {prompt}")
    print(f"  ↳ Seed:     {seed}")

    # Convert arrays directly to Base64 Data URIs for the Fal payload
    def to_base64_data_uri(arr):
        pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if len(arr.shape) == 3 else arr)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"

    payload = {
        "prompt": prompt,
        "image_url": to_base64_data_uri(image_bgr),
        "mask_url": to_base64_data_uri(mask),
        "seed": seed,
        "output_format": OUTPUT_FORMAT
    }

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Key {API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload
    )

    if response.status_code == 200:
        data = response.json()
        if "images" in data and len(data["images"]) > 0:
            final_image_url = data["images"][0]["url"]
            print(f"  ↳ Downloading result from Fal...")
            
            img_resp = requests.get(final_image_url)
            if img_resp.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(img_resp.content)
                print(f"  ✅ Output saved: {output_path}")
                return output_path
            else:
                raise RuntimeError(f"Failed to download generated image: {img_resp.status_code}")
        else:
            raise RuntimeError("API succeeded but returned no images.")
    else:
        msg = response.text
        print(f"\n  ❌ API Error {response.status_code}: {msg}")
        if response.status_code == 401:
            print("    → Invalid API key")
        elif response.status_code == 402:
            print("    → Insufficient credits")
        raise RuntimeError(f"API call failed {response.status_code}: {msg}")


# ─────────────────────────────────────────────────────────────
# STEP 6: BEFORE / AFTER COMPARISON
# ─────────────────────────────────────────────────────────────

def save_comparison(original_path: str, output_path: str, save_path: str):
    """Side-by-side BEFORE / AFTER image."""
    orig = np.array(Image.open(original_path).convert("RGB"))
    out  = np.array(Image.open(output_path).convert("RGB"))
    if orig.shape != out.shape:
        out = cv2.resize(out, (orig.shape[1], orig.shape[0]))
    font   = cv2.FONT_HERSHEY_SIMPLEX
    orig_c = orig.copy()
    out_c  = out.copy()
    cv2.putText(orig_c, "BEFORE", (10, 32), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out_c,  "AFTER",  (10, 32), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    divider    = np.ones((orig.shape[0], 6, 3), dtype=np.uint8) * 230
    comparison = np.hstack([orig_c, divider, out_c])
    Image.fromarray(comparison).save(save_path)
    print(f"  ✅ Comparison saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    image_path: str,
    prompt: str,
    preview_only: bool = False,
    seed: int = 42
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    base         = os.path.splitext(image_path)[0]
    mask_path    = f"{base}_mask.png"
    preview_path = f"{base}_mask_preview.png"
    output_path  = f"{base}_modified.png"
    compare_path = f"{base}_comparison.png"

    print(f"\n{'='*62}")
    print(f"  HAIR & BEARD MODIFIER (Fal AI FLUX.1 Pro Fill)")
    print(f"{'='*62}")
    print(f"  Input:    {image_path}")
    print(f"  Prompt:   {prompt}")
    print(f"{'='*62}")

    mask_path_saved, image_bgr, face = generate_mask(image_path, mask_path)
    mask = cv2.imread(mask_path_saved, cv2.IMREAD_GRAYSCALE)

    print("\n👁️  STEP 2: Saving annotated mask preview...")
    save_mask_preview(image_bgr, mask, face, preview_path)

    if preview_only:
        print(f"\n⏸️  Preview-only mode.")
        print(f"   Open '{preview_path}' and verify the mask.")
        print(f"   Rerun without --preview-only when satisfied.")
        return

    call_fal_flux_inpaint(
        image_bgr=image_bgr, mask=mask, prompt=prompt,
        seed=seed, output_path=output_path
    )

    print("\n🖼️  STEP 4: Creating before/after comparison...")
    save_comparison(image_path, output_path, compare_path)

    print(f"\n{'='*62}")
    print(f"  ✅ DONE!  Files generated:")
    print(f"     • Mask:       {mask_path}")
    print(f"     • Preview:    {preview_path}")
    print(f"     • Result:     {output_path}")
    print(f"     • Comparison: {compare_path}")
    print(f"{'='*62}\n")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Modify hair and beard using Fal AI FLUX inpainting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python hair_beard_modifier.py --image photo.jpg --prompt "long red beard, blonde wavy hair"
  python hair_beard_modifier.py --image photo.jpg --preview-only
  python hair_beard_modifier.py --image photo.jpg --prompt "clean shaven, short curly black hair" --seed 99
        """
    )

    parser.add_argument("--image",        required=True,        help="Path to input image (jpg/png)")
    parser.add_argument("--prompt",       default=None,         help="Describe the desired hair/beard")
    parser.add_argument("--preview-only", action="store_true",  help="Generate mask preview only, no API call")
    parser.add_argument("--seed",         type=int,   default=42, help="Seed for reproducibility")

    args = parser.parse_args()

    if args.prompt:
        prompt = args.prompt
    elif args.preview_only:
        prompt = "photorealistic portrait"
    else:
        parser.print_help()
        print("\n❌ Provide --prompt (or --preview-only to check the mask)")
        sys.exit(1)

    run_pipeline(
        image_path=args.image, prompt=prompt,
        preview_only=args.preview_only, seed=args.seed
    )

if __name__ == "__main__":
    main()