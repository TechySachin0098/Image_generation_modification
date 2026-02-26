import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import uuid
import traceback
import urllib.request

import cv2
import numpy as np
import requests
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS

# ─────────────────────────────────────────────────────────────
# CONFIG  (edit API_KEY here or pass it via env var STABILITY_API_KEY)
# ─────────────────────────────────────────────────────────────

API_KEY = os.environ.get("STABILITY_API_KEY", "sk-fRlpzVaxT5oJp51Ai5xKdULKQreVG4O4regnjV5yXvFkcBle")
API_URL = "https://api.stability.ai/v2beta/stable-image/edit/inpaint"

INPAINT_STRENGTH          = 0.70
OUTPUT_FORMAT             = "png"
HAAR_FOREHEAD_COMPENSATION = 0.30

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "static", "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024   # 20 MB upload limit

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────────────────────
# HAAR CASCADE DOWNLOADER
# ─────────────────────────────────────────────────────────────

CASCADE_PATH = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")


def ensure_cascade() -> str:
    if os.path.exists(CASCADE_PATH):
        return CASCADE_PATH
    url = ("https://raw.githubusercontent.com/opencv/opencv/master/data/"
           "haarcascades/haarcascade_frontalface_default.xml")
    print("  [*] Downloading Haar cascade...")
    urllib.request.urlretrieve(url, CASCADE_PATH)
    print("  [OK] Cascade downloaded.")
    return CASCADE_PATH


# ─────────────────────────────────────────────────────────────
# IMAGE PIPELINE  (ported from test_hair(working).py)
# ─────────────────────────────────────────────────────────────

def load_and_resize(image_path: str, max_size: int = 1024) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image


def detect_face(image_bgr: np.ndarray) -> tuple:
    face_cascade = cv2.CascadeClassifier(ensure_cascade())
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60))

    h_img, w_img = image_bgr.shape[:2]
    if len(faces) == 0:
        fx, fy, fw, fh = int(w_img * 0.20), int(h_img * 0.10), int(w_img * 0.60), int(h_img * 0.65)
    else:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        fx, fy, fw, fh = faces[0]

    forehead_offset = int(fh * HAAR_FOREHEAD_COMPENSATION)
    fy_corrected    = max(0, fy - forehead_offset)
    fh_corrected    = fh + forehead_offset
    return (fx, fy_corrected, fw, fh_corrected)


def generate_hair_mask(image_bgr: np.ndarray, face: tuple) -> np.ndarray:
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
    h_img, w_img = image_bgr.shape[:2]
    fx, fy, fw, fh = face
    mask = np.zeros((h_img, w_img), dtype=np.uint8)

    beard_top    = fy + int(fh * 0.82)
    beard_bottom = min(h_img, fy + fh + int(fh * 0.20))
    beard_left   = max(0, fx - int(fw * 0.10))
    beard_right  = min(w_img, fx + fw + int(fw * 0.10))

    cv2.rectangle(mask, (beard_left, beard_top), (beard_right, beard_bottom), 255, -1)
    return mask


def cut_face_protection_zone(mask: np.ndarray, face: tuple) -> np.ndarray:
    fx, fy, fw, fh = face
    prot_top    = fy + int(fh * 0.05)
    prot_bottom = fy + int(fh * 0.80)
    prot_left   = fx + int(fw * 0.05)
    prot_right  = fx + fw - int(fw * 0.05)
    mask[prot_top:prot_bottom, prot_left:prot_right] = 0
    return mask


def combine_and_refine_mask(hair_mask, beard_mask, face, dilate_px=8, blur_r=11):
    combined = cv2.bitwise_or(hair_mask, beard_mask)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    combined = cv2.dilate(combined, kernel, iterations=1)
    if blur_r % 2 == 0:
        blur_r += 1
    combined = cv2.GaussianBlur(combined, (blur_r, blur_r), 0)
    _, combined = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)
    combined = cut_face_protection_zone(combined, face)
    return combined


def build_mask(image_bgr: np.ndarray):
    face       = detect_face(image_bgr)
    hair_mask  = generate_hair_mask(image_bgr, face)
    beard_mask = generate_beard_mask(image_bgr, face)
    final_mask = combine_and_refine_mask(hair_mask, beard_mask, face)
    return final_mask, face


def to_png_bytes(arr: np.ndarray) -> bytes:
    pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if len(arr.shape) == 3 else arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def call_stability_inpaint(image_bgr, mask, prompt, negative_prompt="",
                            strength=INPAINT_STRENGTH, seed=42):
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
            "grow_mask":       "0",
        }
    )

    if response.status_code == 200:
        return response.content
    else:
        msg = response.text
        raise RuntimeError(f"API error {response.status_code}: {msg}")


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

PROMPTS_TXT = os.path.join(os.path.dirname(__file__), "prompts", "prompts.txt")


def parse_prompts_file() -> dict:
    """
    Parse prompts.txt into a dict:
      { "img1": {"prompt": "...", "negative": "..."}, ... }
    """
    result = {}
    if not os.path.exists(PROMPTS_TXT):
        return result

    with open(PROMPTS_TXT, "r", encoding="utf-8") as f:
        content = f.read()

    # Split blocks by blank lines between entries
    import re
    # Each block starts with an img identifier line
    blocks = re.split(r'\n(?=img\d+\s*[\r\n])', content.strip())

    for block in blocks:
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        if not lines:
            continue

        # First line is the image key e.g. "img1"
        key = lines[0].strip().lower()

        prompt_text   = ""
        negative_text = ""

        for line in lines[1:]:
            if line.lower().startswith("prompt:"):
                prompt_text = line[len("Prompt:"):].strip().strip('"')
            elif line.lower().startswith("negative:"):
                negative_text = line[len("Negative:"):].strip().strip('"')

        result[key] = {
            "prompt":   prompt_text,
            "negative": negative_text,
            "image_url": f"/static/prompts/{key}.png"
        }

    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/prompts", methods=["GET"])
def get_prompts():
    """Return parsed prompt gallery as JSON list."""
    data = parse_prompts_file()
    # Return as sorted list so frontend gets them in order
    gallery = [
        {"key": k, **v}
        for k, v in sorted(data.items())
    ]
    return jsonify(gallery)


@app.route("/api/process", methods=["POST"])
def process():
    """
    Accepts multipart form data:
      - image   : the uploaded image file
      - prompt  : text description of desired hair/beard style
      - negative: (optional) negative prompt
      - strength: (optional) float 0.0–1.0, default 0.70
      - seed    : (optional) int, default 42
    Returns JSON: { output_url, comparison_url }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid or missing image file"}), 400

    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    negative = request.form.get("negative", "")
    strength = float(request.form.get("strength", INPAINT_STRENGTH))
    import random
    seed_val = request.form.get("seed")
    seed = int(seed_val) if seed_val else random.randint(0, 1000000)

    # Save upload
    uid       = uuid.uuid4().hex
    ext       = file.filename.rsplit(".", 1)[1].lower()
    in_path   = os.path.join(UPLOAD_FOLDER, f"{uid}_input.{ext}")
    out_path  = os.path.join(OUTPUT_FOLDER, f"{uid}_output.png")
    comp_path = os.path.join(OUTPUT_FOLDER, f"{uid}_comparison.png")

    file.save(in_path)

    try:
        image_bgr = load_and_resize(in_path)
        mask, _ = build_mask(image_bgr)
        result_bytes = call_stability_inpaint(
            image_bgr, mask, prompt,
            negative_prompt=negative,
            strength=strength,
            seed=seed
        )

        # Save output image
        with open(out_path, "wb") as f:
            f.write(result_bytes)

        # Build comparison (side-by-side)
        orig = np.array(Image.open(in_path).convert("RGB"))
        out  = np.array(Image.open(out_path).convert("RGB"))
        if orig.shape != out.shape:
            out = cv2.resize(out, (orig.shape[1], orig.shape[0]))

        font = cv2.FONT_HERSHEY_SIMPLEX
        orig_c, out_c = orig.copy(), out.copy()
        cv2.putText(orig_c, "BEFORE", (10, 32), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out_c,  "AFTER",  (10, 32), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        divider    = np.ones((orig.shape[0], 6, 3), dtype=np.uint8) * 230
        comparison = np.hstack([orig_c, divider, out_c])
        Image.fromarray(comparison).save(comp_path)

        return jsonify({
            "output_url":     f"/static/outputs/{uid}_output.png",
            "comparison_url": f"/static/outputs/{uid}_comparison.png",
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    ensure_cascade()   # download cascade on startup if needed
    app.run(debug=True, port=5000)
