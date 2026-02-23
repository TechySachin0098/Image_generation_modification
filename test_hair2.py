

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

TARGET_IMAGE_PATH = "images/MAN2.png"
OUTPUT_IMAGE_PATH = "result_local_v9.png"

# ── MODEL ─────────────────────────────────────────────────────────────────────
# SD 1.5 inpainting — best balance of quality and VRAM for a GTX 1050 (4 GB)
MODEL_ID = "runwayml/stable-diffusion-inpainting"

# ── CUDA / DEVICE ─────────────────────────────────────────────────────────────
FORCE_DEVICE      = None    # None = auto | "cuda" | "mps" | "cpu"
VERBOSE_CUDA      = True
USE_TORCH_COMPILE = False

# ── GENERATION ────────────────────────────────────────────────────────────────
# GTX 1050 (4 GB VRAM) — keep steps moderate and resolution at 512
INPAINT_STEPS      = 60
GUIDANCE_SCALE     = 8.0
STRENGTH           = 0.60  # low = face preserved, high = more creative freedom
INPAINT_RESOLUTION = 512
SEED               = None   # None = random; set an int to reproduce a good result
NUM_VARIATIONS     = 10      # increase to generate multiple outputs and pick best

# ── PADDING ───────────────────────────────────────────────────────────────────
ENABLE_PADDING = True
PAD_TOP        = 0.15
PAD_SIDES      = 0.22
PAD_BOTTOM     = 0.10

# ── MASK PREVIEW ──────────────────────────────────────────────────────────────
SAVE_MASK_PREVIEW = True
MASK_PREVIEW_PATH = "mask_preview.png"
OPEN_MASK_PREVIEW = True   # True = auto-open after saving

# ── COMPOSITING ───────────────────────────────────────────────────────────────
COMPOSITE_MODE = "poisson"   # "poisson" | "alpha"

# ── SECOND PASS ───────────────────────────────────────────────────────────────
RUN_SECOND_PASS = False

# ── PROMPTS ───────────────────────────────────────────────────────────────────
# Kept well under CLIP's 77-token limit
PROMPT = (
    "(long dark brown hair:1.4), (hair past shoulders:1.3), "
    "(hair covering ears:1.2), (same man:1.5), (identical face:1.5), "
    "thick natural hair, photorealistic portrait, studio lighting"
)

NEGATIVE_PROMPT = (
    "(short hair:1.8), (bald sides:1.5), (different person:1.8), "
    "(changed face:1.8), deformed, blurry, cartoon, low quality, watermark"
)

# ══════════════════════════════════════════════════════════════════════════════
#  CUDA UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def get_device() -> str:
    if FORCE_DEVICE:
        if FORCE_DEVICE == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("FORCE_DEVICE='cuda' but CUDA is not available!")
        return FORCE_DEVICE
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_cuda_info():
    if not torch.cuda.is_available():
        print("  ⚠️  CUDA not available — running on CPU/MPS")
        return
    n = torch.cuda.device_count()
    print(f"\n{'─'*52}")
    print(f"  CUDA available  |  {n} GPU(s) detected")
    for i in range(n):
        props     = torch.cuda.get_device_properties(i)
        total     = props.total_memory   / 1024**3
        reserved  = torch.cuda.memory_reserved(i)  / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        free      = total - reserved
        print(f"  GPU {i}: {props.name}")
        print(f"         CUDA version : {torch.version.cuda}")
        print(f"         Total  VRAM  : {total:.1f} GB")
        print(f"         Free   VRAM  : {free:.1f} GB")
        print(f"         Allocated   : {allocated:.2f} GB")
    print(f"{'─'*52}\n")


def log_vram(label: str):
    if VERBOSE_CUDA and torch.cuda.is_available():
        alloc  = torch.cuda.memory_allocated() / 1024**3
        reserv = torch.cuda.memory_reserved()  / 1024**3
        print(f"  [VRAM] {label:<36}  alloc={alloc:.2f} GB  reserved={reserv:.2f} GB")


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE PADDING
# ══════════════════════════════════════════════════════════════════════════════

def pad_image(pil_img: Image.Image):
    """
    Adds canvas so SD can generate hair flowing outside the original frame.
    Background color sampled from image corner.
    Returns: (padded_image, pad_x_offset, pad_y_offset)
    """
    w, h  = pil_img.size
    pad_l = int(w * PAD_SIDES)
    pad_r = int(w * PAD_SIDES)
    pad_t = int(h * PAD_TOP)
    pad_b = int(h * PAD_BOTTOM)
    new_w = w + pad_l + pad_r
    new_h = h + pad_t + pad_b
    bg_color = pil_img.getpixel((5, 5))
    padded   = Image.new("RGB", (new_w, new_h), bg_color)
    padded.paste(pil_img, (pad_l, pad_t))
    print(f"  📐 Padded: {w}×{h} → {new_w}×{new_h}  (left={pad_l}px, top={pad_t}px)")
    return padded, pad_l, pad_t


def unpad_image(padded_img: Image.Image, orig_w: int, orig_h: int,
                pad_x: int, pad_y: int) -> Image.Image:
    return padded_img.crop((pad_x, pad_y, pad_x + orig_w, pad_y + orig_h))


# ══════════════════════════════════════════════════════════════════════════════
#  MASK GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def save_mask_preview(image_bgr: np.ndarray, mask: np.ndarray):
    """Saves red-overlay mask preview as PNG. Works on Windows without display."""
    overlay     = image_bgr.copy()
    overlay[mask > 0] = [0, 0, 255]
    preview     = cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0)
    preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    pil_preview = Image.fromarray(preview_rgb)
    pil_preview.save(MASK_PREVIEW_PATH)
    print(f"  🖼️  Mask preview saved → {MASK_PREVIEW_PATH}")
    if OPEN_MASK_PREVIEW:
        pil_preview.show()


def create_organic_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Builds inpainting mask from MediaPipe face landmarks.
    255 = inpaint this region | 0 = keep original pixel.

    Strategy:
    - Hair region: top of head + side columns (pulled inward from ears)
    - Beard region: along jawline
    - Inner face oval: PROTECTED (excluded from mask)
    - Chin/lower jaw: re-added for beard generation
    """
    h, w    = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Download MediaPipe model on first run
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        print("  Downloading MediaPipe face landmarker model…")
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        urllib.request.urlretrieve(url, model_path)
        print("  ✅ Model downloaded.")

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    with vision.FaceLandmarker.create_from_options(options) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result   = detector.detect(mp_image)

    if not result.face_landmarks:
        print("  ⚠️  No face detected — returning empty mask.")
        return np.zeros((h, w), dtype=np.uint8)

    lm = result.face_landmarks[0]
    def px(idx): return (int(lm[idx].x * w), int(lm[idx].y * h))

    JAWLINE     = [234,93,132,58,172,136,150,149,176,148,152,
                   377,400,378,379,365,397,288,323,361,454]
    HAIRLINE    = [103,67,109,10,338,297,332]
    FACE_OVAL   = [10,338,297,332,284,251,389,356,454,323,361,288,397,
                   365,379,378,400,377,152,148,176,149,150,136,172,58,
                   132,93,234,127,162,21,54,103,67,109]
    MOUTH_LOWER = [61,146,91,181,84,17,314,405,321,375,291]

    jaw_pts   = np.array([px(i) for i in JAWLINE],     dtype=np.int32)
    hair_pts  = np.array([px(i) for i in HAIRLINE],    dtype=np.int32)
    oval_pts  = np.array([px(i) for i in FACE_OVAL],   dtype=np.int32)
    mouth_pts = np.array([px(i) for i in MOUTH_LOWER], dtype=np.int32)

    left_ear  = px(234)
    right_ear = px(454)
    top_head  = px(10)

    mask = np.zeros((h, w), dtype=np.uint8)

    # ── HAIR ──────────────────────────────────────────────────────────────────
    # 1. Hairline polyline (covers crown)
    cv2.polylines(mask, [hair_pts], isClosed=False, color=255,
                  thickness=int(w * 0.55))
    # 2. Top-of-head rectangle
    cv2.rectangle(mask, (0, 0),
                  (w, top_head[1] + int(h * 0.12)), 255, -1)
    # 3. Left side column — pulled INWARD from ear to avoid face bleed
    cv2.rectangle(mask,
                  (0, top_head[1] - int(h * 0.05)),
                  (left_ear[0] - int(w * 0.03), h), 255, -1)
    # 4. Right side column — mirror
    cv2.rectangle(mask,
                  (right_ear[0] + int(w * 0.03), top_head[1] - int(h * 0.05)),
                  (w, h), 255, -1)

    # ── BEARD ─────────────────────────────────────────────────────────────────
    cv2.polylines(mask, [jaw_pts], isClosed=False, color=255,
                  thickness=int(w * 0.18))

    # ── PROTECT INNER FACE ────────────────────────────────────────────────────
    face_safe = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(face_safe, [oval_pts], 255)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(face_safe))

    # ── RE-ADD CHIN FOR BEARD ─────────────────────────────────────────────────
    mouth_y   = min(p[1] for p in mouth_pts)
    lower_box = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(lower_box, (0, mouth_y), (w, h), 255, -1)
    chin_area = cv2.bitwise_and(face_safe, lower_box)
    mask      = cv2.bitwise_or(mask, chin_area)

    # ── FEATHER EDGES ─────────────────────────────────────────────────────────
    mask = cv2.GaussianBlur(mask, (81, 81), 0)
    _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)

    if SAVE_MASK_PREVIEW:
        save_mask_preview(image_bgr, mask)

    return mask


# ══════════════════════════════════════════════════════════════════════════════
#  COMPOSITING
# ══════════════════════════════════════════════════════════════════════════════

def composite_result(result_np: np.ndarray,
                     orig_np: np.ndarray,
                     mask_np: np.ndarray) -> Image.Image:
    """
    Blends inpainting result onto original.
    Poisson (seamlessClone) = no visible seams.
    Falls back to PIL alpha composite if seamlessClone fails.
    """
    if COMPOSITE_MODE == "poisson":
        mask_hard = (mask_np > 128).astype(np.uint8) * 255
        kernel    = np.ones((5, 5), np.uint8)
        mask_hard = cv2.erode(mask_hard, kernel, iterations=2)
        center    = (orig_np.shape[1] // 2, orig_np.shape[0] // 2)
        try:
            blended = cv2.seamlessClone(
                result_np, orig_np, mask_hard, center, cv2.NORMAL_CLONE
            )
            print("  ✅ Poisson seamlessClone compositing applied")
            return Image.fromarray(blended)
        except cv2.error as e:
            print(f"  ⚠️  seamlessClone failed ({e}) — using alpha fallback")

    result_pil = Image.fromarray(result_np)
    orig_pil   = Image.fromarray(orig_np)
    mask_pil   = Image.fromarray(mask_np).convert("L")
    print("  ℹ️  Alpha composite mode")
    return Image.composite(result_pil, orig_pil, mask_pil)


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_pipeline(model_id: str):
    """
    Loads SD 1.5 inpainting pipeline.
    IP-Adapter removed entirely — causes attention_processor tuple error
    with this diffusers version. Face identity preserved via mask tightness
    and STRENGTH=0.60 instead.

    VRAM notes for GTX 1050 (4 GB):
    - Model loads at ~3.3 GB fp16
    - Attention slicing enabled automatically
    - VAE tiling via pipe.vae.enable_tiling() (not deprecated form)
    - xformers not installed — skipped
    """
    device = get_device()
    dtype  = torch.float16 if device in ("cuda", "mps") else torch.float32

    print(f"\n  Loading model  : {model_id}")
    print(f"  Device         : {device}  |  dtype: {dtype}")
    print("  (First run downloads ~4 GB — cached afterwards)\n")

    # No variant= param, no IP-Adapter
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # ── NVIDIA CUDA ───────────────────────────────────────────────────────────
    if device == "cuda":
        if VERBOSE_CUDA:
            print_cuda_info()

        pipe = pipe.to("cuda")
        log_vram("after .to('cuda')")

        # Always enable attention slicing on 4 GB GPU
        pipe.enable_attention_slicing(1)
        print("  ✅ Attention slicing enabled (slice_size=1, max VRAM savings)")

        # VAE tiling — use non-deprecated form
        try:
            pipe.vae.enable_tiling()
            print("  ✅ VAE tiling enabled")
        except Exception as e:
            print(f"  ⚠️  VAE tiling unavailable: {e}")

        # Sequential CPU offload as extra safety net for 4 GB VRAM
        # Uncomment the line below if you still get CUDA OOM errors:
        # pipe.enable_sequential_cpu_offload()

        if USE_TORCH_COMPILE:
            try:
                pipe.unet = torch.compile(
                    pipe.unet, mode="reduce-overhead", fullgraph=True
                )
                print("  ✅ torch.compile enabled on UNet")
            except Exception as e:
                print(f"  ⚠️  torch.compile failed: {e}")

        log_vram("after optimisations")

    elif device == "mps":
        pipe = pipe.to("mps")
        pipe.enable_attention_slicing()
        print("  ✅ MPS (Apple Silicon) configured")

    else:
        print("  ⚠️  Running on CPU — expect 5–15 min per image")
        pipe.enable_sequential_cpu_offload()

    return pipe, device


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIONAL SECOND-PASS REFINEMENT
# ══════════════════════════════════════════════════════════════════════════════

def run_second_pass(pipe, device, pass1_result: Image.Image,
                    mask_pil: Image.Image, w: int, h: int) -> Image.Image:
    """
    Gentle refinement pass on the hair-face boundary.
    Uses a tighter (eroded) mask and low strength.
    """
    print("\n  Running refinement pass 2…")
    mask_np    = np.array(mask_pil)
    kernel     = np.ones((15, 15), np.uint8)
    tight_mask = cv2.erode(mask_np, kernel, iterations=3)
    tight_pil  = Image.fromarray(tight_mask).convert("L")

    generator = None
    if SEED is not None:
        generator = torch.Generator(device=device).manual_seed(SEED + 99)

    with torch.inference_mode():
        result2 = pipe(
            prompt              = PROMPT,
            negative_prompt     = NEGATIVE_PROMPT,
            image               = pass1_result.resize((w, h), Image.LANCZOS),
            mask_image          = tight_pil.resize((w, h), Image.NEAREST),
            num_inference_steps = 30,
            guidance_scale      = GUIDANCE_SCALE,
            strength            = 0.35,
            height              = h,
            width               = w,
            generator           = generator,
        ).images[0]

    print("  ✅ Pass 2 complete")
    return result2


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_local():
    print("─── Hair & Beard Inpainting v6 (SD 1.5, GTX 1050 optimised) ───\n")

    if VERBOSE_CUDA:
        print_cuda_info()

    # ── 1. Load original ──────────────────────────────────────────────────────
    orig_pil = Image.open(TARGET_IMAGE_PATH).convert("RGB")
    orig_w, orig_h = orig_pil.size
    print(f"  📷 Original size: {orig_w}×{orig_h}")

    # ── 2. Pad ────────────────────────────────────────────────────────────────
    pad_x, pad_y = 0, 0
    if ENABLE_PADDING:
        orig_pil, pad_x, pad_y = pad_image(orig_pil)

    # ── 3. PIL → BGR for MediaPipe ────────────────────────────────────────────
    img_cv = cv2.cvtColor(np.array(orig_pil), cv2.COLOR_RGB2BGR)

    # ── 4. Build mask ─────────────────────────────────────────────────────────
    print("\n  Building face mask…")
    mask_np  = create_organic_mask(img_cv)
    mask_pil = Image.fromarray(mask_np).convert("L")
    print("  ✅ Mask created")

    # ── 5. Resize to inference resolution ─────────────────────────────────────
    padded_w, padded_h = orig_pil.size
    scale = INPAINT_RESOLUTION / max(padded_w, padded_h)
    new_w = int(padded_w * scale) // 8 * 8
    new_h = int(padded_h * scale) // 8 * 8

    orig_resized = orig_pil.resize((new_w, new_h), Image.LANCZOS)
    mask_resized = mask_pil.resize((new_w, new_h), Image.NEAREST)
    print(f"  🔄 Inference resolution: {new_w}×{new_h}")

    # ── 6. Load pipeline ──────────────────────────────────────────────────────
    pipe, device = load_pipeline(MODEL_ID)

    # ── 7. Generate ───────────────────────────────────────────────────────────
    print(f"\n  Running {NUM_VARIATIONS} inpainting variation(s)…")
    log_vram("before inference")

    results = []
    for i in range(NUM_VARIATIONS):
        seed_i    = SEED + i if SEED is not None else None
        generator = (torch.Generator(device=device).manual_seed(seed_i)
                     if seed_i is not None else None)

        print(f"  Variation {i+1}/{NUM_VARIATIONS}  (seed={seed_i})")

        with torch.inference_mode():
            result = pipe(
                prompt              = PROMPT,
                negative_prompt     = NEGATIVE_PROMPT,
                image               = orig_resized,
                mask_image          = mask_resized,
                num_inference_steps = INPAINT_STEPS,
                guidance_scale      = GUIDANCE_SCALE,
                strength            = STRENGTH,
                height              = new_h,
                width               = new_w,
                generator           = generator,
            ).images[0]

        if RUN_SECOND_PASS:
            result = run_second_pass(pipe, device, result, mask_pil, new_w, new_h)

        results.append(result)

    log_vram("after inference")

    # ── 8. Free VRAM ──────────────────────────────────────────────────────────
    if device == "cuda":
        del pipe
        torch.cuda.empty_cache()
        log_vram("after cache clear")

    # ── 9. Composite and save ─────────────────────────────────────────────────
    print()
    for i, result in enumerate(results):
        result_full = result.resize((padded_w, padded_h), Image.LANCZOS)
        result_np   = np.array(result_full)
        orig_np     = np.array(orig_pil)
        mask_full   = np.array(mask_pil.resize((padded_w, padded_h), Image.LANCZOS))

        composited = composite_result(result_np, orig_np, mask_full)

        if ENABLE_PADDING:
            final = unpad_image(composited, orig_w, orig_h, pad_x, pad_y)
            print(f"  ✂️  Cropped to original: {orig_w}×{orig_h}")
        else:
            final = composited

        if NUM_VARIATIONS > 1:
            base, ext = os.path.splitext(OUTPUT_IMAGE_PATH)
            out_path  = f"{base}_v{i+1}{ext}"
        else:
            out_path = OUTPUT_IMAGE_PATH

        final.save(out_path)
        print(f"  💾 Saved → {out_path}")

    print("\n✅ Done.")
    if NUM_VARIATIONS == 1:
        Image.open(OUTPUT_IMAGE_PATH).show()
    else:
        print(f"   {NUM_VARIATIONS} variations saved — open them to pick the best.")


if __name__ == "__main__":
    run_local()