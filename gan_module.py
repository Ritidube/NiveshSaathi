"""
gan_module.py — Stable Diffusion GAN Visual Generator
Generates "your savings as a growing plant" visuals per profession.
Pre-generate on Google Colab T4, then serve from cache on laptop.
"""

import os
import io
import base64
import threading
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Device setup ──────────────────────────────────────────────────────────────
_pipe      = None
_pipe_lock = threading.Lock()
_device    = "cuda" if torch.cuda.is_available() else "cpu"
_using_gpu = _device == "cuda"
print(f"[GAN] Device: {_device} | GPU: {_using_gpu}")

# ── Growth stage helper ───────────────────────────────────────────────────────
def get_growth_stage(months: int) -> str:
    if months <= 3:  return "seedling"
    if months <= 6:  return "sapling"
    if months <= 12: return "young_tree"
    return "full_tree"

# ── Prompts ───────────────────────────────────────────────────────────────────
GROWTH_PROMPTS = {
    "teacher": {
        "seedling":   "A tiny green seedling sprouting from a single golden rupee coin on a wooden school desk, soft sunlight through classroom window, chalk dust in air, warm watercolor illustration, hopeful mood, Indian school setting, vibrant colors",
        "sapling":    "A young sapling with bright green leaves growing from a stack of rupee coins, classroom chalkboard background with math equations, sunlight streaming in, watercolor art style, optimistic, Indian teacher desk",
        "young_tree": "A lush young tree growing tall inside a sunlit Indian classroom, golden coins hanging as fruit from branches, children drawings on walls, vivid watercolor style, thriving, joyful atmosphere",
        "full_tree":  "A magnificent full-grown banyan tree bursting through a school roof, golden rupee coins as leaves glittering in sunlight, proud teacher standing below, Indian folk art style, abundant and majestic",
    },
    "daily_wage": {
        "seedling":   "A tiny green seedling growing from a worker calloused palm holding a single rupee coin, construction site background at sunrise, warm golden light, folk art watercolor, hopeful and dignified, Indian labor setting",
        "sapling":    "A sapling with strong roots growing from worker tools hammer and wrench, small coins as leaves, sunrise over buildings under construction, vibrant watercolor illustration, empowering mood",
        "young_tree": "A strong young neem tree growing from a hardhat and work boots, rupee coins hanging as bright fruits, blue sky with scaffolding, colorful folk art style, pride and growth",
        "full_tree":  "A towering peepal tree rooted in tools and coins, full of golden rupee fruit, Indian daily wage worker family sheltering under it, sunrise background, vibrant folk art illustration, prosperity and dignity",
    },
    "shopkeeper": {
        "seedling":   "A tiny seedling sprouting from the counter of a small Indian kirana shop, single rupee coin at roots, colorful packaging on shelves behind, warm shop lighting, watercolor style, cozy and hopeful",
        "sapling":    "A small flowering plant growing through a shop counter, small coins as blossoms, colorful goods dal rice oil on shelves, Indian kirana store atmosphere, warm watercolor illustration",
        "young_tree": "A thriving young tree growing inside and out of a small kirana shop, rupee coins as fruit on every branch, happy shopkeeper tending it, vibrant Indian market style, abundant goods",
        "full_tree":  "A grand banyan tree with golden rupee leaves growing from a prosperous kirana shop, community of customers around it, Indian bazaar setting, rich folk art illustration, wealth and community",
    },
    "default": {
        "seedling":   "A tiny green seedling growing from a single shining rupee coin in fertile Indian soil, golden morning light, dewdrops, watercolor art, hopeful beginning",
        "sapling":    "A young sapling with bright leaves growing from a pile of rupee coins, Indian village background, morning mist, vibrant watercolor",
        "young_tree": "A healthy young tree growing from coins in soil, golden fruit hanging, Indian countryside, warm sunlight, colorful folk art style",
        "full_tree":  "A majestic banyan tree with golden rupee coins as leaves, family sitting under it, Indian village setting, abundant harvest, vibrant folk art illustration",
    },
}

NEGATIVE_PROMPT = "ugly, blurry, low quality, distorted, dark, scary, western style, sad, gloomy, text overlay"

# ── Pipeline loader ───────────────────────────────────────────────────────────
def load_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe

    # Skip if all 16 images already cached
    cache_dir = Path("static/gan_cache")
    if cache_dir.exists() and len(list(cache_dir.glob("*.png"))) >= 16:
        print("[GAN] All 16 cached images found — skipping SD model load ✅")
        return None

    with _pipe_lock:
        if _pipe is not None:
            return _pipe
        try:
            from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
            print("[GAN] Loading Stable Diffusion v1.5...")
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if _using_gpu else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config
            )
            pipe = pipe.to(_device)
            if _using_gpu:
                pipe.enable_attention_slicing()
            _pipe = pipe
            print("[GAN] Pipeline loaded ✅")
            return _pipe
        except Exception as e:
            print(f"[GAN] Could not load SD: {e}")
            return None

# ── Main generation function ──────────────────────────────────────────────────
def generate_growth_image(
    profession: str = "default",
    months: int = 6,
    savings_amount: float = 500.0,
) -> dict:
    stage   = get_growth_stage(months)
    prompts = GROWTH_PROMPTS.get(profession, GROWTH_PROMPTS["default"])
    prompt  = prompts.get(stage, prompts["seedling"])

    # Try live SD generation
    pipe = load_pipeline()
    if pipe is not None:
        try:
            print(f"[GAN] Generating: {profession}/{stage}")
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    num_inference_steps=20 if _using_gpu else 8,
                    guidance_scale=7.5,
                    width=512, height=512,
                )
            image = result.images[0]
            b64   = _to_base64(image)
            return {
                "success": True, "method": "stable_diffusion",
                "stage": stage, "image_base64": b64,
                "metaphor": _metaphor(stage, savings_amount, months),
            }
        except Exception as e:
            print(f"[GAN] SD failed: {e}")

    # PIL fallback
    image = _draw_fallback(profession, stage, savings_amount, months)
    return {
        "success": True, "method": "pil_fallback",
        "stage": stage, "image_base64": _to_base64(image),
        "metaphor": _metaphor(stage, savings_amount, months),
    }

def _metaphor(stage, monthly, months):
    total = round(monthly * months)
    names = {"seedling":"एक छोटा बीज 🌱","sapling":"एक उगता पौधा 🌿",
             "young_tree":"एक युवा पेड़ 🌳","full_tree":"एक विशाल पेड़ 🌲"}
    return f"₹{monthly:.0f}/month × {months} months = ₹{total:,} — {names.get(stage,'🌱')}"

def _to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def _draw_fallback(profession, stage, monthly, months) -> Image.Image:
    W, H = 512, 512
    img  = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)

    # Sky gradient
    for i in range(H):
        r = int(135 + (200-135)*i/H)
        g = int(206 + (230-206)*i/H)
        b = int(235 + (200-235)*i/H)
        draw.line([(0,i),(W,i)], fill=(r,g,b))

    # Sun
    draw.ellipse([380,30,460,110], fill=(255,220,50))

    # Ground
    draw.ellipse([50,400,462,470], fill=(101,67,33))
    draw.ellipse([55,395,457,455], fill=(56,142,60))

    params = {
        "seedling":   {"trunk_h":40,  "trunk_w":8,  "canopy_r":0,   "coins":0},
        "sapling":    {"trunk_h":90,  "trunk_w":14, "canopy_r":55,  "coins":5},
        "young_tree": {"trunk_h":150, "trunk_w":22, "canopy_r":90,  "coins":12},
        "full_tree":  {"trunk_h":210, "trunk_w":32, "canopy_r":130, "coins":20},
    }
    p  = params.get(stage, params["seedling"])
    cx, base_y = 256, 420

    # Trunk
    draw.rounded_rectangle(
        [cx-p["trunk_w"]//2, base_y-p["trunk_h"], cx+p["trunk_w"]//2, base_y],
        radius=6, fill=(109,76,65)
    )

    # Canopy
    if p["canopy_r"] > 0:
        cy     = base_y - p["trunk_h"] - p["canopy_r"]//2
        greens = [(56,142,60),(67,160,71),(76,175,80),(129,199,132)]
        for i, col in enumerate(greens[:3]):
            r = p["canopy_r"] - i*18
            if r > 10:
                draw.ellipse(
                    [cx-r, cy-r//2+i*14, cx+r, cy+r//2+i*14+20],
                    fill=col
                )

    # Coins
    for i in range(p["coins"]):
        import math
        angle  = (i / max(p["coins"],1)) * math.pi * 2
        spread = p["canopy_r"] * 0.65
        cy_c   = base_y - p["trunk_h"] - p["canopy_r"]//2
        cx2    = int(cx + spread * math.cos(angle))
        cy2    = int(cy_c + spread * 0.5 * math.sin(angle))
        draw.ellipse([cx2-9,cy2-9,cx2+9,cy2+9], fill=(255,214,0))
        draw.ellipse([cx2-9,cy2-9,cx2+9,cy2+9], outline=(230,160,0), width=2)

    # Caption
    total = round(monthly * months)
    draw.rounded_rectangle([30,450,W-30,495], radius=10, fill=(255,255,255))
    try:
        font = ImageFont.truetype(
            "C:/Windows/Fonts/Arial.ttf", 18
        )
    except Exception:
        font = ImageFont.load_default()
    draw.text(
        (W//2, 472),
        f"Rs.{monthly:.0f}/mo x {months}mo = Rs.{total:,}",
        fill=(26,124,79), font=font, anchor="mm"
    )
    return img

# ── Pre-generate all 16 images (run on Colab T4) ──────────────────────────────
def pregenerate_all(output_dir: str = "static/gan_cache"):
    os.makedirs(output_dir, exist_ok=True)
    professions   = list(GROWTH_PROMPTS.keys())
    stages_months = [("seedling",2),("sapling",5),("young_tree",10),("full_tree",24)]

    for prof in professions:
        for stage, months in stages_months:
            print(f"Generating {prof}/{stage}...")
            result = generate_growth_image(prof, months, 500)
            fname  = f"{output_dir}/{prof}_{stage}.png"
            with open(fname, "wb") as f:
                f.write(base64.b64decode(result["image_base64"]))
            print(f"  Saved → {fname}")

    print(f"\n✅ All 16 images saved to {output_dir}/")

if __name__ == "__main__":
    pregenerate_all()
