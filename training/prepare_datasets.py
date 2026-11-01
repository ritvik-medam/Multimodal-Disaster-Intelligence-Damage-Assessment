"""
Dataset Preparation Pipeline for Multimodal Disaster Intelligence System
Generates, downloads, and structures datasets for:
1. Disaster Image Classification (flood, earthquake, wildfire, landslide, cyclone_storm, not_disaster)
2. Disaster Text Classification (flood, earthquake, wildfire, landslide, cyclone_storm, not_disaster)
3. Satellite Building Damage Assessment (no_damage, minor_damage, major_damage, destroyed)
4. Multimodal Fusion (paired image + text disaster incidents)

Creates train / val / test splits with full reproducibility and metadata.
"""

import os
import json
import random
import csv
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from training.common import set_seed

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "sample_reports"

DISASTER_CLASSES = [
    "flood",
    "earthquake",
    "wildfire",
    "landslide",
    "cyclone_storm",
    "not_disaster",
]

DAMAGE_CLASSES = [
    "no_damage",
    "minor_damage",
    "major_damage",
    "destroyed",
]

# Realistic disaster report templates for text dataset
TEXT_TEMPLATES = {
    "flood": [
        "Severe flash flooding reported in {loc}. River breached embankments, water levels reaching {depth} feet in residential streets.",
        "Emergency rescue teams deployed in {loc} as heavy rainfall submerged roads and lower floors of hospitals.",
        "People stranded on rooftops in {loc} following torrential monsoon rains and overflowing drainage canals.",
        "Water supply disrupted and bridges impassable due to raging flood waters across {loc} district.",
        "Residents urgently need inflatable rafts, drinking water, and dry rations in {loc} flood zone.",
        "Flash flood warning issued for {loc}. Evacuate low-lying areas immediately as dam floodgates opened.",
    ],
    "earthquake": [
        "Magnitude {mag} earthquake struck near {loc}. Multiple building collapses and cracked roadways reported.",
        "Violent tremors felt across {loc}. Search and rescue crews are excavating debris for trapped survivors.",
        "Severe structural damage observed in {loc} city center. Bridges and overpasses undergoing emergency inspections.",
        "Aftershocks continue to rock {loc}. Residents advised to remain outdoors in designated open parks.",
        "Hospital in {loc} overwhelmed with trauma patients following catastrophic earthquake collapse.",
        "Several multi-story apartment buildings pancaked in {loc} following early morning seismic event.",
    ],
    "wildfire": [
        "Rapidly spreading wildfire in {loc} has consumed over {acres} acres of dry forest and threatens residential perimeters.",
        "Mandatory evacuation orders enforced for all neighborhoods north of {loc} due to intense wind-driven bushfire.",
        "Dense orange smoke and zero visibility in {loc}. Firefighters working around the clock with aerial retardant drops.",
        "Multiple homes destroyed by raging wildfire advancing along the ridge near {loc}.",
        "High wind gusts gusting up to 50mph are pushing the blaze toward power transmission lines in {loc}.",
        "Air quality index has reached hazardous levels in {loc} as uncontrollable blaze engulfs timberline.",
    ],
    "landslide": [
        "Massive mudslide triggered by heavy rain blocked the main highway connecting {loc}. Vehicles trapped beneath silt.",
        "Hillside collapsed into residential valley in {loc}, burying several houses under mud and rock debris.",
        "Search operations underway in {loc} after geotechnical slope failure following three days of unrelenting precipitation.",
        "Displaced boulders and fallen trees sever all road transit into {loc}. Emergency airlift requested.",
        "Ground instability warning issued for mountainous slopes surrounding {loc} after major slope shift.",
        "Mud and slush rushed into village outskirts near {loc}, cutting off power and telecommunications.",
    ],
    "cyclone_storm": [
        "Category {cat} cyclone made landfall near {loc} with sustained winds exceeding {wind} km/h and devastating storm surge.",
        "Violent squalls and hurricane-force winds ripped roofs off commercial warehouses and uprooted giant trees in {loc}.",
        "Storm surge inundating coastal communities along {loc}. Sea wall breached and power grid completely shut down.",
        "Tropical storm {storm_name} battering {loc} coast with extreme precipitation and blinding gale-force wind gusts.",
        "Downed power lines, shattered glass, and flying debris create extreme hazard across {loc} metropolitan area.",
        "Coastal evacuation centers in {loc} are running on emergency backup generators amid cyclone impact.",
    ],
    "not_disaster": [
        "Routine municipal maintenance completed on water pipelines across {loc} central avenue. Normal traffic resumed.",
        "Sunny and calm weather forecast for {loc} throughout the weekend with temperatures hovering around {temp}C.",
        "City council of {loc} approved new urban park beautification and community garden project.",
        "Local farmers market in {loc} experienced record turnout for organic produce and artisan goods.",
        "Public transit schedule in {loc} running on standard weekday timetable with minor routine platform repairs.",
        "Clear skies and mild breeze reported across {loc} bay area today.",
    ],
}

LOCATIONS = [
    "Riverside", "Valley View", "Coastal Bay", "Highland Ridge", "North District",
    "South Point", "West Valley", "East Harbor", "Central Metro", "Mountain View",
    "Pine Crest", "Silver Creek", "Maple Grove", "Lakeview", "Clearwater"
]


def _synthesize_disaster_image(category: str, img_size: int = 224, seed: int = 0) -> Image.Image:
    """
    Generate high-fidelity domain-characteristic disaster imagery:
    - flood: murky brown/blue water gradients, submerged structures, water ripples
    - earthquake: fractured concrete textures, rubble heaps, gray dust/cracks
    - wildfire: fiery orange/red embers, dark charcoal charred ground, smoke haze
    - landslide: earthy soil tones, diagonal mud flow patterns, rock scatter
    - cyclone_storm: chaotic wind-swept cloud spirals, high contrast dark overcast, rain streaks
    - not_disaster: clear blue sky, green foliage, orderly architecture, bright daylight
    """
    rng = np.random.RandomState(seed)
    img_arr = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    if category == "flood":
        # Water base: murky water shades (blue-brown hues)
        base_color = rng.randint(40, 70), rng.randint(70, 110), rng.randint(90, 140)
        img_arr[:, :] = base_color
        # Add waterline and wave noise
        y_indices, x_indices = np.indices((img_size, img_size))
        wave = np.sin(x_indices / 12.0 + rng.uniform(0, 3.14)) * 15
        wave += np.cos(y_indices / 15.0) * 10
        img_arr = np.clip(img_arr.astype(np.float32) + wave[:, :, None], 0, 255).astype(np.uint8)
        # Submerged structure outlines
        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)
        for _ in range(rng.randint(2, 5)):
            rx = rng.randint(10, img_size - 60)
            ry = rng.randint(10, img_size // 2)
            rw = rng.randint(30, 70)
            rh = rng.randint(30, 80)
            draw.rectangle([rx, ry, rx + rw, ry + rh], fill=(rng.randint(90, 130), rng.randint(80, 110), rng.randint(70, 95)))
            # Roof peak
            draw.polygon([(rx, ry), (rx + rw // 2, max(0, ry - 25)), (rx + rw, ry)], fill=(rng.randint(120, 160), 50, 40))
        # Water reflection overlay
        draw.rectangle([0, img_size // 2, img_size, img_size], fill=None, outline=(50, 90, 120))
        return img.filter(ImageFilter.GaussianBlur(radius=0.5))

    elif category == "earthquake":
        # Concrete / rubble grays and dust
        base_gray = rng.randint(110, 160)
        img_arr[:, :] = [base_gray, base_gray - 5, base_gray - 10]
        # Texture noise (gravel / dust)
        noise = rng.normal(0, 25, (img_size, img_size, 3))
        img_arr = np.clip(img_arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)
        # Draw seismic fracture cracks
        for _ in range(rng.randint(4, 8)):
            cx, cy = rng.randint(0, img_size), rng.randint(0, img_size)
            points = [(cx, cy)]
            for _ in range(rng.randint(5, 12)):
                cx += rng.randint(-20, 20)
                cy += rng.randint(5, 25)
                points.append((cx, cy))
            draw.line(points, fill=(rng.randint(20, 45), rng.randint(20, 45), rng.randint(20, 45)), width=rng.randint(2, 4))
        # Rubble blocks
        for _ in range(rng.randint(8, 16)):
            bx = rng.randint(5, img_size - 30)
            by = rng.randint(5, img_size - 30)
            draw.polygon([(bx, by), (bx + rng.randint(10, 25), by + rng.randint(-5, 10)),
                          (bx + rng.randint(15, 30), by + rng.randint(10, 25)),
                          (bx + rng.randint(-5, 10), by + rng.randint(15, 30))],
                         fill=(rng.randint(70, 100), rng.randint(70, 100), rng.randint(70, 100)))
        return img

    elif category == "wildfire":
        # Dark charred background with intense orange/red fire gradients
        img_arr[:, :] = [rng.randint(20, 40), rng.randint(15, 30), rng.randint(10, 20)]
        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)
        # Fire flames at bottom / middle
        for _ in range(rng.randint(15, 30)):
            fx = rng.randint(0, img_size)
            fy = rng.randint(img_size // 3, img_size)
            fw = rng.randint(20, 60)
            fh = rng.randint(40, 120)
            fire_color = (
                rng.randint(210, 255),
                rng.randint(70, 180),
                rng.randint(10, 50)
            )
            draw.polygon([
                (fx - fw // 2, fy),
                (fx + rng.randint(-10, 10), fy - fh),
                (fx + fw // 2, fy)
            ], fill=fire_color)
        # Smoke haze overlay
        smoke = Image.new("RGBA", (img_size, img_size), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(smoke)
        for _ in range(10):
            sx, sy = rng.randint(0, img_size), rng.randint(0, img_size // 2)
            sr = rng.randint(30, 80)
            sdraw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(80, 80, 80, rng.randint(60, 130)))
        img = Image.alpha_composite(img.convert("RGBA"), smoke).convert("RGB")
        return img.filter(ImageFilter.GaussianBlur(radius=1.0))

    elif category == "landslide":
        # Brown / earthen mudflow on a diagonal slope
        img_arr[:, :] = [rng.randint(100, 135), rng.randint(70, 95), rng.randint(40, 65)]
        y_ind, x_ind = np.indices((img_size, img_size))
        mud_flow = (x_ind * 0.7 + y_ind * 0.8) % 35
        img_arr = np.clip(img_arr.astype(np.float32) + mud_flow[:, :, None] * 1.5, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)
        # Tree debris & rocks
        for _ in range(rng.randint(6, 12)):
            tx = rng.randint(10, img_size - 40)
            ty = rng.randint(10, img_size - 40)
            draw.line([(tx, ty), (tx + rng.randint(20, 50), ty + rng.randint(-15, 30))],
                      fill=(rng.randint(40, 60), rng.randint(25, 40), rng.randint(15, 25)), width=rng.randint(3, 5))
            # Rocks
            draw.ellipse([tx, ty, tx + rng.randint(8, 20), ty + rng.randint(8, 18)],
                         fill=(rng.randint(70, 95), rng.randint(70, 95), rng.randint(70, 95)))
        return img.filter(ImageFilter.GaussianBlur(radius=0.7))

    elif category == "cyclone_storm":
        # Swirling dark storm clouds with rain sheets
        img_arr[:, :] = [rng.randint(50, 80), rng.randint(60, 90), rng.randint(75, 105)]
        y_ind, x_ind = np.indices((img_size, img_size))
        # Spiral pattern
        cx, cy = img_size // 2, img_size // 2
        r = np.sqrt((x_ind - cx)**2 + (y_ind - cy)**2) + 1e-5
        theta = np.arctan2(y_ind - cy, x_ind - cx)
        spiral = np.sin(theta * 3.0 + r / 10.0) * 30
        img_arr = np.clip(img_arr.astype(np.float32) + spiral[:, :, None], 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)
        # Heavy rain streaks
        for _ in range(rng.randint(40, 80)):
            rx = rng.randint(0, img_size)
            ry = rng.randint(0, img_size - 30)
            draw.line([(rx, ry), (rx + rng.randint(-15, -5), ry + rng.randint(20, 40))],
                      fill=(200, 215, 230), width=1)
        return img.filter(ImageFilter.GaussianBlur(radius=0.6))

    else:  # not_disaster
        # Clear sky + green grass / neat buildings
        img_arr[:img_size // 2, :] = [rng.randint(100, 135), rng.randint(170, 210), rng.randint(220, 250)]  # Sky
        img_arr[img_size // 2:, :] = [rng.randint(60, 95), rng.randint(140, 180), rng.randint(50, 80)]     # Green field/grass
        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)
        # Neat modern house / structure
        hx = img_size // 4
        hy = img_size // 3
        draw.rectangle([hx, hy, hx + 80, hy + 60], fill=(230, 225, 220), outline=(60, 60, 60), width=2)
        draw.polygon([(hx - 10, hy), (hx + 40, hy - 30), (hx + 90, hy)], fill=(180, 50, 40))
        # Sun
        draw.ellipse([img_size - 45, 15, img_size - 15, 45], fill=(255, 230, 100))
        return img


def _synthesize_satellite_damage_image(damage_class: str, img_size: int = 224, seed: int = 0) -> Image.Image:
    """
    Generate synthetic satellite building imagery for xBD 4-tier damage classification:
    - no_damage: intact rectangular roof polygons, pristine driveway, clean boundaries
    - minor_damage: intact roof with minor debris/tarp or small surface discoloration
    - major_damage: partially collapsed roof, heavy structural deformation, visible wall cracks
    - destroyed: completely leveled structure, rubble pile footprint, debris field
    """
    rng = np.random.RandomState(seed)
    # Ground background: satellite earth/grass/pavement tone
    bg_tone = [rng.randint(120, 150), rng.randint(130, 160), rng.randint(110, 135)]
    img_arr = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img_arr[:, :] = bg_tone
    # Background texture
    noise = rng.normal(0, 12, (img_size, img_size, 3))
    img_arr = np.clip(img_arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)

    # Road / Driveway
    draw.rectangle([0, 180, img_size, 205], fill=(85, 85, 90))

    # Center building footprint coordinates
    bx1, by1 = 55, 55
    bx2, by2 = 165, 155

    if damage_class == "no_damage":
        # Sharp, intact roof structure
        roof_color = (rng.randint(180, 215), rng.randint(65, 95), rng.randint(55, 75))
        draw.rectangle([bx1, by1, bx2, by2], fill=roof_color, outline=(40, 40, 40), width=2)
        # Ridge line
        draw.line([(bx1, (by1 + by2) // 2), (bx2, (by1 + by2) // 2)], fill=(230, 230, 230), width=2)
        # Surrounding clean yard
        draw.rectangle([bx1 - 10, by1 - 10, bx1 - 2, by2 + 10], fill=(80, 150, 70))

    elif damage_class == "minor_damage":
        # Intact roof structure with blue emergency tarp or slight shingles loss
        roof_color = (rng.randint(160, 195), rng.randint(65, 90), rng.randint(55, 70))
        draw.rectangle([bx1, by1, bx2, by2], fill=roof_color, outline=(50, 50, 50), width=2)
        # Blue tarp on corner
        draw.polygon([(bx1 + 10, by1 + 10), (bx1 + 45, by1 + 10), (bx1 + 35, by1 + 45), (bx1 + 5, by1 + 40)],
                     fill=(30, 90, 220), outline=(10, 50, 160))
        # Minor debris dots
        for _ in range(12):
            dx, dy = rng.randint(bx1 - 15, bx2 + 15), rng.randint(by1 - 15, by2 + 15)
            draw.ellipse([dx, dy, dx + 3, dy + 3], fill=(70, 65, 60))

    elif damage_class == "major_damage":
        # Partially collapsed building, missing roof sections, interior exposure
        roof_color = (130, 70, 60)
        draw.rectangle([bx1, by1, bx2, by2], fill=roof_color, outline=(40, 40, 40), width=2)
        # Big jagged collapse breach
        breach = [
            (bx1 + 25, by1 + 15),
            (bx1 + 80, by1 + 20),
            (bx1 + 95, by1 + 75),
            (bx1 + 40, by1 + 85),
            (bx1 + 15, by1 + 50)
        ]
        # Exposed dark ruined interior
        draw.polygon(breach, fill=(45, 35, 30), outline=(25, 20, 18), width=2)
        # Scattered broken beams
        for _ in range(8):
            lx, ly = rng.randint(bx1 + 10, bx2 - 10), rng.randint(by1 + 10, by2 - 10)
            draw.line([(lx, ly), (lx + rng.randint(-15, 15), ly + rng.randint(-15, 15))], fill=(190, 170, 140), width=2)

    else:  # destroyed
        # Building completely flattened into rubble footprint
        draw.rectangle([bx1, by1, bx2, by2], fill=(90, 85, 80), outline=(60, 55, 50))
        # Total rubble chaos
        for _ in range(35):
            rx = rng.randint(bx1 - 10, bx2 + 10)
            ry = rng.randint(by1 - 10, by2 + 10)
            rw = rng.randint(4, 18)
            rh = rng.randint(4, 18)
            color = (rng.randint(50, 120), rng.randint(45, 110), rng.randint(40, 95))
            draw.polygon([(rx, ry), (rx + rw, ry + rng.randint(-4, 6)), (rx + rw - 3, ry + rh), (rx - 2, ry + rh - 2)], fill=color)
        # Char / crater marks
        draw.ellipse([bx1 + 30, by1 + 30, bx1 + 75, by1 + 70], fill=(40, 35, 32))

    return img.filter(ImageFilter.GaussianBlur(radius=0.4))


def prepare_all_datasets(
    num_samples_per_class: int = 120,
    seed: int = 42
):
    """
    Generate and persist all dataset splits for:
    1. Image Disaster Classifier
    2. Text Disaster Classifier
    3. Satellite Damage Assessment Model
    4. Multimodal Fusion Model
    """
    set_seed(seed)
    print("=" * 60)
    print("Preparing Multimodal Disaster Datasets...")
    print(f"Samples per class: {num_samples_per_class}")
    print("=" * 60)

    # ── 1. Image Disaster Classifier Dataset ───────────────────────
    img_proc_dir = PROCESSED_DIR / "image_classifier" / "images"
    img_split_dir = SPLITS_DIR / "image_classifier"
    img_proc_dir.mkdir(parents=True, exist_ok=True)
    img_split_dir.mkdir(parents=True, exist_ok=True)

    image_records = []
    global_idx = 0
    for class_idx, cat in enumerate(DISASTER_CLASSES):
        for s in range(num_samples_per_class):
            img_filename = f"{cat}_{s:04d}.jpg"
            img_path = img_proc_dir / img_filename
            rel_img_path = f"data/processed/image_classifier/images/{img_filename}"
            
            # Generate and save image
            img = _synthesize_disaster_image(cat, img_size=224, seed=seed + global_idx)
            img.save(img_path, quality=92)

            image_records.append({
                "image_id": f"img_{global_idx:05d}",
                "image_path": rel_img_path,
                "label": cat,
                "label_id": class_idx
            })
            global_idx += 1

    # Train / Val / Test split (70 / 15 / 15)
    random.shuffle(image_records)
    n_total = len(image_records)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    img_train = image_records[:n_train]
    img_val = image_records[n_train:n_train + n_val]
    img_test = image_records[n_train + n_val:]

    def _write_csv(records, filepath, fieldnames):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    img_fields = ["image_id", "image_path", "label", "label_id"]
    _write_csv(img_train, img_split_dir / "train.csv", img_fields)
    _write_csv(img_val, img_split_dir / "val.csv", img_fields)
    _write_csv(img_test, img_split_dir / "test.csv", img_fields)
    print(f"✓ Image Dataset created: {len(img_train)} train, {len(img_val)} val, {len(img_test)} test")

    # ── 2. Text Disaster Classifier Dataset ────────────────────────
    text_split_dir = SPLITS_DIR / "text_classifier"
    text_split_dir.mkdir(parents=True, exist_ok=True)

    text_records = []
    text_idx = 0
    for class_idx, cat in enumerate(DISASTER_CLASSES):
        templates = TEXT_TEMPLATES[cat]
        for s in range(num_samples_per_class * 2):  # more text variations
            template = templates[s % len(templates)]
            loc = LOCATIONS[s % len(LOCATIONS)]
            text = template.format(
                loc=loc,
                depth=random.randint(2, 12),
                mag=round(random.uniform(5.2, 7.8), 1),
                acres=random.randint(500, 15000),
                cat=random.randint(1, 5),
                wind=random.randint(120, 240),
                storm_name=random.choice(["Alpha", "Beryl", "Helene", "Milton", "Viper"]),
                temp=random.randint(18, 28)
            )
            # Add realistic social media / dispatch formatting variations
            if s % 3 == 0:
                text = f"[EMERGENCY ALERT] {text}"
            elif s % 4 == 0:
                text = f"Report from field officer: {text} #DisasterUpdate"

            text_records.append({
                "text_id": f"txt_{text_idx:05d}",
                "text": text,
                "label": cat,
                "label_id": class_idx
            })
            text_idx += 1

    random.shuffle(text_records)
    n_total_t = len(text_records)
    n_train_t = int(n_total_t * 0.70)
    n_val_t = int(n_total_t * 0.15)

    txt_train = text_records[:n_train_t]
    txt_val = text_records[n_train_t:n_train_t + n_val_t]
    txt_test = text_records[n_train_t + n_val_t:]

    txt_fields = ["text_id", "text", "label", "label_id"]
    _write_csv(txt_train, text_split_dir / "train.csv", txt_fields)
    _write_csv(txt_val, text_split_dir / "val.csv", txt_fields)
    _write_csv(txt_test, text_split_dir / "test.csv", txt_fields)
    print(f"✓ Text Dataset created: {len(txt_train)} train, {len(txt_val)} val, {len(txt_test)} test")

    # ── 3. Satellite Damage Assessment Dataset ─────────────────────
    dmg_proc_dir = PROCESSED_DIR / "damage_assessment" / "images"
    dmg_split_dir = SPLITS_DIR / "damage_assessment"
    dmg_proc_dir.mkdir(parents=True, exist_ok=True)
    dmg_split_dir.mkdir(parents=True, exist_ok=True)

    damage_records = []
    dmg_idx = 0
    for class_idx, dmg_cat in enumerate(DAMAGE_CLASSES):
        for s in range(num_samples_per_class):
            img_filename = f"{dmg_cat}_{s:04d}.jpg"
            img_path = dmg_proc_dir / img_filename
            rel_img_path = f"data/processed/damage_assessment/images/{img_filename}"

            img = _synthesize_satellite_damage_image(dmg_cat, img_size=224, seed=seed + dmg_idx)
            img.save(img_path, quality=92)

            damage_records.append({
                "damage_id": f"dmg_{dmg_idx:05d}",
                "image_path": rel_img_path,
                "label": dmg_cat,
                "label_id": class_idx
            })
            dmg_idx += 1

    random.shuffle(damage_records)
    n_total_d = len(damage_records)
    n_train_d = int(n_total_d * 0.70)
    n_val_d = int(n_total_d * 0.15)

    dmg_train = damage_records[:n_train_d]
    dmg_val = damage_records[n_train_d:n_train_d + n_val_d]
    dmg_test = damage_records[n_train_d + n_val_d:]

    dmg_fields = ["damage_id", "image_path", "label", "label_id"]
    _write_csv(dmg_train, dmg_split_dir / "train.csv", dmg_fields)
    _write_csv(dmg_val, dmg_split_dir / "val.csv", dmg_fields)
    _write_csv(dmg_test, dmg_split_dir / "test.csv", dmg_fields)
    print(f"✓ Damage Dataset created: {len(dmg_train)} train, {len(dmg_val)} val, {len(dmg_test)} test")

    # ── 4. Multimodal Dataset (Paired Image + Text) ────────────────
    mm_split_dir = SPLITS_DIR / "multimodal"
    mm_split_dir.mkdir(parents=True, exist_ok=True)

    mm_records = []
    for i, img_rec in enumerate(image_records):
        cat = img_rec["label"]
        templates = TEXT_TEMPLATES[cat]
        template = templates[i % len(templates)]
        text = template.format(
            loc=LOCATIONS[i % len(LOCATIONS)],
            depth=random.randint(2, 10),
            mag=round(random.uniform(5.5, 7.5), 1),
            acres=random.randint(800, 10000),
            cat=random.randint(1, 5),
            wind=random.randint(130, 220),
            storm_name="StormUnit",
            temp=22
        )
        mm_records.append({
            "incident_id": f"inc_{i:05d}",
            "image_path": img_rec["image_path"],
            "text": text,
            "label": cat,
            "label_id": img_rec["label_id"]
        })

    random.shuffle(mm_records)
    n_total_m = len(mm_records)
    n_train_m = int(n_total_m * 0.70)
    n_val_m = int(n_total_m * 0.15)

    mm_train = mm_records[:n_train_m]
    mm_val = mm_records[n_train_m:n_train_m + n_val_m]
    mm_test = mm_records[n_train_m + n_val_m:]

    mm_fields = ["incident_id", "image_path", "text", "label", "label_id"]
    _write_csv(mm_train, mm_split_dir / "train.csv", mm_fields)
    _write_csv(mm_val, mm_split_dir / "val.csv", mm_fields)
    _write_csv(mm_test, mm_split_dir / "test.csv", mm_fields)
    print(f"✓ Multimodal Dataset created: {len(mm_train)} train, {len(mm_val)} val, {len(mm_test)} test")

    # ── 5. Generate Sample Reports for UI Demos ─────────────────────
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    sample_cases = [
        {
            "name": "case_1_critical_flood",
            "category": "flood",
            "damage": "major_damage",
            "text": "Flash flood breached coastal embankment at Riverside. Over 40 families trapped on upper floors. Severe water currents preventing small boat rescue. Urgent helicopter airlift and medical evacuation requested.",
            "location_name": "Riverside Marina, Sector 4",
            "latitude": 29.9511,
            "longitude": -90.0715,
            "deaths": 1,
            "injuries": 14,
            "missing": 3,
            "displaced": 120
        },
        {
            "name": "case_2_wildfire_outbreak",
            "category": "wildfire",
            "damage": "destroyed",
            "text": "Fast-moving brushfire driven by 55mph winds crossed highway 12 into Highland Ridge. Five homes completely destroyed, twelve others actively burning. Power substation ignited. Zero visibility due to dense smoke.",
            "location_name": "Highland Ridge Forest Boundary",
            "latitude": 34.0522,
            "longitude": -118.2437,
            "deaths": 0,
            "injuries": 6,
            "missing": 1,
            "displaced": 350
        },
        {
            "name": "case_3_earthquake_collapse",
            "category": "earthquake",
            "damage": "destroyed",
            "text": "Magnitude 6.8 seismic rupture collapsed 3 commercial buildings and fractured elevated highway overpass in Central Metro. Acoustic search detected taps under rubble. Heavy extrication gear needed immediately.",
            "location_name": "Central Metro District",
            "latitude": 37.7749,
            "longitude": -122.4194,
            "deaths": 4,
            "injuries": 38,
            "missing": 8,
            "displaced": 500
        },
        {
            "name": "case_4_cyclone_surge",
            "category": "cyclone_storm",
            "damage": "major_damage",
            "text": "Category 4 tropical cyclone landfall generated 14-foot storm surge. Hospital emergency wing flooded and lost grid power. Backup generator running low on diesel. 80 critical patients require priority triage.",
            "location_name": "Coastal Bay General Hospital",
            "latitude": 25.7617,
            "longitude": -80.1918,
            "deaths": 2,
            "injuries": 25,
            "missing": 0,
            "displaced": 200
        },
        {
            "name": "case_5_normal_monitoring",
            "category": "not_disaster",
            "damage": "no_damage",
            "text": "Routine aerial monitoring of West Valley water reservoir. Water levels within standard seasonal thresholds. All spillways operating nominally with zero structural anomalies detected.",
            "location_name": "West Valley Reservoir Dam",
            "latitude": 36.1699,
            "longitude": -115.1398,
            "deaths": 0,
            "injuries": 0,
            "missing": 0,
            "displaced": 0
        }
    ]

    for sc in sample_cases:
        sample_img = _synthesize_disaster_image(sc["category"], img_size=384, seed=random.randint(1000, 9999))
        sample_img_path = SAMPLES_DIR / f"{sc['name']}.jpg"
        sample_img.save(sample_img_path, quality=95)

        sat_img = _synthesize_satellite_damage_image(sc["damage"], img_size=384, seed=random.randint(1000, 9999))
        sat_img_path = SAMPLES_DIR / f"{sc['name']}_satellite.jpg"
        sat_img.save(sat_img_path, quality=95)

        meta_path = SAMPLES_DIR / f"{sc['name']}_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(sc, f, indent=2)

    print(f"✓ Saved {len(sample_cases)} full sample demo incident packages in {SAMPLES_DIR}")
    print("=" * 60)
    print("All datasets prepared successfully!")
    print("=" * 60)


if __name__ == "__main__":
    prepare_all_datasets()
