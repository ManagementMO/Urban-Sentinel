import cv2
import numpy as np
import os
import re

DATA_FOLDER = "dataset_for_statelite"

def get_pixel_color(img_path, point):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    b, g, r = img[point[1], point[0]]
    return np.array([r, g, b])  # RGB

def extract_year(filename):
    match = re.search(r'\d{4}', filename)
    return int(match.group()) if match else -1

def check_if_darker_over_years(point):
    image_files = [
        f for f in os.listdir(DATA_FOLDER)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')) and extract_year(f) != -1
    ]
    image_files.sort(key=lambda f: extract_year(f))

    if len(image_files) < 2:
        return {"error": "Not enough images to compare."}

    brightness_by_year = []
    for filename in image_files:
        full_path = os.path.join(DATA_FOLDER, filename)
        year = extract_year(filename)
        color = get_pixel_color(full_path, point)
        brightness = float(np.mean(color))  # Convert to Python float
        brightness_by_year.append((year, brightness))

    first_year, first_brightness = brightness_by_year[0]
    last_year, last_brightness = brightness_by_year[-1]
    is_darker = bool(last_brightness < first_brightness)  # Convert to Python bool

    return {
        "from_year": first_year,
        "to_year": last_year,
        "brightness_start": round(first_brightness, 2),
        "brightness_end": round(last_brightness, 2),
        "darker_over_time": is_darker
    }
