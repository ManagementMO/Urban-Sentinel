import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import re

selected_point = []

def on_click(event):
    if event.xdata is not None and event.ydata is not None:
        selected_point.append((int(event.xdata), int(event.ydata)))
        plt.close()

def select_point(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title("Click a point on the image")
    plt.connect('button_press_event', on_click)
    plt.show()

    return selected_point[0]

def get_pixel_color(img_path, point):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    b, g, r = img[point[1], point[0]]
    return np.array([r, g, b])  # RGB

def extract_year(filename):
    match = re.search(r'\d{4}', filename)
    return int(match.group()) if match else -1

def check_if_darker_over_years(reference_img_path, folder_path):
    point = select_point(reference_img_path)

    # Gather and sort images by year
    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')) and extract_year(f) != -1
    ]
    image_files.sort(key=lambda f: extract_year(f))

    brightness_by_year = []

    for filename in image_files:
        full_path = os.path.join(folder_path, filename)
        year = extract_year(filename)
        color = get_pixel_color(full_path, point)
        brightness = np.mean(color)  # Average of R, G, B
        brightness_by_year.append((year, brightness))
        print(f"{year}: Brightness = {brightness:.2f} RGB = {color}")

    # Compare the first and last year
    if brightness_by_year and len(brightness_by_year) > 1:
        first_brightness = brightness_by_year[0][1]
        last_brightness = brightness_by_year[-1][1]
        is_darker = last_brightness < first_brightness
        print(f"\nOverall darker from {brightness_by_year[0][0]} to {brightness_by_year[-1][0]}: {is_darker}")
        return is_darker
    else:
        print("Not enough images to determine trend.")
        return False

# Example usage
reference_image = "dataset_for_statelite/2017.png"  # Select a point from this image
images_folder = "dataset_for_statelite"
check_if_darker_over_years(reference_image, images_folder)
