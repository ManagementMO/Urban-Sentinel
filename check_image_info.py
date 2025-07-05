import cv2
import os

def check_image_info():
    data_folder = "dataset_for_statelite"
    image_files = [f for f in os.listdir(data_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    
    print("Checking image information:")
    for filename in image_files:
        full_path = os.path.join(data_folder, filename)
        img = cv2.imread(full_path)
        if img is not None:
            height, width, channels = img.shape
            print(f"{filename}: {width}x{height} (channels: {channels})")
        else:
            print(f"{filename}: Could not load")

def test_coordinates():
    from dataset_for_statelite.main import check_if_darker_over_years
    
    # Test different coordinates
    test_coords = [(10, 10), (50, 50), (100, 100), (200, 200)]
    
    for x, y in test_coords:
        try:
            print(f"\nTesting coordinates ({x}, {y}):")
            result = check_if_darker_over_years((x, y))
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    check_image_info()
    test_coordinates() 