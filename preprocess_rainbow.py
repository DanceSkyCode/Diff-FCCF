import os
import cv2
import numpy as np

input_dir = r"\MRI-PET\MRI"
output_dir = os.path.join(input_dir, "colored_brain_only")
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        img_path = os.path.join(input_dir, filename)
        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray_img is None:
            print(f"can not read the file: {img_path}")
            continue
        colored_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
        save_path = os.path.join(output_dir, f"{filename}")
        cv2.imwrite(save_path, colored_img)
        print(f"save in: {save_path}")
