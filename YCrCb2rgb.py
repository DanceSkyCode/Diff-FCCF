import os
import cv2
import numpy as np
from tqdm import tqdm
base_dir = r""
output_dir = os.path.join(base_dir, "RGB_imgs")
os.makedirs(output_dir, exist_ok=True)
for idx in tqdm(range(1, 25), desc="Processing index groups"):
    idx_str = f"{idx:02d}"
    for step in range(4000):
        step_str = f"{step}"
        y_path = os.path.join(base_dir, f"Diff-FCCF_0_{idx_str}_{step_str}.png")
        cr_path = os.path.join(base_dir, f"Diff-FCCF_1_{idx_str}_{step_str}.png")
        cb_path = os.path.join(base_dir, f"Diff-FCCF_2_{idx_str}_{step_str}.png")
        if not (os.path.exists(y_path) and os.path.exists(cr_path) and os.path.exists(cb_path)):
            print(f"Missing channels for index {idx_str}, step {step_str}")
            continue
        y = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        cr = cv2.imread(cr_path, cv2.IMREAD_GRAYSCALE)
        cb = cv2.imread(cb_path, cv2.IMREAD_GRAYSCALE)
        ycrcb = cv2.merge([y, cr, cb])
        rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        rgb_filename = f"RGB_{idx_str}_{step_str}.png"
        rgb_path = os.path.join(output_dir, rgb_filename)
        cv2.imwrite(rgb_path, rgb)
