import os
import SimpleITK as sitk
import numpy as np
from PIL import Image

def mha2jpg(mha_file, output_folder):
    image = sitk.ReadImage(mha_file)
    img_data = sitk.GetArrayFromImage(image)
    max_val, min_val = np.max(img_data), np.min(img_data)
    img_data = ((img_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    os.makedirs(output_folder, exist_ok=True)
    for s in range(img_data.shape[0]):
        slicer = Image.fromarray(img_data[s, :, :])
        slicer.save(os.path.join(output_folder, f"{s}.jpg"))

if __name__ == "__main__":
    mha_file = r""
    output_folder = r""
    mha2jpg(mha_file, output_folder)