import os
import numpy as np
from tqdm import tqdm
from model.evaluate import Evaluator, image_read_cv2
pet_folder = r"/MRI"
mri_folder = r"/PET"
fusion_folder = r"/Fusion"
metric_result = np.zeros(8)
for img_name in tqdm(os.listdir(fusion_folder)):
    pet_path = os.path.join(pet_folder, img_name)
    mri_path = os.path.join(mri_folder, img_name)
    fusion_path = os.path.join(fusion_folder, img_name)
    if not (os.path.exists(pet_path) and os.path.exists(mri_path) and os.path.exists(fusion_path)):
        print(f"[WARNING] Missing file: {img_name}")
        continue
    ir = image_read_cv2(pet_path, 'GRAY')
    vi = image_read_cv2(mri_path, 'GRAY')
    fi = image_read_cv2(fusion_path, 'GRAY')
    metrics = np.array([
        Evaluator.EN(fi),
        Evaluator.SD(fi),
        Evaluator.SF(fi),
        Evaluator.MI(fi, ir, vi),
        Evaluator.SCD(fi, ir, vi),
        Evaluator.VIFF(fi, ir, vi),
        Evaluator.Qabf(fi, ir, vi),
        Evaluator.SSIM(fi, ir, vi)
    ])
    metric_result += metrics
num_files = len(os.listdir(mri_folder))
if num_files > 0:
    metric_result /= num_files 
print("\t\t EN\t SD\t SF\t MI\t SCD\t VIF\t Qabf\t SSIM")
print("TEST-med-PET" + '\t' + '\t'.join(map(lambda x: str(np.round(x, 2)), metric_result)))
print("=" * 80)