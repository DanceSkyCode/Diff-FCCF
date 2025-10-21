import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

img_path = '/CT-MRI/MRI/16016.png'
save_dir = ''
os.makedirs(save_dir, exist_ok=True)
bgr_img = cv2.imread(img_path)
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
Y, Cr, Cb = cv2.split(ycrcb_img)
cv2.imwrite(os.path.join(save_dir, "rgb_image.png"), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(save_dir, "Y_channel.png"), Y)
cv2.imwrite(os.path.join(save_dir, "Cr_channel.png"), Cr)
cv2.imwrite(os.path.join(save_dir, "Cb_channel.png"), Cb)
print(f"save in: {save_dir}")