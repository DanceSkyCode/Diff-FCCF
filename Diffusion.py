import torch
import math
import torch.nn.functional as F
import os
import cv2
import time
import torch.nn as nn
import numpy as np
from utils import tensor2img
import model.loss as loss
from torch_scatter import scatter_mean 
from torch.nn import LayerNorm
device = "cuda:0" if torch.cuda.is_available() else "cpu"
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as T
from PIL import Image
import torch
import torch.nn.functional as F
save_dir = ''
def normalize_with_ref(f0, ref_values):
    # f0: [B, 1, H, W]
    B = f0.size(0)
    f0_normed = []
    for i in range(B):
        t = f0[i]  # shape: [1, H, W]
        ref_value = ref_values[i]  # ref_value is shape: [H, W]
        min_val = t.min()
        max_val = t.max()
        if (max_val - min_val) > 1e-5:
            t_norm = 2 * (t - min_val) / (max_val - min_val) - 1
        else:
            t_norm = torch.zeros_like(t)
        mask = (t == ref_value)  # mask with the same value as ref_value
        t_norm[mask] = t[mask] 
        f0_normed.append(t_norm)
    return torch.stack(f0_normed)  # shape: [B, 1, H, W]
def normalize_with_ref_distribution(f0, sourceImg2, ref_values):
    # f0: [B, 1, H, W]
    # sourceImg2: [B, 1, H, W]
    # ref_values: [B, H, W]
    B = f0.size(0)
    f0_normed = []
    mean_source = sourceImg2.mean(dim=(1, 2, 3), keepdim=True)
    std_source = sourceImg2.std(dim=(1, 2, 3), keepdim=True)
    for i in range(B):
        t = f0[i]  # shape: [1, H, W]
        ref_value = ref_values[i]  # ref_value: shape: [H, W]
        t_norm = (t - t.mean()) / (t.std() + 1e-5) 
        t_norm = t_norm * std_source[i] + mean_source[i] 
        mask = (t == ref_value)  # mask with the same value as ref_value
        t_norm[mask] = t[mask]  
        f0_normed.append(t_norm)
    return torch.stack(f0_normed)  # shape: [B, 1, H, W]
def save_tensor_batch_as_images(tensor_batch, prefix):
    tensor_batch = tensor_batch.detach().cpu().clone()
    for i in range(tensor_batch.size(0)):
        single_img = tensor_batch[i].squeeze(0)  # [1, H, W] -> [H, W]
        img_array = ((single_img + 1) / 2.0 * 255.0).clamp(0, 255).byte().numpy()
        img = Image.fromarray(img_array)
        img.save(os.path.join(save_dir, f"{prefix}_{i}.png"))
def compute_hessian_determinant(DVField):
    DVField = DVField.permute(0, 3, 1, 2)  # [B, C, H, W]
    B, C, H, W = DVField.shape
    det_list = []
    for b in range(B):
        per_batch = []
        for c in range(C):
            f = DVField[b, c]  # shape: [H, W]
            fy = torch.gradient(f, dim=0)[0]  # d/dy
            fx = torch.gradient(f, dim=1)[0]  # d/dx
            fyy = torch.gradient(fy, dim=0)[0]
            fxx = torch.gradient(fx, dim=1)[0]
            fxy = torch.gradient(fx, dim=0)[0]
            fyx = torch.gradient(fy, dim=1)[0]
            hess_det = fxx * fyy - fxy * fyx
            per_batch.append(hess_det)
        det_list.append(torch.stack(per_batch, dim=0))  # [C, H, W]
    return torch.stack(det_list, dim=0)  # [B, C, H, W]
def mse_loss(sourceImg1, sourceImg2_deformed):
    return F.mse_loss(sourceImg1, sourceImg2_deformed)
def correlation_loss(sourceImg1, sourceImg2_deformed):
    B = sourceImg1.shape[0]
    losses = []
    for i in range(B):
        x1 = sourceImg1[i].view(-1)
        x2 = sourceImg2_deformed[i].view(-1)
        x1_mean = x1.mean()
        x2_mean = x2.mean()
        numerator = ((x1 - x1_mean) * (x2 - x2_mean)).sum()
        denominator = torch.sqrt(((x1 - x1_mean)**2).sum() * ((x2 - x2_mean)**2).sum())
        corr = numerator / (denominator + 1e-8)
        losses.append(-corr) 
    return torch.stack(losses).mean()
def mutual_information_loss(sourceImg1, sourceImg2_deformed, num_bins=64):
    B = sourceImg1.shape[0]
    losses = []
    for i in range(B):
        x = sourceImg1[i].detach().cpu().numpy().flatten()
        y = sourceImg2_deformed[i].detach().cpu().numpy().flatten()
        x = (x + 1) / 2
        y = (y + 1) / 2
        joint_hist, _, _ = np.histogram2d(x, y, bins=num_bins, range=[[0, 1], [0, 1]])
        joint_prob = joint_hist / np.sum(joint_hist)
        px = np.sum(joint_prob, axis=1, keepdims=True)
        py = np.sum(joint_prob, axis=0, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            pxy = joint_prob
            pxy_safe = np.where(pxy == 0, 1e-10, pxy)
            px_safe = np.where(px == 0, 1e-10, px)
            py_safe = np.where(py == 0, 1e-10, py)
            mi = np.sum(pxy_safe * np.log(pxy_safe / (px_safe @ py_safe + 1e-10)))
        losses.append(-mi)
    return torch.tensor(losses).mean()
def variance_ratio_loss(sourceImg1, sourceImg2_deformed):
    B = sourceImg1.shape[0]
    losses = []
    for i in range(B):
        x = sourceImg1[i].view(-1)
        y = sourceImg2_deformed[i].view(-1)

        unique_classes = torch.unique(x)
        total_mean = y.mean()
        SS_total = ((y - total_mean)**2).sum()
        SS_between = 0
        for cls in unique_classes:
            mask = (x == cls)
            if mask.sum() > 0:
                group_mean = y[mask].mean()
                SS_between += ((group_mean - total_mean)**2) * mask.sum()
        eta = 1 - (SS_between / (SS_total + 1e-8))
        losses.append(eta)
    return torch.stack(losses).mean()
def apply_deformation_train(sourceImg1, DVField, save_dir=None, prefix='sample'):
    # sourceImg1: [B, C, H, W]
    # DVField: [B, H, W, 2]
    # DVField = DVField.permute(0,2,3,1)
    B, C, H, W = sourceImg1.size()
    grid_x, grid_y = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
    grid = torch.stack([grid_y.to(sourceImg1.device), grid_x.to(sourceImg1.device)], dim=-1).float()  # [H, W, 2]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]
    deformed_grid = grid + DVField  # [B, H, W, 2]
    norm_factor = torch.tensor([W-1, H-1]).float().to(sourceImg1.device)
    deformed_grid = 2.0 * deformed_grid / norm_factor - 1.0  # 归一化后仍是 [B, H, W, 2]
    warped_img = F.grid_sample(sourceImg1, deformed_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        to_pil = T.ToPILImage()
        for i in range(B):
            img_org = sourceImg1[i]  # [C, H, W]
            img_def = warped_img[i]
            diff = torch.abs(img_org - img_def)
            # Normalize for visualization (0-1)
            img_org_vis = (img_org - img_org.min()) / (img_org.max() - img_org.min() + 1e-8)
            img_def_vis = (img_def - img_def.min()) / (img_def.max() - img_def.min() + 1e-8)
            diff_vis = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
            to_pil(img_org_vis.cpu()).save(os.path.join(save_dir, f'{prefix}_original_{i}.png'))
            to_pil(img_def_vis.cpu()).save(os.path.join(save_dir, f'{prefix}_deformed_{i}.png'))
            to_pil(diff_vis.cpu()).save(os.path.join(save_dir, f'{prefix}_residual_{i}.png'))

    return warped_img
def apply_deformation(sourceImg1, DVField, save_dir=None, prefix='sample'):
    DVField = DVField.permute(0,2,3,1)
    B, C, H, W = sourceImg1.size()
    grid_x, grid_y = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
    grid = torch.stack([grid_y.to(sourceImg1.device), grid_x.to(sourceImg1.device)], dim=-1).float()  # [H, W, 2]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]
    deformed_grid = grid + DVField  # [B, H, W, 2]
    norm_factor = torch.tensor([W-1, H-1]).float().to(sourceImg1.device)
    deformed_grid = 2.0 * deformed_grid / norm_factor - 1.0  # 归一化后仍是 [B, H, W, 2]
    warped_img = F.grid_sample(sourceImg1, deformed_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        to_pil = T.ToPILImage()
        for i in range(B):
            img_org = sourceImg1[i]  # [C, H, W]
            img_def = warped_img[i]
            diff = torch.abs(img_org - img_def)
            # Normalize for visualization (0-1)
            img_org_vis = (img_org - img_org.min()) / (img_org.max() - img_org.min() + 1e-8)
            img_def_vis = (img_def - img_def.min()) / (img_def.max() - img_def.min() + 1e-8)
            diff_vis = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
            to_pil(img_org_vis.cpu()).save(os.path.join(save_dir, f'{prefix}_original_{i}.png'))
            to_pil(img_def_vis.cpu()).save(os.path.join(save_dir, f'{prefix}_deformed_{i}.png'))
            to_pil(diff_vis.cpu()).save(os.path.join(save_dir, f'{prefix}_residual_{i}.png'))
    return warped_img
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
def cosine_beta_schedule(timesteps):
    return betas_for_alpha_bar(
        timesteps,
        lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
    )
# Create a beta schedule that discretizes the given alpha_t_bar function
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)
def composite_loss(output, target_y, original_crcb, alpha=0.5, beta=0.3, gamma=0.2):
    mse = torch.mean((output - target_y)**2)
    boundary_mask = (output - original_crcb).abs() < 0.1
    penalty = torch.mean(boundary_mask.float() * (output - original_crcb)**2)
    local_var = F.avg_pool2d((output - output.mean())**2, 3, padding=1)
    contrast_loss = -local_var.mean()
    wasserstein = torch.abs(torch.sort(output.flatten())[0] - 
                          torch.sort(original_crcb.flatten())[0]).mean()
    return alpha*(mse + penalty) + beta*contrast_loss + gamma*wasserstein
class GaussianDiffusion:
    def __init__(
            self,
            timesteps=2000,
            beta_schedule='cosine'
    ):
        self.timesteps = timesteps
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas
        self.global_save_counter = 0
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(self.alphas)
                / (1.0 - self.alphas_cumprod)
        )
        self.fusion_loss = loss.Fusion_loss(lambda1=30,lambda2=40,lambda3=40).to(device)
    # Get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_1, x_2, t, noise=None, return_noise=False):
        if noise is None:
            noise = torch.randn_like(x_1)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_1.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_1.shape)
        if return_noise:
            return sqrt_alphas_cumprod_t * x_1 + sqrt_one_minus_alphas_cumprod_t * noise, sqrt_alphas_cumprod_t * x_2 + sqrt_one_minus_alphas_cumprod_t * noise, noise
        else:
            return sqrt_alphas_cumprod_t * x_1 + sqrt_one_minus_alphas_cumprod_t * noise, sqrt_alphas_cumprod_t * x_2 + sqrt_one_minus_alphas_cumprod_t * noise
    # Get the mean and variance of q(x_t | x_0).
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    # Compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    i=0
    # Compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, sourceImg1, sourceImg2, x_t1, x_t2, x_t, t, concat_type, clip_denoised=True, fusion_type = "spect", check = 0, DVField_t=None):

        input = torch.cat([sourceImg1, sourceImg2, x_t1, x_t2], dim=1)
        pred_noise, mask1, mask2 , DVField = model(input, t)
        if fusion_type =="pet":
            if check==0:
                imgf = mask1 * x_t1 + ((torch.ones(x_t1.shape).to(device) - mask1) * x_t2)
            else:
                imgf = x_t2 + mask2 * x_t1
        elif fusion_type =="spect":
            if check==0:
                imgf = mask1 * x_t1 + ((torch.ones(x_t1.shape).to(device) - mask1) * x_t2)
            elif check==1:
                imgf = x_t2 + mask2 * x_t1
            else:
                imgf = x_t2 - mask2 * x_t1
        elif fusion_type =="gfp":
            if check==0:
                imgf = mask1 * x_t1 + ((torch.ones(x_t1.shape).to(device) - mask1) * x_t2)
            else:
                imgf = x_t2 + mask2 * x_t1
        elif fusion_type =="ct":
            if check==0:
                imgf = mask1 * x_t1 + ((torch.ones(x_t1.shape).to(device) - mask1) * x_t2)
            else:
                imgf = x_t2 - mask2 * x_t1
        x_t = (imgf + x_t)/2
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        x_recon1 = self.predict_start_from_noise(x_t1, t, pred_noise)
        x_recon2 = self.predict_start_from_noise(x_t2, t, pred_noise)
        DVField_recon = self.predict_start_from_noise(DVField_t, t, torch.cat((pred_noise,pred_noise),dim=1))

        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
            x_recon1 = torch.clamp(x_recon1, min=-1., max=1.)
            x_recon2 = torch.clamp(x_recon2, min=-1., max=1.)
            DVField_recon = torch.clamp(DVField_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_t, t) # Here the input is imgf
        model_mean1, posterior_variance, posterior_log_variance1 = self.q_posterior_mean_variance(x_recon1, x_t1, t)
        model_mean2, posterior_variance, posterior_log_variance2 = self.q_posterior_mean_variance(x_recon2, x_t2, t)
        model_meanDVField, posterior_variance, posterior_log_varianceDVField = self.q_posterior_mean_variance(DVField_recon, x_t2, t)
        return model_mean, posterior_log_variance, model_mean1, posterior_log_variance1, model_mean2, posterior_log_variance2, model_meanDVField, posterior_log_varianceDVField
    @torch.no_grad()
    def p_sample(self, model, sourceImg1, sourceImg2, x_t1, x_t2, x_t, t, concat_type, add_noise, fusion_type = "spect", check = 0,clip_denoised=True, DVField_t = None):
        # predict mean and variance
        model_mean, model_log_variance, model_mean1, model_log_variance1, model_mean2, model_log_variance2 , model_meanDVField, model_log_varianceDVField = self.p_mean_variance(
            model, sourceImg1, sourceImg2, x_t1, x_t2, x_t, t, concat_type, clip_denoised=clip_denoised, fusion_type = fusion_type, check = check, DVField_t = DVField_t)
        if add_noise:
            noise = torch.randn_like(x_t1)
            # no noise when t == 0
            nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t1.shape) - 1))))
            # Compute x_{t-1}
            pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
            pred_img1 = model_mean1 + nonzero_mask * (0.5 * model_log_variance).exp() * noise
            pred_img2 = model_mean2 + nonzero_mask * (0.5 * model_log_variance).exp() * noise
            pred_DV = model_meanDVField + nonzero_mask * (0.5 * model_log_variance).exp() * noise
            return pred_img, pred_img1, pred_img2, pred_DV
        else:
            return model_mean, model_mean1, model_mean2, model_meanDVField
    # denoise: reverse diffusion
    @torch.no_grad()
    def p_sample_loop(self, model, sourceImg1, sourceImg2, concat_type, model_name, model_path, add_noise, timestr, log_info, dataset_name, fusion_type, check):
        step, valid_step_sum, num, generat_imgs_num = log_info
        log_step = 100
        # Start from pure noise (for each example in the batch)
        img1 = torch.randn(sourceImg1.shape, device=device)
        img2 = torch.randn(sourceImg1.shape, device=device)
        imgs = torch.randn(sourceImg1.shape, device=device)
        # reverse process
        sourceImg2_c =sourceImg2
        sourceImg2_ori =sourceImg2
        if fusion_type == "spect":
            if check ==1:
                sourceImg2_flat = sourceImg2_c.view(-1)
                sorted_values, _ = torch.sort(sourceImg2_flat) 
                threshold_index = int(0.95 * len(sorted_values)) 
                threshold = sorted_values[threshold_index] 
                boolv = (sourceImg2_c >= threshold) # shape: [8, 1, 256, 256]
            if check ==2:
                sourceImg2_flat = sourceImg2_c.view(-1)
                sorted_values, _ = torch.sort(sourceImg2_flat)
                threshold_index = int(0.07 * len(sorted_values)) 
                threshold = sorted_values[threshold_index]
                boolv = (sourceImg2_c <= threshold) # shape: [8, 1, 256, 256]
        elif fusion_type == "gfp":
            if check ==1 or check==2:
                sourceImg2_flat = sourceImg2_c.view(-1)
                sorted_values, _ = torch.sort(sourceImg2_flat)
                threshold_index = int(0.065 * len(sorted_values)) 
                threshold = sorted_values[threshold_index]
                boolv = (sourceImg2_c <= threshold) # shape: [8, 1, 256, 256]
        elif fusion_type == "pet":
            if check ==1:
                ref_values = sourceImg2_c[:, 0, 0, 5]  # shape: [8]
                ref_values = ref_values.view(-1, 1, 1, 1)  # shape: [8,1,1,1]
                boolv = (sourceImg2_c >= (10*ref_values))  # shape: [8,1,256,256]
                ref_values = sourceImg2_c[:, 0, 2, 2]
            if check ==2:
                ref_values = sourceImg2_c[:, 0, 0, 5]  # shape: [8]
                ref_values = ref_values.view(-1, 1, 1, 1)  # shape: [8,1,1,1]
                boolv = (sourceImg2_c <= ((-100)*ref_values))  # shape: [8,1,256,256]

        if check:
            mean1 = sourceImg1[boolv.bool()].mean()
            std1 = sourceImg1[boolv.bool()].std()
            mean2 = sourceImg2[boolv.bool()].mean()
            std2 = sourceImg2[boolv.bool()].std()
            # Step 2
            adjusted_sourceImg1 = (sourceImg1 - mean1) / (std1 + 1e-5)  
            adjusted_sourceImg1 = adjusted_sourceImg1 * std2 + mean2
            # Step 3
            sourceImg1 = boolv * adjusted_sourceImg1
            sourceImg2 = boolv * sourceImg2
        pred_DV = torch.full((sourceImg1.shape[0],2,sourceImg1.shape[2],sourceImg1.shape[3]), 0, device=device, dtype=torch.long)
        for i in reversed(range(0, self.timesteps)):
            if i % log_step == 0:
                now_time = time.strftime('%Y%m%d_%H%M%S')
                print(f"[valid step] {int((step - 1) / sourceImg1.shape[0]) + 1}/{valid_step_sum}    "
                      f"[generate step] {num + 1}/{generat_imgs_num}    "
                      f"[reverse process] {i}/{self.timesteps}    "
                      f"[time] {now_time}")
            t = torch.full((sourceImg1.shape[0],), i, device=device, dtype=torch.long)
            # if check==0:
            if i < self.timesteps//3:
                imgs, img1, img2, pred_DV = self.p_sample(model, sourceImg1, sourceImg1, img1, img2, imgs, t, concat_type, add_noise, fusion_type, check, DVField_t = pred_DV)# 最开始
            elif i > self.timesteps-self.timesteps//3:
                imgs, img1, img2, pred_DV = self.p_sample(model, sourceImg2, sourceImg2, img1, img2, imgs, t, concat_type, add_noise, fusion_type, check, DVField_t = pred_DV)
            else:
                imgs, img1, img2, pred_DV = self.p_sample(model, sourceImg1, sourceImg2, img1, img2, imgs, t, concat_type, add_noise, fusion_type, check, DVField_t = pred_DV)
            if fusion_type == "spect":
                if check ==1:
                    ref_values = sourceImg2_ori[:, 0, 2, 2]
                    imgs = normalize_with_ref_distribution(imgs * boolv, sourceImg2_ori * boolv, ref_values)
                    imgs = torch.clamp(imgs, min=-1.0, max=1.0)
                    imgs = (sourceImg2_ori * (~boolv))+ (imgs * boolv)
                if check ==2:
                    ref_values = sourceImg2_ori[:, 0, 2, 2]
                    imgs = normalize_with_ref_distribution(imgs * boolv, sourceImg2_ori * boolv, ref_values)
                    imgs = torch.clamp(imgs, min=-1.0, max=1.0)
                    imgs = (sourceImg2_ori * (~boolv))+ (imgs * boolv)
            elif fusion_type == "gfp":
                if check ==1 or check==2:
                    ref_values = sourceImg2_ori[:, 0, 2, 2]
                    imgs = normalize_with_ref_distribution(imgs * boolv, sourceImg2_ori * boolv, ref_values)
                    imgs = torch.clamp(imgs, min=-1.0, max=1.0)
                    imgs = (sourceImg2_ori * (~boolv))+ (imgs * boolv)
            elif fusion_type == "pet":
                if check ==1:
                    ref_values = sourceImg2_ori[:, 0, 2, 2]
                    imgs = (sourceImg2_ori * (~boolv))+ (imgs * boolv)*(-1) + (0.8 * boolv)
                    imgs = torch.clamp(imgs, -1.0, 1.0)
                    img_transformed = imgs
                if check ==2:
                    ref_values = sourceImg2_ori[:, 0, 2, 2]
                    imgs = normalize_with_ref_distribution(imgs * boolv, sourceImg2_ori * boolv, ref_values)
                    imgs = torch.clamp(imgs, min=-1.0, max=1.0)
                    imgs = (sourceImg2_ori * (~boolv))+ (imgs * boolv)
                    img_transformed = imgs
            for j in range(imgs.shape[0]):
                img_id = step + j
                dirPath = os.path.join("generate_imgs",
                                       dataset_name,
                                       timestr,
                                       model_name,
                                       )
                if not check:
                    image = tensor2img(imgs[j])[:, :, np.newaxis]
                else:
                    image = tensor2img(img_transformed[j])[:, :, np.newaxis]
                image = image.astype(np.uint8)
                extension_list = ["jpg", "tif", "png", "jpeg"]
                for extension in extension_list:
                    subdirPath = os.path.join(dirPath, extension + "_imgs")
                    if not os.path.exists(subdirPath):
                        os.makedirs(subdirPath)

                    # valid log
                    valid_log_path = os.path.join(subdirPath, "valid_log.txt")
                    valid_log = open(valid_log_path, "w")
                    valid_log.write(f"time: {timestr} \n")
                    valid_log.write(f"model_path: {model_path} \n")
                    # Save imgs
                    if generat_imgs_num == 1:
                        if img_id < 10:
                            img_file_path = os.path.join(subdirPath,
                                                         dataset_name + "_"+ str(
                                                              check)+ "_0" + str(
                                                             img_id) + "_" + str(i) + "." + extension)
                        else:
                            img_file_path = os.path.join(subdirPath,
                                                         dataset_name + "_" + str(
                                                             img_id) + "_" + str(i) + "." + extension)
                        cv2.imwrite(img_file_path, image)
                    else:
                        if img_id < 10:
                            img_file_path = os.path.join(subdirPath,
                                                         dataset_name + "_"+ str(
                                                              check)+ "_0" + str(
                                                             img_id) + "_num" + str(
                                                             num) + "_" + str(i) + "." + extension)
                        else:
                            img_file_path = os.path.join(subdirPath,
                                                         dataset_name + "_"+ str(
                                                              check)+ "_" + str(
                                                             img_id) + "_num" + str(
                                                             num) + "_" + str(i) + "." + extension)
                        cv2.imwrite(img_file_path, image)
        return imgs
    # Sample new images
    @torch.no_grad()    
    def sample(self, model, sourceImg1, sourceImg2, add_noise, concat_type, model_name, model_path,
               generat_imgs_num, step, timestr, valid_step_sum, dataset_name,fusion_type):
        extension_list = ["jpg", "tif", "png", "jpeg"]
        for num in range(generat_imgs_num):
            log_info = [step, valid_step_sum, num, generat_imgs_num]
            if fusion_type == "spect" or"pet" or "gfp":
                imgs = self.p_sample_loop(model, sourceImg1, sourceImg2[: ,0 ,: , :].unsqueeze(1), concat_type, model_name, model_path, add_noise, timestr, log_info, dataset_name, fusion_type, check =0)
                imgs2 = self.p_sample_loop(model, sourceImg1, sourceImg2[: ,1 ,: , :].unsqueeze(1), concat_type, model_name, model_path, add_noise, timestr, log_info, dataset_name, fusion_type, check =1)
                imgs3 = self.p_sample_loop(model, sourceImg1, sourceImg2[: ,2 ,: , :].unsqueeze(1), concat_type, model_name, model_path, add_noise, timestr, log_info, dataset_name, fusion_type, check =2)
            else:
                imgs = self.p_sample_loop(model, sourceImg1, sourceImg2[: ,0 ,: , :].unsqueeze(1), concat_type, model_name, model_path, add_noise, timestr, log_info, dataset_name, fusion_type, check =0)
                imgs2 = sourceImg2[: ,1 ,: , :].unsqueeze(1)
                imgs3 = sourceImg2[: ,2 ,: , :].unsqueeze(1)
            print(imgs.shape)
            for i in range(imgs.shape[0]):
                img_id = step + i
                dirPath = os.path.join("generate_imgs",
                                       dataset_name,
                                       timestr,
                                       model_name,
                                       )

                # Save images in multiple formats
                image1 = tensor2img(imgs[i])
                image2 = tensor2img(imgs2[i])
                image3 = tensor2img(imgs3[i])
                # ********* convert Ycbcr to RGB **************
                img_cr = sourceImg2[: ,1 ,: , :]
                img_cb = sourceImg2[: ,2 ,: , :]
                img_cr = img_cr.cpu().numpy().squeeze()
                img_cb = img_cb.cpu().numpy().squeeze()
                img1 = image1[:, :, np.newaxis]
                img2 = image2[:, :, np.newaxis]
                img3 = image3[:, :, np.newaxis]
                img_cr = img_cr[:, :, np.newaxis].astype(np.uint8)
                img_cb = img_cb[:, :, np.newaxis].astype(np.uint8)
                imgs = np.concatenate((img1, img2, img3), axis=2)
                image = cv2.cvtColor(imgs, cv2.COLOR_YCR_CB2BGR)
                image = image.astype(np.uint8)
                cv2.imwrite("image2_gray.png", image2.astype(np.uint8))
                cv2.imwrite("image3_gray.png", image3.astype(np.uint8))
                for extension in extension_list:
                    subdirPath = os.path.join(dirPath, extension + "_imgs")
                    if not os.path.exists(subdirPath):
                        os.makedirs(subdirPath)

                    # valid log
                    valid_log_path = os.path.join(subdirPath, "valid_log.txt")
                    valid_log = open(valid_log_path, "w")
                    valid_log.write(f"time: {timestr} \n")
                    valid_log.write(f"model_path: {model_path} \n")

                    # Save imgs
                    if generat_imgs_num == 1:
                        if img_id < 10:
                            img_file_path1 = os.path.join(subdirPath,
                                                         dataset_name + "_0" + str(
                                                             img_id)+"_1" + "." + extension)
                            img_file_path2 = os.path.join(subdirPath,
                                                         dataset_name + "_0" + str(
                                                             img_id)+"_2" + "." + extension)
                            img_file_path3 = os.path.join(subdirPath,
                                                         dataset_name + "_0" + str(
                                                             img_id)+"_3" + "." + extension)
                        else:
                            img_file_path1 = os.path.join(subdirPath,
                                                         dataset_name + "_" + str(
                                                             img_id)+"_1" + "." + extension)
                            img_file_path2 = os.path.join(subdirPath,
                                                         dataset_name + "_" + str(
                                                             img_id)+"_2" + "." + extension)
                            img_file_path3 = os.path.join(subdirPath,
                                                         dataset_name + "_" + str(
                                                             img_id)+"_3" + "." + extension)
                        cv2.imwrite(img_file_path1, image)
                        cv2.imwrite(img_file_path2, image2)
                        cv2.imwrite(img_file_path3, image3)
                        
                    else:
                        if img_id < 10:
                            img_file_path = os.path.join(subdirPath,
                                                         dataset_name + "_0" + str(
                                                             img_id) + "_num" + str(
                                                             num) + "." + extension)
                        else:
                            img_file_path = os.path.join(subdirPath,
                                                         dataset_name + "_" + str(
                                                             img_id) + "_num" + str(
                                                             num) + "." + extension)
                        cv2.imwrite(img_file_path, image)
    def train_losses(self, model, sourceImg1, sourceImg2, t, loss_scale, fusion_type, pick =0):
        noise = torch.randn_like(sourceImg1)
        sourceImg2_ori = sourceImg2
        sourceImg1_ori = sourceImg1
        if fusion_type == "adni":
            if pick==1 :
                sourceImg2_ori = sourceImg2
                sourceImg1_ori = sourceImg1
                sourceImg2_flat = sourceImg2.view(-1)
                sorted_values, _ = torch.sort(sourceImg2_flat)
                threshold_index = int(0.7 * len(sorted_values)) 
                threshold = sorted_values[threshold_index] 
                boolv = (sourceImg2 >= threshold)
                sourceImg2 = boolv * sourceImg2
                sourceImg1 = boolv * sourceImg1
                mean1 = sourceImg1[boolv.bool()].mean()
                std1 = sourceImg1[boolv.bool()].std()
                mean2 = sourceImg2[boolv.bool()].mean()
                std2 = sourceImg2[boolv.bool()].std()
                adjusted_sourceImg1 = (sourceImg1 - mean1) / (std1 + 1e-5)  
                adjusted_sourceImg1 = adjusted_sourceImg1 * std2 + mean2 
                sourceImg1 = boolv * adjusted_sourceImg1
                x1_noisy, x2_noisy = self.q_sample(sourceImg1, sourceImg2, t, noise=noise)
            elif pick==2:
                sourceImg2_ori = sourceImg2
                sourceImg1_ori = sourceImg1
                sourceImg2_flat = sourceImg2.view(-1)
                sorted_values, _ = torch.sort(sourceImg2_flat) 
                threshold_index = int(0.3 * len(sorted_values))  
                threshold = sorted_values[threshold_index]
                boolv = (sourceImg2 <= threshold) 
                sourceImg2 = boolv * sourceImg2
                sourceImg1 = boolv * sourceImg1
                mean1 = sourceImg1[boolv.bool()].mean()
                std1 = sourceImg1[boolv.bool()].std()
                mean2 = sourceImg2[boolv.bool()].mean()
                std2 = sourceImg2[boolv.bool()].std()
                adjusted_sourceImg1 = (sourceImg1 - mean1) / (std1 + 1e-5)
                adjusted_sourceImg1 = adjusted_sourceImg1 * std2 + mean2
                sourceImg1 = boolv * adjusted_sourceImg1
                x1_noisy, x2_noisy = self.q_sample(sourceImg1, sourceImg2, t, noise=noise)
            else:
                x1_noisy, x2_noisy = self.q_sample(sourceImg1, sourceImg2, t, noise=noise)
            input = torch.cat([sourceImg1, sourceImg2, x1_noisy, x2_noisy], dim=1)
            predicted_noise, mask1, mask2 , DVField= model(input, t)
            sourceImg2_ori = apply_deformation_train(sourceImg2_ori, DVField, save_dir=False)
            noise_loss = loss_scale * torch.norm(noise-predicted_noise, p=1) /(256*256)
            if pick==1:
                imgf = x2_noisy + mask2 * x1_noisy
                fusion_loss =  ( torch.norm((imgf * boolv- mask2 * x1_noisy * boolv), p=1))/(256*256)
                f0 = sourceImg2 + sourceImg1 * boolv * mask2 
                ref_values = f0[:, 0, 2, 2]
                f0 = normalize_with_ref_distribution(f0, sourceImg2 * boolv, ref_values)
                f0 = torch.clamp(f0, min=-1.0, max=1.0)
                f0 = f0 * boolv + sourceImg2_ori *( ~boolv)
                quality_loss = self.fusion_loss( mask2 * sourceImg1 * boolv, mask2 * sourceImg1 * boolv, f0 * boolv)# TODO 可以再试试sourceImg2
                mask1= mask2
            elif pick==2:
                imgf = x2_noisy - mask2 * x1_noisy
                fusion_loss =  ( torch.norm((imgf * boolv- mask2 * x1_noisy * boolv), p=1))/(256*256)
                f0 =sourceImg2 - sourceImg1 * boolv * mask2 
                ref_values = f0[:, 0, 2, 2]
                f0 = normalize_with_ref_distribution(f0, sourceImg2 * boolv, ref_values)
                f0 = torch.clamp(f0, min=-1.0, max=1.0)
                f0 = f0 *boolv+sourceImg2_ori *( ~boolv)
                quality_loss = self.fusion_loss( mask2 * sourceImg1 * boolv, mask2 * sourceImg1 * boolv, f0 * boolv)# TODO 可以再试试sourceImg2
                mask1= mask2
            else:
                sourceImg2_ori = sourceImg2
                sourceImg1_ori = sourceImg1
                imgf = mask1 * x1_noisy + ((torch.ones(mask1.shape).to(device) - mask1) * x2_noisy)
                fusion_loss = (torch.norm(imgf-x1_noisy, p=1) + torch.norm(imgf-x2_noisy, p=1)) / (256*256)
                f0 = mask1 * sourceImg1 + (1 - mask1) * sourceImg2
                quality_loss = self.fusion_loss(sourceImg2, sourceImg1, f0)
            mutual_information = mutual_information_loss(sourceImg1_ori, sourceImg2_ori)
            variance_ratio = variance_ratio_loss(sourceImg1_ori, sourceImg2_ori)
            assert predicted_noise.shape == noise.shape
            return noise_loss, mutual_information, variance_ratio, fusion_loss, quality_loss, f0, mask1.mean(), DVField, sourceImg2_ori #可以加入KL散度损失
        elif fusion_type == "spect":
            if pick==1 :
                sourceImg2_ori = sourceImg2
                sourceImg1_ori = sourceImg1
                sourceImg2_flat = sourceImg2.view(-1)
                sorted_values, _ = torch.sort(sourceImg2_flat)
                threshold_index = int(0.7 * len(sorted_values))  
                threshold = sorted_values[threshold_index]  
                boolv = (sourceImg2 >= threshold) 
                sourceImg2 = boolv * sourceImg2
                sourceImg1 = boolv * sourceImg1
                mean1 = sourceImg1[boolv.bool()].mean()
                std1 = sourceImg1[boolv.bool()].std()
                mean2 = sourceImg2[boolv.bool()].mean()
                std2 = sourceImg2[boolv.bool()].std()
                adjusted_sourceImg1 = (sourceImg1 - mean1) / (std1 + 1e-5) 
                adjusted_sourceImg1 = adjusted_sourceImg1 * std2 + mean2  
                sourceImg1 = boolv * adjusted_sourceImg1
                x1_noisy, x2_noisy = self.q_sample(sourceImg1, sourceImg2, t, noise=noise)
            elif pick==2:
                sourceImg2_ori = sourceImg2
                sourceImg1_ori = sourceImg1
                sourceImg2_flat = sourceImg2.view(-1)
                sorted_values, _ = torch.sort(sourceImg2_flat)  
                threshold_index = int(0.3 * len(sorted_values))  
                threshold = sorted_values[threshold_index]  
                boolv = (sourceImg2 <= threshold) 
                sourceImg2 = boolv * sourceImg2
                sourceImg1 = boolv * sourceImg1
                mean1 = sourceImg1[boolv.bool()].mean()
                std1 = sourceImg1[boolv.bool()].std()
                mean2 = sourceImg2[boolv.bool()].mean()
                std2 = sourceImg2[boolv.bool()].std()
                adjusted_sourceImg1 = (sourceImg1 - mean1) / (std1 + 1e-5) 
                adjusted_sourceImg1 = adjusted_sourceImg1 * std2 + mean2   
                sourceImg1 = boolv * adjusted_sourceImg1
                x1_noisy, x2_noisy = self.q_sample(sourceImg1, sourceImg2, t, noise=noise)
            else:
                x1_noisy, x2_noisy = self.q_sample(sourceImg1, sourceImg2, t, noise=noise)
            input = torch.cat([sourceImg1, sourceImg2, x1_noisy, x2_noisy], dim=1)
            predicted_noise, mask1, mask2 , DVField= model(input, t)
            sourceImg2_ori = apply_deformation_train(sourceImg2_ori, DVField, save_dir=False)
            noise_loss = loss_scale * torch.norm(noise-predicted_noise, p=1) /(256*256)
            if pick==1:
                imgf = x2_noisy + mask2 * x1_noisy
                fusion_loss =  ( torch.norm((imgf * boolv- mask2 * x1_noisy * boolv), p=1))/(256*256)
                f0 = sourceImg2 + sourceImg1 * boolv * mask2 
                ref_values = f0[:, 0, 2, 2]
                f0 = normalize_with_ref_distribution(f0, sourceImg2 * boolv, ref_values)
                f0 = torch.clamp(f0, min=-1.0, max=1.0)
                f0 = f0 * boolv + sourceImg2_ori *( ~boolv)
                quality_loss = self.fusion_loss( mask2 * sourceImg1 * boolv, mask2 * sourceImg1 * boolv, f0 * boolv)
                mask1= mask2
            elif pick==2:
                imgf = x2_noisy - mask2 * x1_noisy
                fusion_loss =  ( torch.norm((imgf * boolv- mask2 * x1_noisy * boolv), p=1))/(256*256)
                f0 =sourceImg2 - sourceImg1 * boolv * mask2 
                ref_values = f0[:, 0, 2, 2]
                f0 = normalize_with_ref_distribution(f0, sourceImg2 * boolv, ref_values)
                f0 = torch.clamp(f0, min=-1.0, max=1.0)
                f0 = f0 *boolv+sourceImg2_ori *( ~boolv)
                quality_loss = self.fusion_loss( mask2 * sourceImg1 * boolv, mask2 * sourceImg1 * boolv, f0 * boolv)
                mask1= mask2
            else:
                sourceImg2_ori = sourceImg2
                sourceImg1_ori = sourceImg1
                imgf = mask1 * x1_noisy + ((torch.ones(mask1.shape).to(device) - mask1) * x2_noisy)
                fusion_loss = (torch.norm(imgf-x1_noisy, p=1) + torch.norm(imgf-x2_noisy, p=1)) / (256*256)
                f0 = mask1 * sourceImg1 + (1 - mask1) * sourceImg2
                quality_loss = self.fusion_loss(sourceImg2, sourceImg1, f0)
            mutual_information = mutual_information_loss(sourceImg1_ori, sourceImg2_ori)
            variance_ratio = variance_ratio_loss(sourceImg1_ori, sourceImg2_ori)
            assert predicted_noise.shape == noise.shape
            return noise_loss, mutual_information, variance_ratio, fusion_loss, quality_loss, f0, mask1.mean(), DVField, sourceImg2_ori
        elif fusion_type == "ct":
            if pick==2 :
                sourceImg2_ori = sourceImg2
                sourceImg1_ori = sourceImg1
                sourceImg2_flat = sourceImg2.view(-1)
                sorted_values, _ = torch.sort(sourceImg2_flat)  
                threshold_index = int(0.85 * len(sorted_values))  
                threshold = sorted_values[threshold_index]  
                boolv = (sourceImg2 >= threshold) 
                sourceImg2 = boolv * sourceImg2
                sourceImg1 = boolv * sourceImg1
                mean1 = sourceImg1[boolv.bool()].mean()
                std1 = sourceImg1[boolv.bool()].std()

                mean2 = sourceImg2[boolv.bool()].mean()
                std2 = sourceImg2[boolv.bool()].std()
                adjusted_sourceImg1 = (sourceImg1 - mean1) / (std1 + 1e-5)
                adjusted_sourceImg1 = adjusted_sourceImg1 * std2 + mean2 
                sourceImg1 = boolv * adjusted_sourceImg1
                x1_noisy, x2_noisy = self.q_sample(sourceImg1, sourceImg2, t, noise=noise)
            elif pick==1:
                sourceImg2_ori = sourceImg2
                sourceImg1_ori = sourceImg1
                sourceImg2_flat = sourceImg2.view(-1)
                sorted_values, _ = torch.sort(sourceImg2_flat)
                threshold_index = int(0.15 * len(sorted_values)) 
                threshold = sorted_values[threshold_index] 
                boolv = (sourceImg2 <= threshold) # shape: [8, 1, 256, 256]
                sourceImg2 = boolv * sourceImg2
                sourceImg1 = boolv * sourceImg1
                mean1 = sourceImg1[boolv.bool()].mean()
                std1 = sourceImg1[boolv.bool()].std()
                mean2 = sourceImg2[boolv.bool()].mean()
                std2 = sourceImg2[boolv.bool()].std()
                adjusted_sourceImg1 = (sourceImg1 - mean1) / (std1 + 1e-5)  
                adjusted_sourceImg1 = adjusted_sourceImg1 * std2 + mean2   
                sourceImg1 = boolv * adjusted_sourceImg1
                x1_noisy, x2_noisy = self.q_sample(sourceImg1, sourceImg2, t, noise=noise)
            else:
                x1_noisy, x2_noisy = self.q_sample(sourceImg1, sourceImg2, t, noise=noise)
            input = torch.cat([sourceImg1, sourceImg2, x1_noisy, x2_noisy], dim=1)
            predicted_noise, mask1, mask2 , DVField= model(input, t)
            sourceImg2_ori = apply_deformation_train(sourceImg2_ori, DVField, save_dir=False)
            noise_loss = loss_scale * torch.norm(noise-predicted_noise, p=1) /(256*256)
            if pick==2:
                imgf = x2_noisy + mask2 * x1_noisy
                fusion_loss =  ( torch.norm((imgf * boolv- mask2 * x1_noisy * boolv), p=1))/(256*256)
                f0 = sourceImg2 + sourceImg1 * boolv * mask2 
                ref_values = f0[:, 0, 2, 2]
                f0 = normalize_with_ref_distribution(f0, sourceImg2 * boolv, ref_values)
                f0 = torch.clamp(f0, min=-1.0, max=1.0)
                f0 = f0 * boolv + sourceImg2_ori *( ~boolv)
                quality_loss = self.fusion_loss( mask2 * sourceImg1 * boolv, mask2 * sourceImg1 * boolv, f0 * boolv)
                mask1= mask2
            elif pick==1:
                imgf = x2_noisy - mask2 * x1_noisy
                fusion_loss =  ( torch.norm((imgf * boolv- mask2 * x1_noisy * boolv), p=1))/(256*256)
                f0 =sourceImg2 - sourceImg1 * boolv * mask2 
                ref_values = f0[:, 0, 2, 2]
                f0 = normalize_with_ref_distribution(f0, sourceImg2 * boolv, ref_values)
                f0 = torch.clamp(f0, min=-1.0, max=1.0)
                f0 = f0 *boolv+sourceImg2_ori *( ~boolv)
                quality_loss = self.fusion_loss( mask2 * sourceImg1 * boolv, mask2 * sourceImg1 * boolv, f0 * boolv)
                mask1= mask2
            else:
                sourceImg2_ori = sourceImg2
                sourceImg1_ori = sourceImg1
                imgf = mask1 * x1_noisy + ((torch.ones(mask1.shape).to(device) - mask1) * x2_noisy)
                fusion_loss = (torch.norm(imgf-x1_noisy, p=1) + torch.norm(imgf-x2_noisy, p=1)) / (256*256)
                f0 = mask1 * sourceImg1 + (1 - mask1) * sourceImg2
                quality_loss = self.fusion_loss(sourceImg2, sourceImg1, f0)
            mutual_information = mutual_information_loss(sourceImg1_ori, sourceImg2_ori)
            variance_ratio = variance_ratio_loss(sourceImg1_ori, sourceImg2_ori)
            assert predicted_noise.shape == noise.shape
            
            return noise_loss, mutual_information, variance_ratio, fusion_loss, quality_loss, f0, mask1.mean(), DVField, sourceImg2_ori
        elif fusion_type=="gfp":
            if pick==1 :
                sourceImg2_ori = sourceImg2
                sourceImg1_ori = sourceImg1
                sourceImg2_flat = sourceImg2.view(-1)
                sorted_values, _ = torch.sort(sourceImg2_flat) 
                threshold_index = int(0.07 * len(sorted_values)) 
                threshold = sorted_values[threshold_index]  
                boolv = (sourceImg2 <= threshold) # shape: [8, 1, 256, 256]
                sourceImg2 = boolv * sourceImg2
                sourceImg1 = boolv * sourceImg1
                mean1 = sourceImg1[boolv.bool()].mean()
                std1 = sourceImg1[boolv.bool()].std()
                mean2 = sourceImg2[boolv.bool()].mean()
                std2 = sourceImg2[boolv.bool()].std()
                adjusted_sourceImg1 = (sourceImg1 - mean1) / (std1 + 1e-5) 
                adjusted_sourceImg1 = adjusted_sourceImg1 * std2 + mean2
                sourceImg1 = boolv * adjusted_sourceImg1
                x1_noisy, x2_noisy = self.q_sample(sourceImg1, sourceImg2, t, noise=noise)
            elif pick==2:
                sourceImg2_ori = sourceImg2
                sourceImg1_ori = sourceImg1
                sourceImg2_flat = sourceImg2.view(-1)
                sorted_values, _ = torch.sort(sourceImg2_flat) 
                threshold_index = int(0.07 * len(sorted_values))  
                threshold = sorted_values[threshold_index]  
                boolv = (sourceImg2 <= threshold)
                sourceImg2 = boolv * sourceImg2
                sourceImg1 = boolv * sourceImg1
                mean1 = sourceImg1[boolv.bool()].mean()
                std1 = sourceImg1[boolv.bool()].std()

                mean2 = sourceImg2[boolv.bool()].mean()
                std2 = sourceImg2[boolv.bool()].std()
                adjusted_sourceImg1 = (sourceImg1 - mean1) / (std1 + 1e-5) 
                adjusted_sourceImg1 = adjusted_sourceImg1 * std2 + mean2 
                sourceImg1 = boolv * adjusted_sourceImg1
                x1_noisy, x2_noisy = self.q_sample(sourceImg1, sourceImg2, t, noise=noise)
            else:
                x1_noisy, x2_noisy = self.q_sample(sourceImg1, sourceImg2, t, noise=noise)
            input = torch.cat([sourceImg1, sourceImg2, x1_noisy, x2_noisy], dim=1)
            predicted_noise, mask1, mask2 , DVField= model(input, t)
            sourceImg2_ori = apply_deformation_train(sourceImg2_ori, DVField, save_dir=False)
            noise_loss = loss_scale * torch.norm(noise-predicted_noise, p=1) /(256*256)
            if pick:
                imgf = x2_noisy - mask2 * x1_noisy * 0.45
                fusion_loss =  ( torch.norm((imgf * boolv- mask2 * x1_noisy * boolv), p=1))/(256*256)
                f0 =sourceImg2 - sourceImg1 * boolv * mask2 *0.45
                ref_values = f0[:, 0, 2, 2]
                f0 = normalize_with_ref_distribution(f0, sourceImg2 * boolv, ref_values)
                f0 = torch.clamp(f0, min=-1.0, max=1.0)
                f0 = f0 * boolv + sourceImg2_ori *( ~boolv)
                quality_loss = self.fusion_loss( mask2 * sourceImg1 * boolv, mask2 * sourceImg1 * boolv, f0 * boolv)
                mask1= mask2
            else:
                sourceImg2_ori = sourceImg2
                sourceImg1_ori = sourceImg1
                imgf = mask1 * x1_noisy + ((torch.ones(mask1.shape).to(device) - mask1) * x2_noisy)
                fusion_loss = (torch.norm(imgf-x1_noisy, p=1) + torch.norm(imgf-x2_noisy, p=1)) / (256*256)
                f0 = mask1 * sourceImg1 + (1 - mask1) * sourceImg2
                quality_loss = self.fusion_loss(sourceImg2, sourceImg1, f0)
            mutual_information = mutual_information_loss(sourceImg1_ori, sourceImg2_ori)
            variance_ratio = variance_ratio_loss(sourceImg1_ori, sourceImg2_ori)
            assert predicted_noise.shape == noise.shape
            return noise_loss, mutual_information, variance_ratio, fusion_loss, quality_loss, f0, mask1.mean(), DVField, sourceImg2_ori
        elif fusion_type=="pet":
            if pick==1 :
                ref_values = sourceImg2[:, 0, 0, 5]  
                sourceImg2_ori = sourceImg2
                sourceImg1_ori = sourceImg1
                ref_values = ref_values.view(-1, 1, 1, 1)  # shape: [8,1,1,1]
                boolv = (sourceImg2 >= (10*ref_values))  # shape: [8,1,256,256]
                sourceImg2 = boolv * sourceImg2
                sourceImg1 = boolv * sourceImg1
                mean1 = sourceImg1[boolv.bool()].mean()
                std1 = sourceImg1[boolv.bool()].std()
                mean2 = sourceImg2[boolv.bool()].mean()
                std2 = sourceImg2[boolv.bool()].std()
                adjusted_sourceImg1 = (sourceImg1 - mean1) / (std1 + 1e-5) 
                adjusted_sourceImg1 = adjusted_sourceImg1 * std2 + mean2 
                sourceImg1 = boolv * adjusted_sourceImg1
                x1_noisy, x2_noisy = self.q_sample(sourceImg1, sourceImg2, t, noise=noise)
            elif pick==2:
                ref_values = sourceImg2[:, 0, 0, 5]  # shape: [8]
                sourceImg2_ori = sourceImg2
                sourceImg1_ori = sourceImg1
                ref_values = ref_values.view(-1, 1, 1, 1)  # shape: [8,1,1,1]
                boolv = (sourceImg2 <= ((-100)*ref_values))  # shape: [8,1,256,256]
                sourceImg2 = boolv * sourceImg2
                sourceImg1 = boolv * sourceImg1
                mean1 = sourceImg1[boolv.bool()].mean()
                std1 = sourceImg1[boolv.bool()].std()
                mean2 = sourceImg2[boolv.bool()].mean()
                std2 = sourceImg2[boolv.bool()].std()
                adjusted_sourceImg1 = (sourceImg1 - mean1) / (std1 + 1e-5) 
                adjusted_sourceImg1 = adjusted_sourceImg1 * std2 + mean2   
                sourceImg1 = boolv * adjusted_sourceImg1
                x1_noisy, x2_noisy = self.q_sample(sourceImg1, sourceImg2, t, noise=noise)
            else:
                x1_noisy, x2_noisy = self.q_sample(sourceImg1, sourceImg2, t, noise=noise)
            input = torch.cat([sourceImg1, sourceImg2, x1_noisy, x2_noisy], dim=1)
            predicted_noise, mask1, mask2 , DVField= model(input, t)
            sourceImg2_ori = apply_deformation_train(sourceImg2_ori, DVField, save_dir=False)
            noise_loss = loss_scale * torch.norm(noise-predicted_noise, p=1) /(256*256)
            if pick:
                imgf = x2_noisy - mask2 * x1_noisy
                fusion_loss =  ( torch.norm((imgf * boolv- mask2 * x1_noisy * boolv), p=1))/(256*256)
                f0 =sourceImg2 - sourceImg1 * boolv * mask2 
                ref_values = f0[:, 0, 2, 2]
                f0 = normalize_with_ref_distribution(f0, sourceImg2 * boolv, ref_values)
                f0 = torch.clamp(f0, min=-1.0, max=1.0)
                f0 = f0 * boolv + sourceImg2_ori *( ~boolv)
                quality_loss = self.fusion_loss( mask2 * sourceImg1 * boolv, mask2 * sourceImg1 * boolv, f0 * boolv)
                mask1= mask2
            else:
                imgf = mask1 * x1_noisy + ((torch.ones(mask1.shape).to(device) - mask1) * x2_noisy)
                fusion_loss = (torch.norm(imgf-x1_noisy, p=1) + torch.norm(imgf-x2_noisy, p=1)) / (256*256)
                f0 = mask1 * sourceImg1 + (1 - mask1) * sourceImg2
                quality_loss = self.fusion_loss(sourceImg2, sourceImg1, f0)
            save_dir = "./save_debug"
            os.makedirs(save_dir, exist_ok=True)
            DVField_x = DVField[..., 0]  
            DVField_y = DVField[..., 1]  
            DVField_x = (DVField_x - DVField_x.min()) / (DVField_x.max() - DVField_x.min() + 1e-8)
            DVField_y = (DVField_y - DVField_y.min()) / (DVField_y.max() - DVField_y.min() + 1e-8)
            for b in range(DVField.shape[0]):
                vutils.save_image(DVField_x[b:b+1], os.path.join(save_dir, f'step{self.global_save_counter}_b{b}_DVField_x.png'))
                vutils.save_image(DVField_y[b:b+1], os.path.join(save_dir, f'step{self.global_save_counter}_b{b}_DVField_y.png'))
            mask1_vis = (mask1 - mask1.min()) / (mask1.max() - mask1.min() + 1e-8)
            mask2_vis = (mask2 - mask2.min()) / (mask2.max() - mask2.min() + 1e-8)
            for b in range(mask1.shape[0]):
                vutils.save_image(mask1_vis[b], os.path.join(save_dir, f'step{self.global_save_counter}_b{b}_mask1.png'))
                vutils.save_image(mask2_vis[b], os.path.join(save_dir, f'step{self.global_save_counter}_b{b}_mask2.png'))
            x1_noisy_vis = (x1_noisy - x1_noisy.min()) / (x1_noisy.max() - x1_noisy.min() + 1e-8)
            x2_noisy_vis = (x2_noisy - x2_noisy.min()) / (x2_noisy.max() - x2_noisy.min() + 1e-8)
            for b in range(x1_noisy.shape[0]):
                vutils.save_image(x1_noisy_vis[b], os.path.join(save_dir, f'step{self.global_save_counter}_b{b}_x1_noisy.png'))
                vutils.save_image(x2_noisy_vis[b], os.path.join(save_dir, f'step{self.global_save_counter}_b{b}_x2_noisy.png'))
            self.global_save_counter = self.global_save_counter + 1
            mutual_information = mutual_information_loss(sourceImg1_ori, sourceImg2_ori)
            variance_ratio = variance_ratio_loss(sourceImg1_ori, sourceImg2_ori)
            assert predicted_noise.shape == noise.shape
            return noise_loss, mutual_information, variance_ratio, fusion_loss, quality_loss, f0, mask1.mean(), DVField, sourceImg2_ori
        elif fusion_type=="ct":
            x1_noisy, x2_noisy = self.q_sample(sourceImg1, sourceImg2, t, noise=noise)
            input = torch.cat([sourceImg1, sourceImg2, x1_noisy, x2_noisy], dim=1)
            predicted_noise, mask1, mask2 , DVField= model(input, t)
            sourceImg2_ori = apply_deformation_train(sourceImg2_ori, DVField, save_dir=False)
            noise_loss = loss_scale * torch.norm(noise-predicted_noise, p=1) /(256*256)
            sourceImg2_ori = sourceImg2
            sourceImg1_ori = sourceImg1
            imgf = mask1 * x1_noisy + ((torch.ones(mask1.shape).to(device) - mask1) * x2_noisy)
            fusion_loss = (torch.norm(imgf-x1_noisy, p=1) + torch.norm(imgf-x2_noisy, p=1)) / (256*256)
            f0 = mask1 * sourceImg1 + (1 - mask1) * sourceImg2
            quality_loss = self.fusion_loss(sourceImg2, sourceImg1, f0)
            mutual_information = mutual_information_loss(sourceImg1_ori, sourceImg2_ori)
            variance_ratio = variance_ratio_loss(sourceImg1_ori, sourceImg2_ori)
            assert predicted_noise.shape == noise.shape
            return noise_loss, mutual_information, variance_ratio, fusion_loss, quality_loss, f0, mask1.mean(), DVField, sourceImg2_ori 
