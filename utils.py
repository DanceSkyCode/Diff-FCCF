import numpy as np
from torchvision import transforms
import os
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import cv2
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import torch.nn.functional as F
def compute_reward(merged_losses, history, total_loss, loss_ref):
    for k in loss_ref:
        loss_ref[k] = 0.95*loss_ref[k] + 0.05*merged_losses[k].item()
    ratios = [merged_losses[k].item()/(loss_ref[k]+1e-8) for k in merged_losses]
    balance_penalty = -np.std(ratios)
    trends = list(history.get_features().values())
    trend_stability = np.mean(trends) if trends else 0.0
    reward = -np.log(total_loss + 1e-8) + 0.3*balance_penalty + 0.2*(1.0 if abs(trend_stability)<0.1 else -0.5)
    return reward
def update_loss_ref(merged_losses, loss_ref, alpha=0.95):
    for k in loss_ref:
        loss_ref[k] = alpha*loss_ref[k] + (1-alpha)*merged_losses[k].item()
def save_fusion_images(save_dir, f0, f02, f03, train_images, imgname):
    B = f0.shape[0]
    f0 = f0.detach()
    f02 = f02.detach()
    f03 = f03.detach()
    for batch_idx in range(B):
        image = tensor2img(f0[batch_idx, :, :, :])
        img_cr = tensor2img(train_images[1][batch_idx, 1, :, :].unsqueeze(0))
        img_cb = tensor2img(train_images[1][batch_idx, 2, :, :].unsqueeze(0))
        img = image[:, :, np.newaxis]
        img_cr = img_cr[:, :, np.newaxis]
        img_cb = img_cb[:, :, np.newaxis]
        image = np.concatenate((img, img_cr, img_cb), axis=2)
        image = cv2.cvtColor(image, cv2.COLOR_YCR_CB2BGR)
        cv2.imwrite(os.path.join(save_dir, f"{imgname[batch_idx]}_1.png"), image.astype(np.uint8))
        image = tensor2img(f0[batch_idx, :, :, :])
        img_cr = tensor2img(f02[batch_idx, :, :, :])
        img_cb = tensor2img(f03[batch_idx, :, :, :])
        img = image[:, :, np.newaxis]
        img_cr = img_cr[:, :, np.newaxis]
        img_cb = img_cb[:, :, np.newaxis]
        image = np.concatenate((img, img_cr, img_cb), axis=2)
        image = cv2.cvtColor(image, cv2.COLOR_YCR_CB2BGR)
        cv2.imwrite(os.path.join(save_dir, f"{imgname[batch_idx]}.png"), image.astype(np.uint8))
        cv2.imwrite(os.path.join(save_dir, f"{imgname[batch_idx]}_f01.png"), tensor2img(f0[batch_idx, :, :, :]).astype(np.uint8))
        cv2.imwrite(os.path.join(save_dir, f"{imgname[batch_idx]}_f02.png"), tensor2img(f02[batch_idx, :, :, :]).astype(np.uint8))
        cv2.imwrite(os.path.join(save_dir, f"{imgname[batch_idx]}_f03.png"), tensor2img(f03[batch_idx, :, :, :]).astype(np.uint8))
        cv2.imwrite(os.path.join(save_dir, f"{imgname[batch_idx]}_Y.png"), tensor2img(train_images[1][batch_idx, 0, :, :].unsqueeze(0)).astype(np.uint8))
        cv2.imwrite(os.path.join(save_dir, f"{imgname[batch_idx]}_cr.png"), tensor2img(train_images[1][batch_idx, 1, :, :].unsqueeze(0)).astype(np.uint8))
        cv2.imwrite(os.path.join(save_dir, f"{imgname[batch_idx]}_cb.png"), tensor2img(train_images[1][batch_idx, 2, :, :].unsqueeze(0)).astype(np.uint8))
class SpatialTransformer(torch.nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')  # add indexing for PyTorch 1.10+
        grid = torch.stack(grids)  # shape: [2, H, W]
        grid = torch.unsqueeze(grid, 0)  # shape: [1, 2, H, W]
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode
    def forward(self, src, flow):
        new_locs = self.grid + flow  
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)[..., [1, 0]] 
        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=True)
def create_grid(size):
    num1, num2 = (size[0] + 10) // 10, (size[1] + 10) // 10
    x, y = np.meshgrid(np.linspace(-2, 2, num1), np.linspace(-2, 2, num2))
    fig = plt.figure(figsize=(size[1] / 100, size[0] / 100), dpi=100)
    plt.plot(x, y, color="black")
    plt.plot(x.T, y.T, color="black")
    plt.axis('off')
    plt.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw()
    grid_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    grid_img = grid_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    grid_gray = cv2.cvtColor(grid_img, cv2.COLOR_RGB2GRAY)
    return grid_gray


def save_deformation_field_visualization(dv_field, imgid, id, imgname, img):
    for i in range(dv_field.shape[0]):
        dx = dv_field[i, ..., 0].detach().cpu().numpy()
        dy = dv_field[i, ..., 1].detach().cpu().numpy()
        dx *= 3.0
        dy *= 3.0
        dx = cv2.GaussianBlur(dx, (11, 11), 3)
        dy = cv2.GaussianBlur(dy, (11, 11), 3)
        image = img[i].detach()
        ori_img = tensor2img(image)
        ori_img = np.stack([ori_img] * 3, axis=-1)
        save_path = os.path.join(imgid, f"{imgname[2][i]}_WarpedGrid{id}_ori.png")
        cv2.imwrite(save_path, ori_img)
        H, W, _= ori_img.shape
        grid_img = ori_img.copy()
        step = 30
        for y in range(0, H, step):
            cv2.line(grid_img, (0, y), (W, y), (0, 0, 255), 2)  # 红色横线
        for x in range(0, W, step):
            cv2.line(grid_img, (x, 0), (x, H), (0, 0, 255), 2)  # 红色竖线
        grid_tensor = torch.from_numpy(grid_img).float().unsqueeze(0).permute(0, 3, 1, 2) / 255.0
        flow_tensor = torch.from_numpy(np.stack([dx, dy], axis=0)).unsqueeze(0)
        STN = SpatialTransformer(size=(H, W))
        warped_grid = STN(grid_tensor, flow_tensor)
        warped_grid = warped_grid[0].permute(1, 2, 0).detach().cpu().numpy()  # [C, H, W] -> [H, W, C]
        warped_grid = (warped_grid * 255).astype(np.uint8)
        save_path = os.path.join(imgid, f"{imgname[2][i]}_WarpedGrid{id}.png")
        cv2.imwrite(save_path, warped_grid)
def build_hyperedges(y_channel):
    segments = slic(y_channel.cpu().numpy(), 
                   n_segments=200, 
                   compactness=10)
    return torch.from_numpy(segments)
def tensor2img(img):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.cpu().numpy().squeeze().astype(np.uint8)),
    ])
    imgs = reverse_transforms(img)
    return imgs
def tensorboard_writer(timestr):
    log_path = os.path.join('logs', timestr)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    return writer
def logger(timestr):
    log_dir = os.path.join('logs',timestr)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, "log.txt")
    fw = open(log_path, "a+")
    return fw
def save_model(model, epoch,timestr):
    dir_path = os.path.join("weight",timestr)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    ckpt_name = "epoch_" + str(epoch) + ".pt"
    ckpt_path = os.path.join(dir_path, ckpt_name)
    torch.save(model.state_dict(), ckpt_path)
