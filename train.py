import time
import json
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import random
from dataset import MFI_Dataset
from Diffusion import GaussianDiffusion
from Condition_Noise_Predictor.UNet import NoisePred
from utils import tensorboard_writer, logger, save_model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_ref = {'noise': 10.0, 'folding': 1e5, 'smooth': 1e6, 'fusion': 1e3, 'quality': 1e4}  # init
def train(config_path):
    timestr = time.strftime('%Y%m%d_%H%M%S')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # ********************* load train dataset *************************
    train_datasePath = config["dataset"]["train"]["path"]
    train_phase = config["dataset"]["train"]["phase"]
    train_batch_size = config["dataset"]["train"]["batch_size"]
    train_use_dataTransform = config["dataset"]["train"]["use_dataTransform"]
    train_resize = config["dataset"]["train"]["resize"]
    train_imgSize = config["dataset"]["train"]["imgSize"]
    train_shuffle = config["dataset"]["train"]["shuffle"]
    train_drop_last = config["dataset"]["train"]["drop_last"]

    # Dataset of various fusion tasks
    # ************** Medical ****************
    pet_dataset_path = "/Dataset/Medical/Train/PET-MRI"
    train_pet_dataset = MFI_Dataset(pet_dataset_path, phase=train_phase, use_dataTransform=train_use_dataTransform,
                                resize=train_resize, imgSzie=train_imgSize, fusion_type="pet")
    train_pet_dataloader = DataLoader(train_pet_dataset, batch_size=train_batch_size, shuffle=train_shuffle,
                                  drop_last=train_drop_last)
    medical_dataset_path = "/Dataset/Medical/Train/SPECT-MRI"
    train_spect_dataset = MFI_Dataset(medical_dataset_path, phase=train_phase, use_dataTransform=train_use_dataTransform,
                                resize=train_resize, imgSzie=train_imgSize, fusion_type="spect")
    train_spect_dataloader = DataLoader(train_spect_dataset, batch_size=train_batch_size, shuffle=train_shuffle,
                                  drop_last=train_drop_last)
    medical_dataset_path = "/Dataset/Medical/Train/GFP"
    train_gfp_dataset = MFI_Dataset(medical_dataset_path, phase=train_phase, use_dataTransform=train_use_dataTransform,
                                resize=train_resize, imgSzie=train_imgSize, fusion_type="gfp")
    train_gfp_dataloader = DataLoader(train_gfp_dataset, batch_size=train_batch_size, shuffle=train_shuffle,
                                  drop_last=train_drop_last)
    medical_dataset_path = "/Dataset/Medical/Train/CT-MRI"
    train_ct_dataset = MFI_Dataset(medical_dataset_path, phase=train_phase, use_dataTransform=train_use_dataTransform,
                                resize=train_resize, imgSzie=train_imgSize, fusion_type="ct")
    train_ct_dataloader = DataLoader(train_ct_dataset, batch_size=train_batch_size, shuffle=train_shuffle,
                                  drop_last=train_drop_last)
    medical_dataset_path = "/Dataset/Medical/Train/ADNI"
    train_adni_dataset = MFI_Dataset(medical_dataset_path, phase=train_phase, use_dataTransform=train_use_dataTransform,
                                resize=train_resize, imgSzie=train_imgSize, fusion_type="adni")
    train_adni_dataloader = DataLoader(train_adni_dataset, batch_size=train_batch_size, shuffle=train_shuffle,
                                  drop_last=train_drop_last)
    in_channels = config["Condition_Noise_Predictor"]["UNet"]["in_channels"]
    out_channels = config["Condition_Noise_Predictor"]["UNet"]["out_channels"]
    model_channels = config["Condition_Noise_Predictor"]["UNet"]["model_channels"]
    num_res_blocks = config["Condition_Noise_Predictor"]["UNet"]["num_res_blocks"]
    dropout = config["Condition_Noise_Predictor"]["UNet"]["dropout"]
    time_embed_dim_mult = config["Condition_Noise_Predictor"]["UNet"]["time_embed_dim_mult"]
    down_sample_mult = config["Condition_Noise_Predictor"]["UNet"]["down_sample_mult"]
    model = NoisePred(in_channels, out_channels, model_channels, num_res_blocks, dropout, time_embed_dim_mult,
                      down_sample_mult)

    # whether to use the pre-training model
    use_preTrain_model = config["Condition_Noise_Predictor"]["use_preTrain_model"]
    if use_preTrain_model:
        preTrain_Model_path = config["Condition_Noise_Predictor"]["preTrain_Model_path"]
        model.load_state_dict(torch.load(preTrain_Model_path, map_location=device),strict=False)
        print(f"using pre-trained model:{preTrain_Model_path}")
    model = model.to(device)

    # optimizer
    init_lr = config["optimizer"]["init_lr"]
    use_lr_scheduler = config["optimizer"]["use_lr_scheduler"]
    StepLR_size = config["optimizer"]["StepLR_size"]
    StepLR_gamma = config["optimizer"]["StepLR_gamma"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)
    if use_lr_scheduler:
        learningRate_scheduler = lr_scheduler.StepLR(optimizer, step_size=StepLR_size, gamma=StepLR_gamma)

    # diffusion model
    T = config["diffusion_model"]["T"]
    beta_schedule_type = config["diffusion_model"]["beta_schedule_type"]
    loss_scale = config["diffusion_model"]["loss_scale"]
    diffusion = GaussianDiffusion(T, beta_schedule_type)

    # log
    writer = tensorboard_writer(timestr)
    log = logger(timestr)
    print(f"time: {timestr}")
    log.write(f"time: {timestr} \n")
    log.write(f"config:  \n")
    log.write(json.dumps(config, ensure_ascii=False, indent=4))
    if use_lr_scheduler:
        log.write(
            f"\n learningRate_scheduler = lr_scheduler.StepLR(optimizer, step_size={StepLR_size}, gamma={StepLR_gamma})  \n\n")

    epochs = config["hyperParameter"]["epochs"]
    start_epoch = config["hyperParameter"]["start_epoch"]
    save_model_epoch_step = config["hyperParameter"]["save_model_epoch_step"]
    num_train_step = 0
    fusion_task = ["adni", "pet", "gfp", "ct", "adni"]
    for epoch in range(start_epoch, epochs):
        # train
        model.train()
        print_loss_sum = 0
        writer.add_scalar('lr_epoch: ', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

        fusion_type = random.choice(fusion_task)
        if fusion_type == "pet":
            dataloader = train_pet_dataloader
        if fusion_type == "spect":
            dataloader = train_spect_dataloader        
        if fusion_type == "gfp":
            dataloader = train_gfp_dataloader        
        if fusion_type == "ct":
            dataloader = train_ct_dataloader       
        if fusion_type == "adni":
            dataloader = train_adni_dataloader                         
        train_step_sum = len(dataloader)
        for train_step, train_images in tqdm(enumerate(dataloader), desc="train step"):
            optimizer.zero_grad()
            train_images[2] = tuple(name.replace('.png', '') for name in train_images[2]) 
            train_sourceImg1 = train_images[0].to(device)
            train_sourceImg2 = train_images[1][: ,0 ,: , :].unsqueeze(1).to(device)
            
            t = torch.randint(0, T, (train_batch_size,), device=device).long()
            noise_loss, folding, smooth, fusion_loss, quality_loss, f0, mask1, DVField1, sourceImg2_ori1 = diffusion.train_losses(model, train_sourceImg1, train_sourceImg2, t, loss_scale, fusion_type, pick =0)
            train_sourceImg3 = train_images[1][: ,1 ,: , :].unsqueeze(1).to(device)
            t = torch.randint(0, T, (train_batch_size,), device=device).long()
            noise_loss2, folding2, smooth2, fusion_loss2, quality_loss2, f02, mask2, DVField2,sourceImg2_ori2 = diffusion.train_losses(model, train_sourceImg1, train_sourceImg3, t, loss_scale, fusion_type, pick =1)
            train_sourceImg4 = train_images[1][: ,2 ,: , :].unsqueeze(1).to(device)
            t = torch.randint(0, T, (train_batch_size,), device=device).long()
            noise_loss3, folding3, smooth3, fusion_loss3, quality_loss3, f03, mask3, DVField3, sourceImg2_ori3 = diffusion.train_losses(model, train_sourceImg1, train_sourceImg4, t, loss_scale, fusion_type, pick =2)
            loss_total = 10 * (noise_loss + noise_loss2 + noise_loss3) +1* (folding + folding2 + folding3) + 1 * (smooth + smooth2 + smooth3) + (1*fusion_loss + 1*fusion_loss2 + 1*fusion_loss3) + (10*quality_loss + 10*quality_loss2 + 10*quality_loss3)
            loss_total.backward()
            optimizer.step()

            # loss_total.backward()
            # optimizer.step()
            num_train_step += 1
            print_loss_sum += loss_total
        if fusion_type!= "abc":
            print(
                f" [epoch] {epoch}/{epochs}    "
                f"[epoch_step] {train_step}/{train_step_sum}     "
                f"[train_step] {num_train_step}     "
                f"[noise_loss1] { 1 * noise_loss.item() :.6f}     "
                f"[folding_loss1] { 1 * folding.item() :.6f}     "
                f"[smooth_loss1] {1 * smooth.item() :.6f}     "
                f"[fusion_loss1] {1 * fusion_loss.item() :.6f}     "
                f"[quality_loss1] {1 * quality_loss.item() :.6f}"
                f"[mask1] {1 * mask1 :.6f}"
                
                f"[noise_loss2] { 1 * noise_loss2.item() :.6f}     "
                f"[folding_loss2] { 1 * folding2.item() :.6f}     "
                f"[smooth_loss2] {1 * smooth2.item() :.6f}     "
                f"[fusion_loss2] {1 * fusion_loss2.item() :.6f}     "
                f"[quality_loss2] {1 * quality_loss2.item() :.6f}"
                f"[mask2] {1 * mask2 :.6f}"

                f"[noise_loss3] { 1 * noise_loss3.item() :.6f}     "
                f"[folding_loss3] { 1 * smooth3.item() :.6f}     "
                f"[smooth_loss3] {1 * fusion_loss3.item() :.6f}     "
                f"[fusion_loss3] {1 * fusion_loss3.item() :.6f}     "
                f"[quality_loss3] {1 * quality_loss3.item() :.6f}"
                f"[mask3] {1 * mask3 :.6f}"

                # f"[loss_total] {loss_total.item() :.6f}     "
                f"[lr] {optimizer.state_dict()['param_groups'][0]['lr'] :.6f}     "
                f"[t] {t.cpu().numpy()}")
        else:
            print(
                f" [epoch] {epoch}/{epochs}    "
                f"[epoch_step] {train_step}/{train_step_sum}     "
                f"[train_step] {num_train_step}     "
                f"[noise_loss1] { 1 * noise_loss.item() :.6f}     "
                f"[folding_loss1] { 1 * folding.item() :.6f}     "
                f"[smooth_loss1] {1 * smooth.item() :.6f}     "
                f"[fusion_loss1] {1 * fusion_loss.item() :.6f}     "
                f"[quality_loss1] {1 * quality_loss.item() :.6f}"
                f"[mask1] {1 * mask1 :.6f}"

                # f"[loss_total] {loss_total.item() :.6f}     "
                f"[lr] {optimizer.state_dict()['param_groups'][0]['lr'] :.6f}     "
                f"[t] {t.cpu().numpy()}")
        aver_loss_print = print_loss_sum / train_step_sum
        print('aver_loss', aver_loss_print)

        if epoch % save_model_epoch_step == 0:
            save_model(model, epoch, timestr)
        if epoch == epochs - 1:
            save_model(model, epoch, timestr)

        # update learning rate
        if use_lr_scheduler:
            learningRate_scheduler.step()
        # writer.add_scalar('aver_loss_epoch: ', aver_loss, epoch)
        log.write("\n")

    print("End of training")
    log.write("End of training \n")
    writer.close()


if __name__ == '__main__':
    config_path = "config.json"
    train(config_path)