# LFDT-Fusion

The code of "A Novel End-to-End Unsupervised Multi-Task Diffusion Network for Hierarchical Medical Multi-Modality Image Full-Color Channel Fusion"

## Update
- [2025/10/20] Release data setting and preprocess for Diff-FCCF.
- [2025/10/21] Release all the code for Diff-FCCF.



## Abstract
Medical image fusion (MIF) aims to integrate complementary information from multiple imaging modalities into a single, informative image to enhance clinical diagnosis and treatment. While deep learning-based MIF methods have achieved significant progress, they still face critical challenges, including the separate-stage optimization of multi-tasks, limited adaptability in modality information imbalance, and suboptimal training results. Recent diffusion probabilistic models offer promising generative capabilities for image processing; however, existing diffusion-based fusion frameworks remain constrained by the overlook in handling full-color functional data, and fixed inference procedures in simplified grayscale fusion pipelines. To address these challenges, we propose Diff-FCCF, an end-to-end unsupervised multi-task diffusion network for full-color channel fusion in multi-modality medical imaging. Diff-FCCF unifies full-color image fusion, image registration, and denoising tasks within a single framework. A novel Chebyshev graph-guided U-Net is introduced in the diffusion denoising module to enhance cross-modal alignment. Additionally, a hierarchical inference strategy dynamically adjusts the fusion process to mitigate modality dominance and information imbalance. To optimize training without ground truth, we introduce an exploration-to-stability reinforcement learning strategy guided by a multi-objective loss, improving convergence and generalization. Extensive experiments on five benchmark fusion groups demonstrate that Diff-FCCF achieves state-of-the-art performance in both quantitative and qualitative evaluations, especially outperforming comparable methods by more than 20% in the Mutual Information (MI) metric, and producing diagnostically valuable, interpretable fusion results.
### Framework
<div style="text-align: center;">
    <img src="./docs/image.png" width="90%">
</div>
<p> Overall framework of Diff-FCCF. (a) The training pipeline employs a multi-task U-Net to predict the noise, deformable field, and fusion masks based on a full-color strategy and an exploration-to-stability reinforcement learning optimization. (b) The inference pipeline adopts a Hierarchical Condition Enhancement strategy to predict the noise, deformable field, and fusion masks, and refines foreground details across three progressive stages.</p>



## Environment

We test the code on PyTorch 2.6.0 + CUDA 12.9.

1. Create a new conda environment
```
conda create -n Diff-FCCF python=3.12.7
conda activate Diff-FCCF
```

2. Install dependencies
```
pip install -r requirements.txt
```



## Test
You can directly test our model to generate fused images using the following code (note: the pre-training weights sholud be saved in the './check/' file)

Link:  [checkpoint](https://drive.google.com/drive/folders/1BDhQ7wgpRDpF_4Yv6wqBzfykQHE_Dil8?usp=drive_link)
```
#Visible and infrared image fusion
python test-ir.py

#Visible and near-infrared image fusion
python test-nir.py

# Visible and polarized image fusion
python test-pif.py

#Medical image fusion
python test-med.py

#Multi-exposure image fusion
python test-mef.py

#Multi-focus image fusion
python test-mff.py
```
You can find their corresponding configuration file paths in './config/'.
## Train

### 1. Prepare dataset
The datasets are available for download at the following link. (We greatly appreciate the authors opening these datasets.)

<div align="center">

|  Task  |  dataset Name  |  Link  |
| :----------: | :----------: | :-----: |
|         | M3FD   |  https://github.com/JinyuanLiu-CV/TarDAL  |
|  VI-IR  | MSRS   |  https://github.com/Linfeng-Tang/MSRS  |
|         | LLVIP  |  https://github.com/bupt-ai-cz/LLVIP |
|  VI-NIR |  RGB-NIR Scene   |  https://www.epfl.ch/labs/ivrl/research/downloads/rgb-nir-scene-dataset/  |
|   |  MCubeS  |   https://github.com/kyotovision-public/multimodal-material-segmentation |
|  MED    |  Harvard  |  https://www.med.harvard.edu/AANLIB/home.html  |
|  VI-PIF    | Polarization Image  |  http://www.ok.sc.e.titech.ac.jp/res/PolarDem/index.html  |
|       | RSP  |  https://github.com/donyakh/polar-densification/tree/master  |
|  MEF  |  SICE  |  https://github.com/csjcai/SICE |
|       |   MEFB |  https://github.com/xingchenzhang/MEFB |
|  MFF  |  RealMFF  | https://github.com/Zancelot/Real-MFF  |
|       | MFI-WHU   |  https://github.com/HaoZhang1018/MFI-WHU  |
</div>



The data should organized in the following format:
```
train
├── VI-IR                                      ├── VI-NIR
│   ├──ir                                      │   ├──NIR
│   │   ├──0000.png                            │   │   ├──0000.png
...                                            ...
│   ├──vi                                      │   ├──VI
│   │   ├──0000.png                            │   │   ├──0000.png
...                                            ...


├── Med                                        ├── PIF
│   ├──SPECT                                   │   ├──DoLP
│   │   ├──0000.png                            │   │   ├──0000.png
...                                            ...
│   ├──MRI                                     │   ├──S0
│   │   ├──0000.png                            │   │   ├──0000.png
...                                            ...


├── MEF                                        ├── MFF
│   ├──over                                    │   ├──source_1
│   │   ├──0000.png                            │   │   ├──0000.png
...                                            ...
│   ├──under                                   │   ├──source_2
│   │   ├──0000.png                            │   │   ├──0000.png
...                                            ...
```
### 2. Start training
You can use the following code to train the LFDT-Fusion model for different fusion tasks.
```
python -m torch.distributed.launch  --nproc_per_node=8 train.py --sample_selected ddp-solver++ --model_selected DFT --batch_size 4 --fusion_task VI-IR --strategy MAX
```
* nproc_per_node: This parameter represents the number of GPU. （Note: If you want to change nproc_per_node, the device number of "gpu_ids" in the configuration file './config/train.json' needs to be changed as well. For example, if nproc_per_node=2, gpu_ids=[0,1].）

* sample_selected： This parameter represents the selection of the diffusion sampler. There are nine diffusion samplers to choose from: "DDIM, ddp-solver, ddp-solver++, Deis, Unipc, Heun, Euler". The default value is ddp-solver++. Here ddp-solver++ is DPM-Solver++.

* model_selected: This parameter represents a denoising network, which defaults to a Diffusion Transformer network (DFT).

* batch_size: This parameter represents the batch size.

* fusion_task: This parameter represents the selection of the fusion task. There are seven options: "MEF, VI-IR, VI-NIR, PIF, MFF, SPECT-MRI, PET-MRI". 

* strategy: This parameter represents the fusion strategy. There are two options: "MAX, MEAN". 

If you want to train six fusion tasks at once, you can also run the following code:

```
python sample_all.py
```

## Fusion examples
### 1. VI-IR imaging
<div style="text-align: center;">
    <img src="./figure/VI_IR_show.jpg" width="90%">
</div>
<p> Fig. 2.   Qualitative comparisons among various fusion methods for VI-IR fusion task. (a) VI image. (b) IR image. (c) DDFM. (d) DeFusion. (e) DIFNet. (f) DiF-fusion. (g) MUFusion. (h) SwinFusion. (i) CDDFuse. (j) TUFusion [37]. (k) U2Fusion. (l) SeAFusion. (m) TarDAL. (n) Ours. (The lower left corner is a closeup view of the different color boxes.).</p>

### 2. Medical imaging
<div style="text-align: center;">
    <img src="./figure/PET_show.jpg" width="90%">
</div>
<p> Fig. 3.  Qualitative comparisons among various fusion methods for Med (PET-MRI) fusion task. (a) SPECT image. (b) MRI image. (c) DDFM. (d) DeFusion. (e) DIFNet. (f) CDDFuse. (g) MATR. (h) MUFusion. (i) SwinFusion. (j) TUFusion. (k) U2Fusion. (l) Ours. (The lower left corner is a closeup view of the different color boxes.).</p>

* See the article for more examples
## Citation
```
@article{
    author    = {Bo Yang, Zhaohui Jiang, Dong Pan, Haoyang Yu, Gui Gui, Weihua Gui},
    title     = {LFDT-Fusion: A Latent Feature-guided Diffusion Transformer Model for General Image Fusion},
    booktitle = {Infornation Fusion},
    year      = {2024},
    pages     = {102639},
    doi       = {doi.org/10.1016/j.inffus.2024.102639},
}
```