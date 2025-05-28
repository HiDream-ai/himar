# Hi-MAR

<p align="center">
    <img src="assets/show_imgs.png" width="800"/>
<p>

<p align="center">
    🖥️ <a href="https://github.com/HiDream-ai/himar">GitHub</a> &nbsp&nbsp ｜ &nbsp&nbsp  🌐 <a href="https://Tom-zgt.github.io/Hi-MAR-page/"><b>Project Page</b></a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/HiDream-ai/Hi-MAR/tree/main">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="">Paper </a> &nbsp&nbsp | &nbsp&nbsp 📖 <a href="">PDF</a> &nbsp&nbsp 
<br>

[**Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots**](https://Tom-zgt.github.io/Hi-MAR-page/) (ICML 2025)<be>

This is the official repository for the Paper "Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots"

## Overview

We present a Hierarchical Masked Autoregressive models (Hi-MAR) that pivot on low-resolution image tokens to trigger hierarchical autoregressive modeling in a multi-phase manner.

#### 🔍 What We're Working to Solve?

- **Incapable of utilizing global context** in early-stage predictions of the next-token paradigm
- **Training-inference discrepancy** across multi-scale predictions
- **Suboptimal multi-scale probability distribution modeling** 
- **Lack of global information in the denoising process of the MLP-based Diffusion head**


## 🔥 Updates

- [x] **\[2025.05.22\]** Upload inference code and pretrained class-conditional Hi-MAR models trained on ImageNet 256x256.

## 🏃🏼 Inference

<details open>
<summary><strong>Environment Requirement</strong></summary>


Clone the repo:

```
git clone https://github.com/HiDream-ai/himar.git
cd himar
```

Install dependencies:

```
conda env create -f environment.yaml

conda activate himar
```

</details>

<details open>
<summary><strong>Model Download</strong></summary>

Download VAE from the [link](https://www.dropbox.com/scl/fi/hhmuvaiacrarfg28qxhwz/kl16.ckpt?rlkey=l44xipsezc8atcffdp4q7mwmh&dl=0) in the [MAR Github](https://github.com/LTH14/mar/).

You can download our pre-trained Hi-MAR models directly from the links provided here.

| Models                                                       | FID-50K | Inception Score | #params |
| ------------------------------------------------------------ | ------- | --------------- | ------- |
| [Hi-MAR-B](https://huggingface.co/HiDream-ai/Hi-MAR/blob/main/Hi-MAR-B/checkpoint-last.pth) | 1.93    | 293.0           | 244M    |
| [Hi-MAR-L](https://huggingface.co/HiDream-ai/Hi-MAR/blob/main/Hi-MAR-L/checkpoint-last.pth) | 1.66    | 322.3           | 529M    |
| [Hi-MAR-H](https://huggingface.co/HiDream-ai/Hi-MAR/blob/main/Hi-MAR-H/checkpoint-last.pth) | 1.52    | 322.78          | 1090M   |

</details>

<details open>
<summary><strong>Evaluation</strong></summary>

Evaluate Hi-MAR-B on ImageNet256x256:

```
torchrun --nproc_per_node=8 --nnodes=1 main_himar.py --img_size 256 --vae_path /path/to/vae --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model himar_base --diffloss_d 6 --diffloss_w 1024 --output_dir ./himar_base_test --resume /path/to/Hi-MAR-B  --num_images 50000 --num_iter 4 --cfg 2.5 --re_cfg 2.7 --cfg_schedule linear --cond_scale 8 --cond_dim 16 --two_diffloss --global_dm --gdm_d 6 --gdm_w 512 --eval_bsz 256 --load_epoch -1 --head 8 --ratio 4 --cos --evaluate
```

Evaluate Hi-MAR-L on ImageNet256x256:

```
torchrun --nproc_per_node=8 --nnodes=1 main_himar.py --img_size 256 --vae_path /path/to/vae --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model himar_large --diffloss_d 8 --diffloss_w 1280 --output_dir ./himar_large_test --resume /path/to/Hi-MAR-L  --num_images 50000 --num_iter 4 --cfg 3.5 --re_cfg 3.5 --cfg_schedule linear --cond_scale 8 --cond_dim 16 --two_diffloss --global_dm --gdm_d 8 --gdm_w 512 --eval_bsz 256 --load_epoch -1 --head 8 --ratio 4 --cos --evaluate
```

Evaluate Hi-MAR-H on ImageNet256x256:

```
torchrun --nproc_per_node=8 --nnodes=1 main_himar.py --img_size 256 --vae_path /path/to/vae --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model himar_huge --diffloss_d 12 --diffloss_w 1536 --output_dir ./himar_huge_test --resume /path/to/Hi-MAR-H  --num_images 50000 --num_iter 12 --cfg 3.2 --re_cfg 5.5 --cfg_schedule linear --cond_scale 8 --cond_dim 16 --two_diffloss --global_dm --gdm_d 12 --gdm_w 768 --eval_bsz 256 --load_epoch -1 --head 12 --ratio 4 --cos --evaluate
```

</details>


## 🌟 Star and Citation

If you find our work helpful for your research, please consider giving a star⭐ on this repository and citing our work.

```
@misc{zheng2025hierarchicalmaskedautoregressivemodels,
      title={Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots}, 
      author={Guangting Zheng and Yehao Li and Yingwei Pan and Jiajun Deng and Ting Yao and Yanyong Zhang and Tao Mei},
      year={2025},
      eprint={2505.20288},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.20288}, 
}
```


## 💖 Acknowledgement

<span id="acknowledgement"></span>

Thanks to the contribution of [MAR](https://github.com/LTH14/mar)
