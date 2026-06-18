# COSP-Net

Camouflage-Oriented Scanpath Prediction
via Fixation-Aligned Camouflage Attention

## 1. Environment

The experiments were run on Linux with CUDA GPUs.

Recommended environment:

```bash
conda create -n cosp python=3.9 -y
conda activate cosp

# Install PyTorch according to your CUDA version first.
# Example for CUDA 11.7:
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Install the remaining packages.
pip install numpy scipy pandas matplotlib opencv-python pillow tqdm tensorboard
pip install fvcore iopath pycocotools einops timm multimatch-gaze
```
## 2. Installation
- Install [Detectron2](https://github.com/facebookresearch/detectron2)

- Install MSDeformableAttention:

  ```bash
  cd ./hat/pixel_decoder/ops
  sh make.sh
  ```

 - Download pretrained model weights (ResNet-50 and Deformable Transformer) with the following python code
  ```
    if not os.path.exists("./pretrained_models/"):
        os.mkdir('./pretrained_models')

    print('downloading pretrained model weights...')
    url = f"http://vision.cs.stonybrook.edu/~cvlab_download/HAT/pretrained_models/M2F_R50_MSDeformAttnPixelDecoder.pkl"
    wget.download(url, 'pretrained_models/')
    url = f"http://vision.cs.stonybrook.edu/~cvlab_download/HAT/pretrained_models/M2F_R50.pkl"
    wget.download(url, 'pretrained_models/')
   
    os.makedirs("./pretrained_models/sinetv2", exist_ok=True)
    os.makedirs("./pretrained_models/res2net", exist_ok=True)

    gdown.download(
        "https://drive.google.com/uc?id=1D3RKQ8Nzd0ArV_c47StVKEuaoYTwnclR",
        "pretrained_models/sinetv2/Net_epoch_best.pth",
        quiet=False
    )

    url = "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth"
    wget.download(url, "pretrained_models/res2net/res2net50_v1b_26w_4s-3cf99910.pth")
  ```

## 3. Dataset Preparation

- Prepare the data following https://github.com/cvlab-stonybrook/Scanpath_Prediction.

Coordinates are in the resized image space：`512 x 320`.

## 4. Training

Train COSP-Adaptive:

```bash
python train.py \
  --hparams configs/cosp_camo_align_p4_a0p05_w0p02_free.json \
  --dataset-root <dataset_root>  \
  --gpu-id 0
```

Train fixed-length COSP:

```bash
python train.py \
  --hparams configs/cosp_camo_align_p4_a0p05_w0p02_len8.json \
  --dataset-root <dataset_root>  \
  --gpu-id 0
```

## 5. Evaluation

Evaluate a trained checkpoint:

```bash
python train.py \
  --hparams configs/cosp_camo_align_p4_a0p05_w0p02_free_15k.json \
  --dataset-root <dataset_root>  \
  --eval-only \
  --eval-mode greedy \
  --gpu-id 0
```

Evaluate Sequence Score under a fixed prediction budget:

```bash
python eval_ss_budget.py \
  --cluster-path datasets/COD10K_real/clusters.npy \
  --asset-dirs \
    assets/cosp_camo_align_p4_a0p05_w0p02_free \
    assets/cosp_camo_align_p4_a0p05_w0p02_len8 \
  --max-len 8 \
  --out-csv assets/ss_budget_eval.csv
```

## 6. Acknowledgement

This implementation is based on the HAT codebase:

```bibtex
@InProceedings{yang2024unify,
  author = {Yang, Zhibo and Mondal, Sounak and Ahn, Seoyoung and Xue, Ruoyu and Zelinsky, Gregory and Hoai, Minh and Samaras, Dimitris},
  title = {Unifying Top-down and Bottom-up Scanpath Prediction Using Transformers},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2024}
}
```
