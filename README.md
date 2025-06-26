# Real-time Domain Adaptation in Semantic Segmentation

This project focuses on applying **domain adaptation techniques** to **real-time semantic segmentation networks**, using **Cityscapes** as the real-world dataset and **GTA5** as the synthetic source domain. The goal is to study the impact of domain gap and evaluate adaptation strategies such as photometric and geometric augmentations, Fourier Domain Adaptation (FDA), Domain Adaptation via Cross-domain Mixed Sampling (DACS), and histogram matching with architectures like **BiSeNet** and **DeepLabV2**. In addition, we extend our research by implementing loss functions such as Dice loss and Focal loss to reduce class imbalancing and improve predictions on under represented categories. Results are reported in terms of mean Intersection over Union (mIoU), inference latency, and FLOPs.


## Directory Overview

- **`configs/`**  
  YAML configuration files specifying training parameters.

- **`datasets/`**  
  Custom dataset classes for **Cityscapes** and **GTA5**.

- **`models/`**  
  Implementations of **BiSeNet** and **DeepLabV2** architectures.

- **`scripts/`**  
  Contains training logic, checkpoint handling, data downloading, visualization utilities, and evaluation functions.

- **`*.ipynb`**  
  Each notebook trains a separate model or demonstrates a specific domain adaptation technique.

- **`requirements.txt`**  
  List of required Python packages for running the project.

- **`README.md`**  
  Project documentation you’re reading now.

### Notebooks Overview

- `DeepLabV2.ipynb` — training DeepLabV2 on Cityscapes
- `BiSeNet.ipynb` — training BiSeNet on Cityscapes
- `DomainShift.ipynb` — training on GTA5, validation on Cityscapes
- `Augmentations.ipynb` — basic augmentations to reduce domain shift
- `FDA.ipynb` — FDA
- `DACS.ipynb` — DACS
- `Bisenet_Loss_function.ipynb` — Dice and Focal loss
- `Bisenet_Histogram_Matching.ipynb` — Histogram Matching

## How to Run

Each notebook (`*.ipynb`) is standalone and contains:
- Data downloading scripts
- Preprocessing
- Model training and validation
- Evaluation (mIoU, latency, FLOPs)


## Pretrained Models & Datasets

Download links:

- **DeepLab pretrained weights:**  
  https://drive.google.com/file/d/1ZX0UCXvJwqd2uBGCX7LI2n-DfMg3t74v/view?usp=sharing
  
- **Cityscapes dataset:**  
  https://drive.google.com/file/d/1Qb4UrNsjvlU-wEsR9d7rckB0YS_LXgb2/view?usp=sharing
  
- **GTA5 dataset:**  
  https://drive.google.com/file/d/1xYxlcMR2WFCpayNrW2-Rb7N-950vvl23/view?usp=sharing


## Requirements

Install dependencies:

```bash
pip install -r requirements.txt

