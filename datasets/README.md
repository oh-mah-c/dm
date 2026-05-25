Here are a few transaction databases in SPMF format for high-utility itemset mining with negative unit profit values. Those datasets have been generated to be used in the FHN paper (Fournier-Viger et al., 2014) and can be used with the FHN and HUINIV-Mine algorithms. See the FHN paper for details.

---

## Vision Model Benchmarks

The vision models implemented in `src/models/vision/` are evaluated on standard image classification benchmarks described in their respective papers.

### VGGNet (Simonyan & Zisserman, 2015)

**Paper:** K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," *ICLR 2015*. arXiv:1409.1556v6. https://arxiv.org/abs/1409.1556

**Standard benchmark:** ImageNet ILSVRC-2012 (1.28M training, 50K validation, 1000 classes).

| Config | Name   | Weight layers | Params | Top-1 val. err. | Top-5 val. err. |
|--------|--------|---------------|--------|-----------------|-----------------|
| A      | VGG-11 | 11            | 133M   | 29.6%           | 10.4%           |
| B      | VGG-13 | 13            | 133M   | 28.7%           |  9.9%           |
| C      | —      | 16            | 134M   | 27.3%           |  8.8%           |
| D      | VGG-16 | 16            | 138M   | 25.6%           |  8.1%           |
| E      | VGG-19 | 19            | 144M   | **25.5%**       |  **8.0%**       |

*(Single-scale evaluation, S∈[256;512], Q=384 — paper Table 3)*

Training: SGD, lr=0.01, momentum=0.9, weight_decay=5e-4, batch=256, 74 epochs.

### ResNet (He et al., 2016)

**Paper:** K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *CVPR 2016*. https://doi.org/10.1109/CVPR.2016.90

**Standard benchmark:** ImageNet ILSVRC-2012 (same split as above).

| Variant   | Weight layers | Params | Top-1 val. err. | Top-5 val. err. |
|-----------|---------------|--------|-----------------|-----------------|
| ResNet-18 | 18            | 11.7M  | 27.88%          | —               |
| ResNet-34 | 34            | 21.8M  | 25.03%          | —               |
| ResNet-50 | 50            | 25.6M  | 22.85%          | 6.71%           |
| ResNet-101| 101           | 44.5M  | 21.75%          | 6.05%           |
| ResNet-152| 152           | 60.2M  | **21.43%**      | **5.71%**       |

*(Single-model, 10-crop testing — paper Table 3)*

Training: SGD, lr=0.1, momentum=0.9, weight_decay=1e-4, batch=256, 90 epochs.

### Swin Transformer (Liu et al., ICCV 2021)

**Paper:** Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," *ICCV 2021*. https://arxiv.org/abs/2103.14030

**Standard benchmark:** ImageNet-1K (regular training, single crop top-1 accuracy).

| Variant | C   | Depths       | Heads          | Params | FLOPs | Top-1 acc. |
|---------|-----|--------------|----------------|--------|-------|------------|
| Swin-T  | 96  | {2,2,6,2}    | {3,6,12,24}    | 29M    | 4.5G  | 81.3%      |
| Swin-S  | 96  | {2,2,18,2}   | {3,6,12,24}    | 50M    | 8.7G  | 83.0%      |
| Swin-B  | 128 | {2,2,18,2}   | {4,8,16,32}    | 88M    | 15.4G | 83.5%      |
| Swin-L  | 192 | {2,2,18,2}   | {6,12,24,48}   | 197M   | 34.5G | 86.4%\*    |

*(\* Swin-L with ImageNet-22K pre-training fine-tuned on ImageNet-1K — paper Table 1)*

*(Single-crop 224² evaluation — paper Table 1(a))*

Architecture highlights (Section 3):
- Patch partition: 4×4 non-overlapping patches → linear embedding to dim C
- 4 hierarchical stages; patch merging between stages doubles channels
- Swin Transformer blocks alternate W-MSA (regular windows) and SW-MSA (shifted windows)
- Window size M=7; relative position bias per head (Eq. 4)
- Efficient cyclic-shift trick for batched shifted-window attention (Fig. 4)

Training (Section 4.1 regular ImageNet-1K): AdamW, lr=0.001, weight_decay=0.05,
cosine LR decay, 20-epoch linear warm-up, 300 epochs, batch=1024.