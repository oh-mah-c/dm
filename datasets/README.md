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