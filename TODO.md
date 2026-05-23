# DM — TODO

## 🟡 Medium Priority

### Automatic differentiation (autograd)
- [ ] Implement reverse-mode autograd on `DM_Tensor`
- [ ] `dm.GradientTape` context manager (Python)
- [ ] Backward pass for: conv2d, softmax, batch_norm, layer_norm (relu/linear already have backward)
- [ ] Optimizer `.step()` that pulls from tape automatically

### Model saving / serialization (custom models)
- [ ] Define a `.dmw` weight file format (header + raw float32 blobs) for custom `dm.Model` subclasses
- [ ] `model.save("weights.dmw")` / `model.load("weights.dmw")`
- [ ] Export to ONNX (optional, stretch goal)

---

## 🟢 Low Priority / Future

### Backend — Vulkan cooperative matrix GEMM
- [ ] Implement real Vulkan coop-matrix GEMM shader (currently stubbed in `src/core/dm_backend.c`)
- [ ] Wire `DM_BACKEND_VULKAN_COOP_MAT` path in `dm_matmul_dispatch()`

### Backend — ROCm/HIP (Tier 3)
- [ ] Implement `probe_rocm()` via `dlsym` on `hipInit`
- [ ] Implement `dm_rocm_matmul()` dispatch path

### Dataset API
- [ ] `dm.Dataset.from_csv(path)` — already partially exists, expose to all bindings
- [ ] `dm.Dataset.from_images(dir)` image folder loader
- [ ] `dm.DataLoader` with shuffle, batch, prefetch

### `@tf.function` equivalent
- [ ] Lazy/deferred execution graph for fusing ops before dispatch
- [ ] Useful for Vulkan: batch multiple ops into one command buffer

---

## ✅ Completed

- [x] All `dm_engine` primitives exposed as first-class public API in all 6 bindings (C, C++, Python, Go, JS, Java)
- [x] Backend dispatch system: CPU (Tier 0) → Vulkan compute (Tier 1) → TensorFlow/XLA (Tier 2)
- [x] `DM_BACKEND` env var override (`cpu | vulkan | tensorflow | auto`)
- [x] Runtime TF detection via `dlsym(RTLD_DEFAULT, "TFE_NewContext")` — no hard link required
- [x] `dm.backend_init/get/set/query/name` in all language bindings
- [x] `DM_BackendInfo` struct (6-field plain C, ABI-safe across all bindings)
- [x] 130+ classical data mining algorithms (FP-Growth, ECLAT, PrefixSpan, SPADE, …)
- [x] 8 tokenizers (BPE, BPE-Dropout, SentencePiece-lite, Unigram, FastWordPiece, Volt, GPE, MaximalMunch)
- [x] Pre-built models: ResNet, ViT, TinyViT-5M/11M/21M, MobileNet-Tiny, BERT, Transformer, GAN, VAE, TinyStories
- [x] Portable backend: "Portable by design. Accelerated when possible. Vendor-locked never."
- [x] **Python** `dm.models.*` — ResNet, ViT, TinyViT, MobileNetTiny, BERTModel, TransformerModel, VAEModel, GANModel
- [x] **Python** `dm.tokenizer.*` — BPE, BPEDropout, Unigram, SentencePiece, WordPiece, GPE, ParityBPE, MaximalMunch, Volt, Faro, TokenizerLab
- [x] **Python** `dm.losses` — cross_entropy, mse, binary_cross_entropy, kl_divergence
- [x] **Python** `dm.Model` base class — forward, parameters, zero_grad, fit(), evaluate(), predict()
- [x] **C++** `dm::models::ResNet/ViT/TinyViT/MobileNetTiny/BERTModel/TransformerModel/VAEModel/GANModel`
- [x] **C++** `dm::tokenizer::BPE/BPEDropout/Unigram/SentencePiece/WordPiece/GPE/ParityBPE/MaximalMunch/Volt/Faro/TokenizerLab`
- [x] **Go** `dm.NewResNet/NewViT/NewMobileNetTiny/NewBERT/NewTransformer/NewVAE/NewGAN`
- [x] **Go** `dm.NewTokenizerBPE/BPEDropout/Unigram/SentencePiece/WordPiece/GPE/Volt`
- [x] **JS** `dm.models.resNet/viT/mobileNetTiny/bert/transformer` + `dm.tokenizer.bpe/bpeDropout/unigram/...`
- [x] **Java** `DM.Models.ResNet/ViT/MobileNetTiny/BERTModel/TransformerModel/VAEModel/GANModel`
- [x] **Java** `DM.Tokenizers.BPE/BPEDropout/Unigram/SentencePiece/WordPiece/GPE/ParityBPE/MaximalMunch/Volt/Faro/TokenizerLab`
- [x] `dm.h` extended with `dm_transformer_*_raw`, `dm_bert_save_raw`, `dm_mobilenet_tiny_forward_raw2` declarations
- [x] `dm_lib.c` extended with Transformer raw FFI wrappers (all 8 functions)
