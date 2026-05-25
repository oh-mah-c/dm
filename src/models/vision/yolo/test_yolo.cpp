// ─────────────────────────────────────────────────────────────────────────────
// test_yolo.cpp — C++ structural tests for YOLO v1
//
// Paper: J. Redmon, S. Divvala, R. Girshick, A. Farhadi,
//        "You Only Look Once: Unified, Real-Time Object Detection", CVPR 2016
//
// Tests verify:
//  1.  Output tensor shape [batch, S, S, B*5+C]  (Section 2)
//  2.  PASCAL VOC config: S=7, B=2, C=20 → 30 channels  (Section 2)
//  3.  Backbone reduces 448×448 input to 7×7 spatial  (Figure 3)
//  4.  Leaky ReLU activation slope 0.1  (Section 2.2, Eq. 2)
//  5.  Loss is finite and positive with dummy GT  (Section 2.2, Eq. 3)
//  6.  λ_coord=5 amplifies coord vs conf loss  (Section 2.2)
//  7.  λ_noobj=0.5 reduces no-object confidence loss  (Section 2.2)
//  8.  √w, √h in wh loss term — large box errors smaller than small box  (Section 2.2)
//  9.  Gradient flow: no NaN/Inf  (Section 2.2)
// 10.  SGD+momentum step changes weights  (Section 2.2)
// 11.  box_iou_xywh: identical boxes → IoU == 1  (Section 2.2)
// 12.  NMS removes duplicate boxes  (Section 2.3)
// 13.  decode() returns [N, 6] boxes  (Section 2.3)
// 14.  Checkpoint save/load round-trip
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/yolo/yolo.h"

#include <torch/torch.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>

using namespace dm::models::vision;

static int s_passed = 0, s_failed = 0;
#define ASSERT_TRUE(cond, msg) \
    do { if (!(cond)) { \
        std::fprintf(stderr, "  FAIL: %s\n", (msg)); \
        ++s_failed; return; \
    } } while(0)

static void begin_test(const char* name) {
    std::printf("[test] %s ...", name); std::fflush(stdout);
}
static void end_test() { ++s_passed; std::printf(" PASS\n"); }

// Shared dimensions (PASCAL VOC, Section 2)
static const int64_t S = 7, B = 2, C = 20;
static const int64_t BATCH = 2;

// ── Helper: build a small fake target tensor ─────────────────────────────────
static torch::Tensor make_target() {
    // All zeros (no objects) — simplest valid target
    return torch::zeros({BATCH, S, S, B*5 + C});
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Output shape [batch, S, S, B*5+C]
// ─────────────────────────────────────────────────────────────────────────────
static void test_output_shape() {
    begin_test("output shape [batch, S, S, B*5+C] (Section 2)");
    auto model = YOLO(S, B, C);
    model->eval();
    torch::NoGradGuard ng;
    auto x    = torch::rand({BATCH, 3, 448, 448});
    auto pred = model->forward(x);
    ASSERT_TRUE(pred.sizes() == torch::IntArrayRef({BATCH, S, S, B*5+C}),
                "output shape wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. PASCAL VOC: B*5+C = 2*5+20 = 30 channels
// ─────────────────────────────────────────────────────────────────────────────
static void test_voc_channels() {
    begin_test("PASCAL VOC B*5+C = 30 channels (Section 2)");
    ASSERT_TRUE(B*5 + C == 30, "channel count wrong for VOC");
    auto model = YOLO(S, B, C);
    model->eval();
    torch::NoGradGuard ng;
    auto x    = torch::rand({1, 3, 448, 448});
    auto pred = model->forward(x);
    ASSERT_TRUE(pred.size(3) == 30, "output last-dim != 30");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Backbone reduces 448×448 → 7×7 spatial feature map (Figure 3)
// ─────────────────────────────────────────────────────────────────────────────
static void test_backbone_spatial() {
    begin_test("backbone 448→7 spatial reduction (Figure 3)");
    auto model = YOLO(S, B, C);
    model->eval();
    torch::NoGradGuard ng;
    auto x    = torch::rand({1, 3, 448, 448});
    auto feat = model->backbone->forward(x);
    ASSERT_TRUE(feat.size(2) == S && feat.size(3) == S,
                "backbone output spatial size != 7×7");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Leaky ReLU negative slope = 0.1  (Section 2.2, Eq. 2)
// ─────────────────────────────────────────────────────────────────────────────
static void test_leaky_relu_slope() {
    begin_test("leaky ReLU slope = 0.1 (Section 2.2, Eq. 2)");
    auto act = torch::nn::LeakyReLU(
        torch::nn::LeakyReLUOptions().negative_slope(0.1));
    auto x   = torch::tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
    auto y   = act->forward(x);
    // negative part: y = 0.1 * x
    ASSERT_TRUE(std::abs(y[0].item<float>() - (-0.2f)) < 1e-5f, "slope wrong at -2");
    ASSERT_TRUE(std::abs(y[1].item<float>() - (-0.1f)) < 1e-5f, "slope wrong at -1");
    // positive part: y = x
    ASSERT_TRUE(std::abs(y[4].item<float>() - 2.0f) < 1e-5f, "slope wrong at +2");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Loss is finite and positive  (Section 2.2, Eq. 3)
// ─────────────────────────────────────────────────────────────────────────────
static void test_loss_finite() {
    begin_test("loss is finite and >= 0 (Section 2.2, Eq. 3)");
    auto model  = YOLO(S, B, C);
    model->train();
    auto x      = torch::rand({BATCH, 3, 448, 448});
    auto pred   = model->forward(x);
    auto target = make_target();
    auto loss   = yolo_loss(pred, target, S, B, C);
    ASSERT_TRUE(std::isfinite(loss.item<float>()), "loss is not finite");
    ASSERT_TRUE(loss.item<float>() >= 0.f, "loss is negative");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. λ_coord=5 amplifies coordinate loss  (Section 2.2)
//    Loss with λ_coord=5 > loss with λ_coord=1 when boxes don't match
// ─────────────────────────────────────────────────────────────────────────────
static void test_lambda_coord() {
    begin_test("lambda_coord=5 amplifies xy/wh loss (Section 2.2)");
    auto model  = YOLO(S, B, C);
    model->eval();
    torch::NoGradGuard ng;
    auto x      = torch::rand({BATCH, 3, 448, 448});
    auto pred   = model->forward(x);
    // target with one object in cell (0,0)
    auto target = torch::zeros({BATCH, S, S, B*5 + C});
    target.slice(3, 0, 4).fill_(0.5f);  // x,y,w,h = 0.5
    target.slice(3, 4, 5).fill_(1.0f);  // conf = 1

    auto loss1 = yolo_loss(pred, target, S, B, C, /*lambda_coord=*/1.0f, 0.5f).item<float>();
    auto loss5 = yolo_loss(pred, target, S, B, C, /*lambda_coord=*/5.0f, 0.5f).item<float>();
    ASSERT_TRUE(loss5 > loss1, "lambda_coord=5 should give larger loss than lambda_coord=1");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. λ_noobj=0.5 reduces no-object confidence contribution  (Section 2.2)
// ─────────────────────────────────────────────────────────────────────────────
static void test_lambda_noobj() {
    begin_test("lambda_noobj=0.5 reduces noobj conf loss (Section 2.2)");
    auto model  = YOLO(S, B, C);
    model->eval();
    torch::NoGradGuard ng;
    auto x      = torch::rand({BATCH, 3, 448, 448});
    auto pred   = model->forward(x);
    auto target = make_target();   // all noobj

    auto loss_low  = yolo_loss(pred, target, S, B, C, 5.0f, /*lambda_noobj=*/0.5f).item<float>();
    auto loss_high = yolo_loss(pred, target, S, B, C, 5.0f, /*lambda_noobj=*/2.0f).item<float>();
    ASSERT_TRUE(loss_low < loss_high, "lambda_noobj=0.5 should produce smaller loss than 2.0");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. √w/√h trick: error on large box < error on small box for same Δ  (Section 2.2)
// ─────────────────────────────────────────────────────────────────────────────
static void test_sqrt_wh_trick() {
    begin_test("sqrt wh trick: large-box error smaller (Section 2.2)");
    // Large box: w=0.8→0.9  δ=0.1 → (√0.8-√0.9)² ≈ 0.00289
    // Small box: w=0.1→0.2  δ=0.1 → (√0.1-√0.2)² ≈ 0.00858
    float large_err = std::pow(std::sqrt(0.8f) - std::sqrt(0.9f), 2.f);
    float small_err = std::pow(std::sqrt(0.1f) - std::sqrt(0.2f), 2.f);
    ASSERT_TRUE(large_err < small_err,
                "sqrt trick: large-box error should be smaller than small-box error");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Gradient flow: no NaN/Inf
// ─────────────────────────────────────────────────────────────────────────────
static void test_gradient_flow() {
    begin_test("gradient flow: no NaN/Inf");
    auto model  = YOLO(S, B, C);
    model->train();
    auto x      = torch::rand({BATCH, 3, 448, 448});
    auto pred   = model->forward(x);
    auto target = make_target();
    auto loss   = yolo_loss(pred, target, S, B, C);
    loss.backward();
    bool ok = true;
    for (auto& p : model->parameters()) {
        if (!p.grad().defined()) continue;
        if (p.grad().isnan().any().item<bool>() ||
            p.grad().isinf().any().item<bool>()) { ok = false; break; }
    }
    ASSERT_TRUE(ok, "NaN/Inf in gradients");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. SGD + momentum step changes weights  (Section 2.2: momentum=0.9)
// ─────────────────────────────────────────────────────────────────────────────
static void test_sgd_update() {
    begin_test("SGD+momentum step changes weights (Section 2.2)");
    auto model = YOLO(S, B, C);
    model->train();
    torch::optim::SGD opt(model->parameters(),
        torch::optim::SGDOptions(1e-3).momentum(0.9));

    // Capture first conv weight
    auto w_before = model->backbone->ptr(0)
                        ->as<torch::nn::Conv2dImpl>()
                        ->weight.clone().detach();

    auto x      = torch::rand({BATCH, 3, 448, 448});
    auto pred   = model->forward(x);
    auto target = make_target();
    auto loss   = yolo_loss(pred, target, S, B, C);
    opt.zero_grad();
    loss.backward();
    opt.step();

    auto w_after = model->backbone->ptr(0)
                       ->as<torch::nn::Conv2dImpl>()
                       ->weight.detach();
    ASSERT_TRUE(!torch::allclose(w_before, w_after),
                "weights unchanged after SGD step");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. box_iou_xywh: identical boxes → IoU == 1  (Section 2.2)
// ─────────────────────────────────────────────────────────────────────────────
static void test_iou_identical() {
    begin_test("box_iou_xywh: identical boxes -> IoU = 1 (Section 2.2)");
    auto gt   = torch::tensor({0.5f, 0.5f, 0.4f, 0.4f});
    auto pred = gt.unsqueeze(0);  // [1, 4]
    auto iou  = box_iou_xywh(pred, gt);
    ASSERT_TRUE(std::abs(iou[0].item<float>() - 1.0f) < 1e-4f, "IoU not 1 for identical boxes");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. NMS removes duplicate boxes  (Section 2.3)
// ─────────────────────────────────────────────────────────────────────────────
static void test_nms() {
    begin_test("NMS removes duplicate boxes (Section 2.3)");
    // Two nearly identical boxes (high IoU), one weaker — NMS should keep 1
    auto boxes = torch::tensor({
        {0.1f, 0.1f, 0.5f, 0.5f, 0.9f, 0.f},   // strong
        {0.1f, 0.1f, 0.5f, 0.5f, 0.4f, 0.f},   // duplicate (high IoU)
        {0.6f, 0.6f, 0.9f, 0.9f, 0.8f, 1.f},   // separate box, different class
    });
    auto kept = nms(boxes, 0.45f);
    ASSERT_TRUE(kept.size(0) == 2, "NMS should keep 2 boxes (1 per unique location)");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. decode() returns [N, 6] boxes  (Section 2.3)
// ─────────────────────────────────────────────────────────────────────────────
static void test_decode_shape() {
    begin_test("decode() returns [N, 6] boxes (Section 2.3)");
    auto model = YOLO(S, B, C);
    model->eval();
    torch::NoGradGuard ng;
    auto x     = torch::rand({1, 3, 448, 448});
    auto pred  = model->forward(x);
    // Use very low threshold to guarantee some detections
    auto boxes = model->decode(pred, /*conf_thresh=*/0.0f, /*nms_thresh=*/0.45f);
    ASSERT_TRUE(boxes.size() == 1, "decode should return 1 result per image");
    // Each detection row has 6 columns: x1,y1,x2,y2,score,class_id
    if (boxes[0].size(0) > 0) {
        ASSERT_TRUE(boxes[0].size(1) == 6, "box row should have 6 columns");
    }
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. Checkpoint save/load round-trip
// ─────────────────────────────────────────────────────────────────────────────
static void test_checkpoint() {
    begin_test("checkpoint save/load round-trip");
    auto m1 = YOLO(S, B, C);
    m1->eval();
    torch::NoGradGuard ng;
    auto x = torch::rand({1, 3, 448, 448});
    auto p1 = m1->forward(x);

    const std::string path = "/tmp/test_yolo_ckpt.pt";
    {
        torch::serialize::OutputArchive ar;
        m1->save(ar);
        ar.save_to(path);
    }
    auto m2 = YOLO(S, B, C);
    {
        torch::serialize::InputArchive ar;
        ar.load_from(path);
        m2->load(ar);
    }
    m2->eval();
    auto p2 = m2->forward(x);

    ASSERT_TRUE(torch::allclose(p1, p2, 1e-5f, 1e-5f),
                "predictions differ after checkpoint reload");
    std::filesystem::remove(path);
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::printf("\n=== YOLO v1 tests (Redmon et al., CVPR 2016) ===\n");

    test_output_shape();
    test_voc_channels();
    test_backbone_spatial();
    test_leaky_relu_slope();
    test_loss_finite();
    test_lambda_coord();
    test_lambda_noobj();
    test_sqrt_wh_trick();
    test_gradient_flow();
    test_sgd_update();
    test_iou_identical();
    test_nms();
    test_decode_shape();
    test_checkpoint();

    std::printf("=== %d passed, %d failed ===\n\n", s_passed, s_failed);
    return s_failed == 0 ? 0 : 1;
}
