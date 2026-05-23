// Package dm provides Go bindings for libdm via CGo.
//
// Build prerequisites:
//   CGO_CFLAGS="-I/path/to/dm/include"
//   CGO_LDFLAGS="-L/path/to/dm -ldm"
//   go build ./...
//
// Or set them in go.env / GOFLAGS, or use a build tag.
//
// SPDX-License-Identifier: MIT
package dm

/*
#cgo CFLAGS:  -I../../../include
#cgo LDFLAGS: -ldm

#include "dm.h"
#include <stdlib.h>
#include <string.h>

// Backend shims — static inline so gopls resolves them from this preamble
// without needing to chase the #cgo CFLAGS include path.
typedef struct {
    int active;
    int tf_available;
    int vulkan_available;
    int coop_mat_available;
    int cuda_available;
    int rocm_available;
} DmCgoBackendInfo;

static inline void dm_cgo_backend_init(void) { dm_backend_init(); }
static inline int  dm_cgo_backend_get(void)  { return (int)dm_backend_get(); }
static inline void dm_cgo_backend_set(int b) { dm_backend_set((DM_Backend)b); }
static inline DmCgoBackendInfo dm_cgo_backend_query(void) {
    DM_BackendInfo q = dm_backend_query();
    DmCgoBackendInfo r;
    r.active             = (int)q.active;
    r.tf_available       = q.tf_available;
    r.vulkan_available   = q.vulkan_available;
    r.coop_mat_available = q.coop_mat_available;
    r.cuda_available     = q.cuda_available;
    r.rocm_available     = q.rocm_available;
    return r;
}
static inline const char *dm_cgo_backend_name(int b) {
    return dm_backend_name((DM_Backend)b);
}
*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

// ── Status helpers ────────────────────────────────────────────────────────────

func statusErr(s C.DM_Status, ctx string) error {
	if s == C.DM_OK {
		return nil
	}
	msg := C.dm_strerror(s)
	return fmt.Errorf("%s: %s (code %d)", ctx, C.GoString(msg), int(s))
}

// ── § 1  Core ─────────────────────────────────────────────────────────────────

// Version returns the libdm version string.
func Version() string { return C.GoString(C.dm_version()) }

// VersionNumber returns (major, minor, patch).
func VersionNumber() (int, int, int) {
	v := uint32(C.dm_version_number())
	return int(v >> 16), int((v >> 8) & 0xFF), int(v & 0xFF)
}

// Init initialises the library (idempotent).
func Init() error { return statusErr(C.dm_init(), "dm.Init") }

// Strerror returns the human-readable message for a status code.
func Strerror(code int) string {
	return C.GoString(C.dm_strerror(C.DM_Status(code)))
}

// ── § 8a  Backend selection ───────────────────────────────────────────────────

// DM_Backend tier constants mirror the C enum.
const (
	BackendCPU            = 0   // pure-C, always available
	BackendVulkanCompute  = 1   // Vulkan portable compute
	BackendVulkanCoopMat  = 2   // Vulkan cooperative matrix (optional)
	BackendTensorFlow     = 3   // TFE — XLA/cuDNN/oneDNN
	BackendCUDA           = 4   // CUDA (future)
	BackendROCm           = 5   // ROCm (future)
	BackendAuto           = 255 // runtime auto-detect
)

// BackendInfo mirrors DM_BackendInfo.
type BackendInfo struct {
	Active           int
	TFAvailable      bool
	VulkanAvailable  bool
	CoopMatAvailable bool
	CUDAAvailable    bool
	ROCmAvailable    bool
}

// BackendInit detects available backends and selects the best one.
// Respects the DM_BACKEND environment variable (cpu|vulkan|tensorflow|auto).
// Idempotent.
func BackendInit() { C.dm_cgo_backend_init() }

// BackendGet returns the currently active backend constant.
func BackendGet() int { return int(C.dm_cgo_backend_get()) }

// BackendSet overrides the active backend.
// Pass BackendAuto to re-run auto-detection.
func BackendSet(b int) { C.dm_cgo_backend_set(C.int(b)) }

// BackendQuery returns a full capability snapshot.
func BackendQuery() BackendInfo {
	qi := C.dm_cgo_backend_query()
	return BackendInfo{
		Active:           int(qi.active),
		TFAvailable:      qi.tf_available != 0,
		VulkanAvailable:  qi.vulkan_available != 0,
		CoopMatAvailable: qi.coop_mat_available != 0,
		CUDAAvailable:    qi.cuda_available != 0,
		ROCmAvailable:    qi.rocm_available != 0,
	}
}

// BackendName returns a human-readable name for a backend constant.
func BackendName(b int) string { return C.GoString(C.dm_cgo_backend_name(C.int(b))) }

// ── § 2  Dataset ──────────────────────────────────────────────────────────────

// Dataset wraps a DM_Dataset handle.
type Dataset struct{ h C.DM_Dataset }

// OpenDataset opens a dataset file.
//   dtype: "transactional" | "utility" | "sequence" | "quantity" | "matrix"
func OpenDataset(path, dtype string) (*Dataset, error) {
	cp := C.CString(path)
	ct := C.CString(dtype)
	defer C.free(unsafe.Pointer(cp))
	defer C.free(unsafe.Pointer(ct))

	h := C.dm_dataset_open(cp, ct)
	if h == nil {
		return nil, fmt.Errorf("dm.OpenDataset: failed to open '%s' as '%s'", path, dtype)
	}
	return &Dataset{h: h}, nil
}

func (d *Dataset) Close()        { C.dm_dataset_free(d.h) }
func (d *Dataset) Count() uint   { return uint(C.dm_dataset_count(d.h)) }
func (d *Dataset) MaxID() uint32 { return uint32(C.dm_dataset_max_id(d.h)) }

// ── § 3  Algorithm ────────────────────────────────────────────────────────────

// Algorithm wraps a DM_Algorithm handle.
type Algorithm struct{ h C.DM_Algorithm }

// NewAlgorithm creates an algorithm by ID (see dm.h § 3 for all 132 IDs).
func NewAlgorithm(id string) (*Algorithm, error) {
	cid := C.CString(id)
	defer C.free(unsafe.Pointer(cid))
	h := C.dm_algorithm_create(cid)
	if h == nil {
		return nil, fmt.Errorf("dm.NewAlgorithm: unknown id '%s'", id)
	}
	return &Algorithm{h: h}, nil
}

func (a *Algorithm) Close() { C.dm_algorithm_free(a.h) }

// Run executes the algorithm.
//
//	extraArgs may be nil or a slice of "key=value" pairs.
func (a *Algorithm) Run(datasetPath, outputPath string, minSupport float64, extraArgs []string) error {
	cdp := C.CString(datasetPath)
	cop := C.CString(outputPath)
	defer C.free(unsafe.Pointer(cdp))
	defer C.free(unsafe.Pointer(cop))

	// Build NULL-terminated char** array.
	argc := len(extraArgs)
	ptrs := make([]*C.char, argc+1)
	for i, s := range extraArgs {
		ptrs[i] = C.CString(s)
	}
	ptrs[argc] = nil
	defer func() {
		for i := range extraArgs {
			C.free(unsafe.Pointer(ptrs[i]))
		}
	}()

	var cpp **C.char
	if argc > 0 {
		cpp = (**C.char)(unsafe.Pointer(&ptrs[0]))
	}
	return statusErr(C.dm_algorithm_run(a.h, cdp, cop, C.double(minSupport), cpp),
		"dm.Algorithm.Run")
}

// ListAlgorithms returns all registered algorithm IDs.
func ListAlgorithms() ([]string, error) {
	buf := make([]byte, 32768)
	s := C.dm_algorithm_list((*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf)))
	if err := statusErr(s, "dm.ListAlgorithms"); err != nil {
		return nil, err
	}
	raw := string(buf[:C.strlen((*C.char)(unsafe.Pointer(&buf[0])))])
	return splitLines(raw), nil
}

// ── § 4  Tokenizer ────────────────────────────────────────────────────────────

// Tokenizer wraps a DM_Tokenizer handle.
type Tokenizer struct{ h C.DM_Tokenizer }

// NewTokenizer creates a tokenizer.
//
//	ttype: "bpe" | "bpe_dropout" | "unigram" | "sentencepiece" | "wordpiece"
//	       "gpe" | "parity_bpe" | "volt" | "maximal_munch" | "faro" | "tokenizer_lab"
func NewTokenizer(ttype string) (*Tokenizer, error) {
	ct := C.CString(ttype)
	defer C.free(unsafe.Pointer(ct))
	h := C.dm_tokenizer_create(ct)
	if h == nil {
		return nil, fmt.Errorf("dm.NewTokenizer: unknown type '%s'", ttype)
	}
	return &Tokenizer{h: h}, nil
}

func (t *Tokenizer) Close() { C.dm_tokenizer_free(t.h) }

func (t *Tokenizer) Train(corpusPath string, vocabSize int, outputPath string) error {
	cc := C.CString(corpusPath)
	co := C.CString(outputPath)
	defer C.free(unsafe.Pointer(cc))
	defer C.free(unsafe.Pointer(co))
	return statusErr(C.dm_tokenizer_train(t.h, cc, C.int(vocabSize), co), "dm.Tokenizer.Train")
}

func (t *Tokenizer) Load(modelPath string) error {
	cm := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cm))
	return statusErr(C.dm_tokenizer_load(t.h, cm), "dm.Tokenizer.Load")
}

// Encode converts text → token IDs.
func (t *Tokenizer) Encode(text string) ([]uint32, error) {
	ct := C.CString(text)
	defer C.free(unsafe.Pointer(ct))

	capacity := C.int(8192)
	ids := make([]C.uint32_t, 8192)
	s := C.dm_tokenizer_encode(t.h, ct,
		(*C.uint32_t)(unsafe.Pointer(&ids[0])), &capacity)

	if s == C.DM_ERR_MEMORY {
		ids = make([]C.uint32_t, int(capacity))
		s = C.dm_tokenizer_encode(t.h, ct,
			(*C.uint32_t)(unsafe.Pointer(&ids[0])), &capacity)
	}
	if err := statusErr(s, "dm.Tokenizer.Encode"); err != nil {
		return nil, err
	}
	out := make([]uint32, int(capacity))
	for i := range out {
		out[i] = uint32(ids[i])
	}
	return out, nil
}

// Decode converts token IDs → UTF-8 string.
func (t *Tokenizer) Decode(ids []uint32) (string, error) {
	cids := make([]C.uint32_t, len(ids))
	for i, v := range ids {
		cids[i] = C.uint32_t(v)
	}
	buf := make([]byte, len(ids)*8+64)
	s := C.dm_tokenizer_decode(t.h,
		(*C.uint32_t)(unsafe.Pointer(&cids[0])), C.int(len(ids)),
		(*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf)))
	if err := statusErr(s, "dm.Tokenizer.Decode"); err != nil {
		return "", err
	}
	n := C.strlen((*C.char)(unsafe.Pointer(&buf[0])))
	return string(buf[:n]), nil
}

func (t *Tokenizer) VocabSize() int {
	return int(C.dm_tokenizer_vocab_size(t.h))
}

func (t *Tokenizer) TokenText(id uint32) string {
	var length C.uint32_t
	ptr := C.dm_tokenizer_token_text(t.h, C.uint32_t(id), &length)
	if ptr == nil {
		return ""
	}
	return C.GoStringN(ptr, C.int(length))
}

// VoltRun runs VOLT vocabulary learning.
func VoltRun(corpusPath string, minSize, maxSize, nSteps int, outputPath string) error {
	cc := C.CString(corpusPath)
	co := C.CString(outputPath)
	defer C.free(unsafe.Pointer(cc))
	defer C.free(unsafe.Pointer(co))
	return statusErr(C.dm_tokenizer_volt_run(cc, C.int(minSize), C.int(maxSize),
		C.int(nSteps), co), "dm.VoltRun")
}

// ── § 5  Vision ───────────────────────────────────────────────────────────────

// Vision wraps a DM_Vision handle.
type Vision struct{ h C.DM_Vision }

// NewVision creates a vision model.  modelType: "mobilenet_tiny"
func NewVision(modelType string) (*Vision, error) {
	cm := C.CString(modelType)
	defer C.free(unsafe.Pointer(cm))
	h := C.dm_vision_create(cm)
	if h == nil {
		return nil, errors.New("dm.NewVision: create failed")
	}
	return &Vision{h: h}, nil
}

func (v *Vision) Close() { C.dm_vision_free(v.h) }

func (v *Vision) Init(savedModelDir string, classes, imageSize int,
	widthMult, learningRate float32) error {
	cs := C.CString(savedModelDir)
	defer C.free(unsafe.Pointer(cs))
	return statusErr(C.dm_vision_init(v.h, cs, C.int(classes),
		C.int(imageSize), C.float(widthMult), C.float(learningRate)),
		"dm.Vision.Init")
}

func (v *Vision) Train(manifestPath string, epochs, batchSize int, lr float32) error {
	cm := C.CString(manifestPath)
	defer C.free(unsafe.Pointer(cm))
	return statusErr(C.dm_vision_train(v.h, cm, C.int(epochs),
		C.int(batchSize), C.float(lr)), "dm.Vision.Train")
}

// EvalResult holds the results of an evaluation pass.
type EvalResult struct{ Loss, Accuracy float32 }

func (v *Vision) Eval(manifestPath string) (EvalResult, error) {
	cm := C.CString(manifestPath)
	defer C.free(unsafe.Pointer(cm))
	var loss, acc C.float
	err := statusErr(C.dm_vision_eval(v.h, cm, &loss, &acc), "dm.Vision.Eval")
	return EvalResult{float32(loss), float32(acc)}, err
}

// Predict runs inference on row-major float32 RGB [h*w*3] in [0,1].
func (v *Vision) Predict(rgb []float32, h, w, nClasses int) ([]float32, error) {
	crgb := make([]C.float, len(rgb))
	for i, f := range rgb {
		crgb[i] = C.float(f)
	}
	probs := make([]C.float, nClasses)
	err := statusErr(C.dm_vision_predict(v.h,
		(*C.float)(unsafe.Pointer(&crgb[0])), C.int(h), C.int(w),
		(*C.float)(unsafe.Pointer(&probs[0])), C.int(nClasses)),
		"dm.Vision.Predict")
	if err != nil {
		return nil, err
	}
	out := make([]float32, nClasses)
	for i, f := range probs {
		out[i] = float32(f)
	}
	return out, nil
}

func (v *Vision) Save() error {
	return statusErr(C.dm_vision_save(v.h), "dm.Vision.Save")
}

// ── § 6  Language Model ───────────────────────────────────────────────────────

// LM wraps a DM_LM handle.
type LM struct{ h C.DM_LM }

// NewLM creates a language model.  modelType: "bert" | "tiny_transformer" | "tinystories"
func NewLM(modelType string) (*LM, error) {
	cm := C.CString(modelType)
	defer C.free(unsafe.Pointer(cm))
	h := C.dm_lm_create(cm)
	if h == nil {
		return nil, fmt.Errorf("dm.NewLM: create failed for type '%s'", modelType)
	}
	return &LM{h: h}, nil
}

func (l *LM) Close() { C.dm_lm_free(l.h) }

func (l *LM) Train(corpusPath, checkpointDir string,
	epochs, batchSize int, lr float32) error {
	cc := C.CString(corpusPath)
	cd := C.CString(checkpointDir)
	defer C.free(unsafe.Pointer(cc))
	defer C.free(unsafe.Pointer(cd))
	return statusErr(C.dm_lm_train(l.h, cc, cd,
		C.int(epochs), C.int(batchSize), C.float(lr)), "dm.LM.Train")
}

func (l *LM) Load(checkpointDir string) error {
	cd := C.CString(checkpointDir)
	defer C.free(unsafe.Pointer(cd))
	return statusErr(C.dm_lm_load(l.h, cd), "dm.LM.Load")
}

// Generate produces text from prompt. For BERT, prompt is space-separated token IDs
// and the returned string contains pooled [CLS] values.
func (l *LM) Generate(prompt string, maxTokens int) (string, error) {
	cp := C.CString(prompt)
	defer C.free(unsafe.Pointer(cp))
	buf := make([]byte, maxTokens*4+256)
	err := statusErr(C.dm_lm_generate(l.h, cp, C.int(maxTokens),
		(*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))),
		"dm.LM.Generate")
	if err != nil {
		return "", err
	}
	n := C.strlen((*C.char)(unsafe.Pointer(&buf[0])))
	return string(buf[:n]), nil
}

// ── § 9  Benchmark ────────────────────────────────────────────────────────────

// BenchPhase selects the phase to measure.
type BenchPhase = C.DM_BenchPhase

const (
	BenchPhaseLoad  BenchPhase = C.DM_BENCH_PHASE_LOAD
	BenchPhaseAlgo  BenchPhase = C.DM_BENCH_PHASE_ALGO
	BenchPhaseWrite BenchPhase = C.DM_BENCH_PHASE_WRITE
	BenchPhaseTotal BenchPhase = C.DM_BENCH_PHASE_TOTAL
)

// BenchReport mirrors DM_BenchReport.
type BenchReport struct {
	PhaseMS           [4]float64
	PeakKB            uint64
	UserMS, SysMS     float64
	ResultRAM         uint64
	ResultDisk        uint64
	NumPatterns       uint64
	TotalItems        uint64
	ThroughputMBPerS  float64
}

func BenchReset()                    { C.dm_bench_reset() }
func BenchStart(p BenchPhase)        { C.dm_bench_start(p) }
func BenchStop(p BenchPhase)         { C.dm_bench_stop(p) }
func BenchRecord(n, t uint)          { C.dm_bench_record(C.size_t(n), C.size_t(t)) }

func GetBenchReport() BenchReport {
	r := C.dm_bench_get_report()
	var out BenchReport
	for i := 0; i < 4; i++ {
		out.PhaseMS[i] = float64(r.phase_times_ms[i])
	}
	out.PeakKB           = uint64(r.peak_memory_kb)
	out.UserMS           = float64(r.user_cpu_ms)
	out.SysMS            = float64(r.sys_cpu_ms)
	out.ResultRAM        = uint64(r.result_ram_bytes)
	out.ResultDisk       = uint64(r.result_disk_est_bytes)
	out.NumPatterns      = uint64(r.num_patterns)
	out.TotalItems       = uint64(r.total_items)
	out.ThroughputMBPerS = float64(r.throughput_mb_s)
	return out
}

func BenchPrint(algo, dataset string) {
	ca := C.CString(algo)
	cd := C.CString(dataset)
	defer C.free(unsafe.Pointer(ca))
	defer C.free(unsafe.Pointer(cd))
	C.dm_bench_print(ca, cd)
}

// ── § 10  BitSet ──────────────────────────────────────────────────────────────

// BitSet wraps a DM_BitSet handle.
type BitSet struct{ h C.DM_BitSet }

// NewBitSet creates a bit array of n bits (all zero).
func NewBitSet(nBits uint) (*BitSet, error) {
	h := C.dm_bitset_create(C.size_t(nBits))
	if h == nil {
		return nil, errors.New("dm.NewBitSet: allocation failed")
	}
	return &BitSet{h: h}, nil
}

func (b *BitSet) Copy() (*BitSet, error) {
	h := C.dm_bitset_copy(b.h)
	if h == nil {
		return nil, errors.New("dm.BitSet.Copy: allocation failed")
	}
	return &BitSet{h: h}, nil
}

func (b *BitSet) Close()             { C.dm_bitset_free(b.h) }
func (b *BitSet) Set(pos uint)       { C.dm_bitset_set(b.h, C.size_t(pos)) }
func (b *BitSet) Clear(pos uint)     { C.dm_bitset_clear(b.h, C.size_t(pos)) }
func (b *BitSet) Get(pos uint) bool  { return bool(C.dm_bitset_get(b.h, C.size_t(pos))) }
func (b *BitSet) SetAll()            { C.dm_bitset_set_all(b.h) }
func (b *BitSet) Flip()              { C.dm_bitset_not(b.h) }
func (b *BitSet) Popcount() uint     { return uint(C.dm_bitset_popcount(b.h)) }
func (b *BitSet) And(o *BitSet)      { C.dm_bitset_and(b.h, o.h) }
func (b *BitSet) Or(o *BitSet)       { C.dm_bitset_or(b.h, o.h) }

// ── § 11  DataGen ─────────────────────────────────────────────────────────────

// DataGen wraps a DM_DataGen handle.
type DataGen struct{ h C.DM_DataGen }

// NewDataGen creates a data generator.  dtype: "medm" | "textbook"
func NewDataGen(dtype string) (*DataGen, error) {
	cd := C.CString(dtype)
	defer C.free(unsafe.Pointer(cd))
	h := C.dm_datagen_create(cd)
	if h == nil {
		return nil, fmt.Errorf("dm.NewDataGen: unknown type '%s'", dtype)
	}
	return &DataGen{h: h}, nil
}

func (g *DataGen) Close() { C.dm_datagen_free(g.h) }

func (g *DataGen) Run(spec, outputPath string, seed uint) error {
	cs := C.CString(spec)
	co := C.CString(outputPath)
	defer C.free(unsafe.Pointer(cs))
	defer C.free(unsafe.Pointer(co))
	return statusErr(C.dm_datagen_run(g.h, cs, co, C.uint(seed)), "dm.DataGen.Run")
}

// ── § 14  GPU ─────────────────────────────────────────────────────────────────

// GpuCtx wraps a DM_GpuCtx handle (soft-fail: nil handle = no GPU).
type GpuCtx struct{ h C.DM_GpuCtx }

// NewGpuCtx creates a Vulkan compute context.
//   deviceIndex: 0 = first discrete GPU, -1 = driver pick.
//   shaderDir:   "" = automatic search.
func NewGpuCtx(deviceIndex int, shaderDir string) *GpuCtx {
	var cs *C.char
	if shaderDir != "" {
		cs = C.CString(shaderDir)
		defer C.free(unsafe.Pointer(cs))
	}
	return &GpuCtx{h: C.dm_gpu_create(C.int(deviceIndex), cs)}
}

func (g *GpuCtx) Close() {
	if g.h != nil {
		C.dm_gpu_free(g.h)
		g.h = nil
	}
}

func (g *GpuCtx) Ready() bool {
	return g.h != nil && C.dm_gpu_ready(g.h) != 0
}

func (g *GpuCtx) DeviceName() string {
	if g.h == nil {
		return "(no GPU)"
	}
	buf := make([]byte, 256)
	C.dm_gpu_device_name(g.h, (*C.char)(unsafe.Pointer(&buf[0])), 256)
	n := C.strlen((*C.char)(unsafe.Pointer(&buf[0])))
	return string(buf[:n])
}

// ── § 18  Experiment ──────────────────────────────────────────────────────────

func TimerNow() float64 { return float64(C.dm_timer_now()) }
func PeakRAMMB() float64 { return float64(C.dm_peak_ram_mb()) }
func FileSizeMB(path string) float64 {
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	return float64(C.dm_file_size_mb(cp))
}
func DirSizeMB(path string) float64 {
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	return float64(C.dm_dir_size_mb(cp))
}
func EnsureDir(path string) error {
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	if C.dm_ensure_dir(cp) != 0 {
		return fmt.Errorf("dm.EnsureDir: failed for '%s'", path)
	}
	return nil
}
func PathBasename(path string) string {
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	return C.GoString(C.dm_path_basename(cp))
}
func ExperimentWriteHeader(csvPath string) {
	cp := C.CString(csvPath)
	defer C.free(unsafe.Pointer(cp))
	C.dm_experiment_write_header(cp)
}
func ExperimentGenerateReport(resultsRoot string) {
	cr := C.CString(resultsRoot)
	defer C.free(unsafe.Pointer(cr))
	C.dm_experiment_generate_report(cr)
}

// ── § 13  CLI ─────────────────────────────────────────────────────────────────

// CLIRun executes a dm sub-command.
func CLIRun(command string, args ...string) int {
	cc := C.CString(command)
	defer C.free(unsafe.Pointer(cc))

	ptrs := make([]*C.char, len(args)+1)
	for i, a := range args {
		ptrs[i] = C.CString(a)
	}
	ptrs[len(args)] = nil
	defer func() {
		for i := range args {
			C.free(unsafe.Pointer(ptrs[i]))
		}
	}()

	var argv **C.char
	if len(args) > 0 {
		argv = (**C.char)(unsafe.Pointer(&ptrs[0]))
	}
	return int(C.dm_cli_run(cc, C.int(len(args)), argv))
}

// ── utilities ─────────────────────────────────────────────────────────────────

func splitLines(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			if i > start {
				out = append(out, s[start:i])
			}
			start = i + 1
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}

// ── § 8  Engine — dm_engine ops (TFE backend, NCHW float32) ──────────────────
//
// Users compose custom models from these primitives the same way they would
// compose TensorFlow layers — everything dispatches through TFE so XLA,
// cuDNN, and oneDNN acceleration is automatic.
//
// Layout conventions:
//   Tensors      — NCHW (n, c, h, w), row-major, contiguous float32
//   Conv weights — OIHW [out_c][in_c][ky][kx]
//   Linear weights — [out][in]
//
// Quick example:
//   x := dm.NewTensor(1, 3, 224, 224)
//   y := dm.NewTensor(1, 64, 112, 112)
//   defer x.Free(); defer y.Free()
//   dm.OpConv2dSame(x, y, weights, bias, 64, 3, 2)
//   dm.OpRelu(y)

// Tensor wraps a C DM_Tensor (NCHW float32, owned).
type Tensor struct {
	t C.DM_Tensor
}

// NewTensor allocates a new NCHW float32 tensor.
func NewTensor(n, ch, h, w int) (*Tensor, error) {
	t := &Tensor{}
	s := C.dm_tensor_alloc(&t.t, C.int(n), C.int(ch), C.int(h), C.int(w))
	if err := statusErr(s, "dm.NewTensor"); err != nil {
		return nil, err
	}
	return t, nil
}

// Free releases the native buffer.
func (t *Tensor) Free() { C.dm_tensor_free(&t.t) }

func (t *Tensor) N() int     { return int(t.t.n) }
func (t *Tensor) C() int     { return int(t.t.c) }
func (t *Tensor) H() int     { return int(t.t.h) }
func (t *Tensor) W() int     { return int(t.t.w) }
func (t *Tensor) Count() int { return int(C.dm_tensor_count(&t.t)) }

// Fill sets every element to v.
func (t *Tensor) Fill(v float32) { C.dm_tensor_fill(&t.t, C.float(v)) }

// Get returns element at (n, c, y, x).
func (t *Tensor) Get(n, c, y, x int) float32 {
	return float32(C.dm_tensor_get(&t.t, C.int(n), C.int(c), C.int(y), C.int(x)))
}

// Set writes v at (n, c, y, x).
func (t *Tensor) Set(n, c, y, x int, v float32) {
	C.dm_tensor_set(&t.t, C.int(n), C.int(c), C.int(y), C.int(x), C.float(v))
}

// ToSlice copies the tensor data into a new []float32.
func (t *Tensor) ToSlice() []float32 {
	n := t.Count()
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(*(*C.float)(unsafe.Pointer(
			uintptr(unsafe.Pointer(t.t.data)) + uintptr(i)*unsafe.Sizeof(C.float(0)))))
	}
	return out
}

func floatSlice(v []float32) *C.float {
	if len(v) == 0 {
		return nil
	}
	return (*C.float)(unsafe.Pointer(&v[0]))
}

func intSlice(v []int32) *C.int {
	if len(v) == 0 {
		return nil
	}
	return (*C.int)(unsafe.Pointer(&v[0]))
}

// ── Convolutions ──────────────────────────────────────────────────────────────

// OpConv2dSame runs a standard conv2d with SAME padding. w: OIHW layout.
func OpConv2dSame(in, out *Tensor, w, b []float32, outC, kernel, stride int) error {
	return statusErr(
		C.dm_op_conv2d_same(&in.t, &out.t, floatSlice(w), floatSlice(b),
			C.int(outC), C.int(kernel), C.int(stride)),
		"dm.OpConv2dSame")
}

// OpDepthwiseConv runs a depthwise separable conv with SAME padding. w: [c][ky][kx].
func OpDepthwiseConv(in, out *Tensor, w, b []float32, kernel, stride int) error {
	return statusErr(
		C.dm_op_depthwise_conv(&in.t, &out.t, floatSlice(w), floatSlice(b),
			C.int(kernel), C.int(stride)),
		"dm.OpDepthwiseConv")
}

// OpPointwiseConv runs a 1×1 conv. w: [out_c][in_c].
func OpPointwiseConv(in, out *Tensor, w, b []float32, outC int) error {
	return statusErr(
		C.dm_op_pointwise_conv(&in.t, &out.t, floatSlice(w), floatSlice(b), C.int(outC)),
		"dm.OpPointwiseConv")
}

// ── Linear ────────────────────────────────────────────────────────────────────

// OpLinear is a fully-connected layer. in: [n,in_c,1,1] → out: [n,out_c,1,1].
func OpLinear(in, out *Tensor, w, b []float32, outC int) error {
	return statusErr(
		C.dm_op_linear(&in.t, &out.t, floatSlice(w), floatSlice(b), C.int(outC)),
		"dm.OpLinear")
}

// ── Pooling ───────────────────────────────────────────────────────────────────

// OpGlobalAvgPool computes global average pooling over H×W.
func OpGlobalAvgPool(in, out *Tensor) error {
	return statusErr(C.dm_op_global_avg_pool(&in.t, &out.t), "dm.OpGlobalAvgPool")
}

// OpMaxPool2dSame computes max pooling with SAME padding.
func OpMaxPool2dSame(in, out *Tensor, kernel, stride int) error {
	return statusErr(
		C.dm_op_max_pool2d_same(&in.t, &out.t, C.int(kernel), C.int(stride)),
		"dm.OpMaxPool2dSame")
}

// ── Normalisation ─────────────────────────────────────────────────────────────

// OpBatchNorm applies batch normalisation in-place.
func OpBatchNorm(t *Tensor, gamma, beta, mean, variance []float32, eps float32) error {
	return statusErr(
		C.dm_op_batch_norm(&t.t,
			floatSlice(gamma), floatSlice(beta),
			floatSlice(mean), floatSlice(variance), C.float(eps)),
		"dm.OpBatchNorm")
}

// OpLayerNorm applies layer normalisation in-place on x [seqLen × dModel].
func OpLayerNorm(x []float32, seqLen, dModel int,
	gamma, beta []float32, eps float32) error {
	return statusErr(
		C.dm_op_layer_norm(floatSlice(x), C.int(seqLen), C.int(dModel),
			floatSlice(gamma), floatSlice(beta), C.float(eps)),
		"dm.OpLayerNorm")
}

// ── Elementwise ───────────────────────────────────────────────────────────────

// OpAdd computes out += in element-wise.
func OpAdd(out, in *Tensor) error {
	return statusErr(C.dm_op_tensor_add(&out.t, &in.t), "dm.OpAdd")
}

// ── Activations ───────────────────────────────────────────────────────────────

func OpRelu   (t *Tensor) { C.dm_op_relu(&t.t)    }
func OpRelu6  (t *Tensor) { C.dm_op_relu6(&t.t)   }
func OpTanh   (t *Tensor) { C.dm_op_tanh(&t.t)    }
func OpSigmoid(t *Tensor) { C.dm_op_sigmoid(&t.t) }

// OpGelu applies GELU in-place to the raw slice.
func OpGelu(x []float32) { C.dm_op_gelu(floatSlice(x), C.int(len(x))) }

// ── Softmax ───────────────────────────────────────────────────────────────────

// OpSoftmax applies softmax over the channel dim of an [n,c,1,1] tensor.
func OpSoftmax(t *Tensor) { C.dm_op_softmax(&t.t) }

// OpSoftmaxRows applies softmax row-wise over a [rows × cols] raw slice.
func OpSoftmaxRows(x []float32, rows, cols int) {
	C.dm_op_softmax_rows(floatSlice(x), C.int(rows), C.int(cols))
}

// ── Matrix multiplication ─────────────────────────────────────────────────────

// OpMatmulNT computes C = A × Bᵀ  (A[M×K], B[N×K] → C[M×N]).
func OpMatmulNT(A, B []float32, M, N, K int) []float32 {
	C_ := make([]float32, M*N)
	C.dm_op_matmul_nt(floatSlice(A), floatSlice(B), floatSlice(C_),
		C.int(M), C.int(N), C.int(K))
	return C_
}

// OpMatmulNN computes C = A × B  (A[M×K], B[K×N] → C[M×N]).
func OpMatmulNN(A, B []float32, M, K, N int) []float32 {
	C_ := make([]float32, M*N)
	C.dm_op_matmul_nn(floatSlice(A), floatSlice(B), floatSlice(C_),
		C.int(M), C.int(K), C.int(N))
	return C_
}

// ── Backward passes ───────────────────────────────────────────────────────────

func OpLinearBackward(in, gradOut, gradIn *Tensor,
	gradW, gradB, w []float32, outC int) error {
	return statusErr(
		C.dm_op_linear_backward(&in.t, &gradOut.t, &gradIn.t,
			floatSlice(gradW), floatSlice(gradB), floatSlice(w), C.int(outC)),
		"dm.OpLinearBackward")
}

func OpReluBackward(in, gradOut, gradIn *Tensor) {
	C.dm_op_relu_backward(&in.t, &gradOut.t, &gradIn.t)
}

func OpTanhBackward(out_, gradOut, gradIn *Tensor) {
	C.dm_op_tanh_backward(&out_.t, &gradOut.t, &gradIn.t)
}

// ── Maxout ────────────────────────────────────────────────────────────────────

// OpMaxout applies maxout pooling (groups of k) and returns the argmax buffer.
func OpMaxout(in, out *Tensor, k int) ([]int32, error) {
	argmax := make([]int32, in.N()*in.C()/k)
	err := statusErr(
		C.dm_op_maxout(&in.t, &out.t, C.int(k), intSlice(argmax)),
		"dm.OpMaxout")
	return argmax, err
}

func OpMaxoutBackward(gradOut, gradIn *Tensor, k int, argmax []int32) error {
	return statusErr(
		C.dm_op_maxout_backward(&gradOut.t, &gradIn.t, C.int(k), intSlice(argmax)),
		"dm.OpMaxoutBackward")
}

// ── Dropout ───────────────────────────────────────────────────────────────────

// OpDropout applies inverted dropout and returns the mask buffer.
func OpDropout(in, out *Tensor, dropProb float32) []int32 {
	mask := make([]int32, in.Count())
	C.dm_op_dropout(&in.t, &out.t, C.float(dropProb), intSlice(mask))
	return mask
}

func OpDropoutBackward(gradOut, gradIn *Tensor, dropProb float32, mask []int32) {
	C.dm_op_dropout_backward(&gradOut.t, &gradIn.t, C.float(dropProb), intSlice(mask))
}

// ── Optimisers ────────────────────────────────────────────────────────────────

// OpAdamStep performs an in-place Adam update. t is the 1-indexed step number.
func OpAdamStep(param, grad, m, v []float32,
	lr, beta1, beta2, eps, weightDecay float32, t int) {
	C.dm_op_adam_step(
		floatSlice(param), floatSlice(grad), floatSlice(m), floatSlice(v),
		C.int(len(param)), C.float(lr),
		C.float(beta1), C.float(beta2), C.float(eps),
		C.float(weightDecay), C.int(t))
}

// OpAdagradStep performs an in-place Adagrad update.
func OpAdagradStep(param, grad, gSum []float32, lr, eps, weightDecay float32) {
	C.dm_op_adagrad_step(
		floatSlice(param), floatSlice(grad), floatSlice(gSum),
		C.int(len(param)), C.float(lr), C.float(eps), C.float(weightDecay))
}

// OpSgdMomentumStep performs an in-place SGD+momentum update.
func OpSgdMomentumStep(param, grad, velocity []float32,
	lr, momentum, weightDecay float32, nesterov bool) {
	n := 0
	if nesterov {
		n = 1
	}
	C.dm_op_sgd_momentum_step(
		floatSlice(param), floatSlice(grad), floatSlice(velocity),
		C.int(len(param)), C.float(lr), C.float(momentum),
		C.float(weightDecay), C.int(n))
}

// ═══════════════════════════════════════════════════════════════════════════
// Models — pre-built model wrappers
// ═══════════════════════════════════════════════════════════════════════════

// ── Vision ────────────────────────────────────────────────────────────────

// ResNet wraps the ResNet-18 image classifier.
type ResNet struct {
	Classes int
	Seed    uint32
}

// NewResNet creates a ResNet-18 classifier.
func NewResNet(classes int, seed uint32) *ResNet {
	return &ResNet{Classes: classes, Seed: seed}
}

// Forward runs ResNet-18.  input is an NCHW tensor (1,3,H,W).
func (m *ResNet) Forward(input *Tensor) ([]float32, error) {
	out := NewTensor(1, m.Classes, 1, 1)
	defer out.Free()
	err := statusErr(
		C.dm_op_resnet18_forward(&input.t, &out.t, C.int(m.Classes), C.uint(m.Seed)),
		"dm.ResNet.Forward")
	if err != nil {
		return nil, err
	}
	return out.ToSlice(), nil
}

// ViTVariant enumerates Vision Transformer size variants.
type ViTVariant int

const (
	ViTTiny  ViTVariant = 0
	ViTSmall ViTVariant = 1
	ViTBase  ViTVariant = 2
	ViTLarge ViTVariant = 3
	ViTHuge  ViTVariant = 4
)

// ViT wraps the Vision Transformer model.
type ViT struct {
	Variant  ViTVariant
	Classes  int
	ImgSize  int
	Seed     uint32
}

// NewViT creates a ViT model.
func NewViT(variant ViTVariant, classes, imgSize int, seed uint32) *ViT {
	return &ViT{Variant: variant, Classes: classes, ImgSize: imgSize, Seed: seed}
}

// Forward runs ViT inference.  input is an NCHW tensor (1,3,H,W).
func (m *ViT) Forward(input *Tensor) ([]float32, error) {
	out := NewTensor(1, m.Classes, 1, 1)
	defer out.Free()
	err := statusErr(
		C.dm_op_vit_forward(&input.t, &out.t,
			C.int(m.Variant), C.int(m.Classes), C.uint(m.Seed)),
		"dm.ViT.Forward")
	if err != nil {
		return nil, err
	}
	return out.ToSlice(), nil
}

// WeightCount returns the number of float32 parameters for this ViT config.
func (m *ViT) WeightCount() int {
	return int(C.dm_op_vit_param_count(C.int(m.Variant), C.int(m.ImgSize), 16, C.int(m.Classes)))
}

// MobileNetTiny wraps the MobileNetV4-Tiny classifier.
type MobileNetTiny struct {
	ImageSize int
	Classes   int
	Seed      uint32
}

// NewMobileNetTiny creates a MobileNetV4-Tiny model.
func NewMobileNetTiny(imgSize, classes int, seed uint32) *MobileNetTiny {
	return &MobileNetTiny{ImageSize: imgSize, Classes: classes, Seed: seed}
}

// Forward runs MobileNetTiny inference.  inputNCHW is a flat slice [3×H×W].
func (m *MobileNetTiny) Forward(inputNCHW []float32) ([]float32, error) {
	logits := make([]float32, m.Classes)
	err := statusErr(
		C.dm_mobilenet_tiny_forward_raw2(
			(*C.float)(unsafe.Pointer(&inputNCHW[0])),
			C.int(m.ImageSize), C.int(m.Classes), C.uint(m.Seed),
			(*C.float)(unsafe.Pointer(&logits[0]))),
		"dm.MobileNetTiny.Forward")
	return logits, err
}

// ── Language ──────────────────────────────────────────────────────────────

// BERTVariant enumerates BERT model sizes.
type BERTVariant int

const (
	BERTBase  BERTVariant = 0
	BERTLarge BERTVariant = 1
)

// BERTModel wraps the BERT encoder.
type BERTModel struct {
	Variant    BERTVariant
	VocabSize  int
	MaxSeqLen  int
	weights    *C.float
	ownsWeights bool
}

// NewBERT creates a BERT model handle.
func NewBERT(variant BERTVariant, vocabSize, maxSeqLen int) *BERTModel {
	return &BERTModel{Variant: variant, VocabSize: vocabSize, MaxSeqLen: maxSeqLen}
}

// Load reads weights from a file.
func (m *BERTModel) Load(path string) error {
	m.freeWeights()
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	var v, vs, ms C.int
	var ptr *C.float
	rc := C.dm_bert_load_raw(cp, &v, &vs, &ms, &ptr)
	if err := statusErr(rc, "dm.BERT.Load"); err != nil {
		return err
	}
	m.Variant = BERTVariant(v)
	m.VocabSize = int(vs)
	m.MaxSeqLen = int(ms)
	m.weights = ptr
	m.ownsWeights = true
	return nil
}

// Free releases loaded weights.
func (m *BERTModel) Free() { m.freeWeights() }

func (m *BERTModel) freeWeights() {
	if m.weights != nil && m.ownsWeights {
		C.dm_bert_free_weights(m.weights)
		m.weights = nil
	}
}

// WeightCount returns the total number of float32 parameters.
func (m *BERTModel) WeightCount() int {
	return int(C.dm_bert_weight_count_raw(C.int(m.Variant), C.int(m.VocabSize), C.int(m.MaxSeqLen)))
}

// Forward runs BERT encoding.  Returns (hidden [seq×H], cls [H]).
func (m *BERTModel) Forward(tokenIDs, segmentIDs []int32) ([]float32, []float32, error) {
	if m.weights == nil {
		return nil, nil, errors.New("dm.BERT: no weights loaded")
	}
	seq := len(tokenIDs)
	H := 768
	if m.Variant == BERTLarge {
		H = 1024
	}
	hidden := make([]float32, seq*H)
	cls := make([]float32, H)
	rc := C.dm_bert_forward_raw(
		C.int(m.Variant), C.int(m.VocabSize), C.int(m.MaxSeqLen), m.weights,
		(*C.int)(unsafe.Pointer(&tokenIDs[0])),
		(*C.int)(unsafe.Pointer(&segmentIDs[0])),
		C.int(seq),
		(*C.float)(unsafe.Pointer(&hidden[0])),
		(*C.float)(unsafe.Pointer(&cls[0])))
	return hidden, cls, statusErr(rc, "dm.BERT.Forward")
}

// TransformerVariant enumerates Transformer sizes.
type TransformerVariant int

const (
	TransformerBase TransformerVariant = 0
	TransformerBig  TransformerVariant = 1
)

// TransformerModel wraps the full encoder-decoder Transformer.
type TransformerModel struct {
	Variant    TransformerVariant
	VocabSize  int
	MaxSeqLen  int
	weights    *C.float
	ownsWeights bool
}

// NewTransformer creates a Transformer model handle.
func NewTransformer(variant TransformerVariant, vocabSize, maxSeqLen int) *TransformerModel {
	return &TransformerModel{Variant: variant, VocabSize: vocabSize, MaxSeqLen: maxSeqLen}
}

// Free releases loaded weights.
func (m *TransformerModel) Free() { m.freeWeights() }

func (m *TransformerModel) freeWeights() {
	if m.weights != nil && m.ownsWeights {
		C.dm_transformer_free_weights(m.weights)
		m.weights = nil
	}
}

// Load reads weights from a .bin file.
func (m *TransformerModel) Load(path string) error {
	m.freeWeights()
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	var v, vs, ms C.int
	var ptr *C.float
	rc := C.dm_transformer_load_raw(cp, &v, &vs, &ms, &ptr)
	if err := statusErr(rc, "dm.Transformer.Load"); err != nil {
		return err
	}
	m.Variant = TransformerVariant(v)
	m.VocabSize = int(vs)
	m.MaxSeqLen = int(ms)
	m.weights = ptr
	m.ownsWeights = true
	return nil
}

// WeightCount returns the total number of float32 parameters.
func (m *TransformerModel) WeightCount() int {
	return int(C.dm_transformer_weight_count_raw(
		C.int(m.Variant), C.int(m.VocabSize), C.int(m.MaxSeqLen)))
}

// Forward runs a full encoder-decoder pass.  Returns logits [tgtSeq × vocabSize].
func (m *TransformerModel) Forward(srcTokens, tgtTokens []int32) ([]float32, error) {
	if m.weights == nil {
		return nil, errors.New("dm.Transformer: no weights loaded")
	}
	tgtSeq := len(tgtTokens)
	logits := make([]float32, tgtSeq*m.VocabSize)
	rc := C.dm_transformer_forward_raw(
		C.int(m.Variant), C.int(m.VocabSize), C.int(m.MaxSeqLen), m.weights,
		(*C.int)(unsafe.Pointer(&srcTokens[0])), C.int(len(srcTokens)),
		(*C.int)(unsafe.Pointer(&tgtTokens[0])), C.int(tgtSeq),
		(*C.float)(unsafe.Pointer(&logits[0])))
	return logits, statusErr(rc, "dm.Transformer.Forward")
}

// Encode runs only the encoder stack.  Returns enc_out [srcSeq × dModel].
func (m *TransformerModel) Encode(srcTokens []int32) ([]float32, error) {
	if m.weights == nil {
		return nil, errors.New("dm.Transformer: no weights loaded")
	}
	dModel := 512
	if m.Variant == TransformerBig {
		dModel = 1024
	}
	enc := make([]float32, len(srcTokens)*dModel)
	rc := C.dm_transformer_encode_raw(
		C.int(m.Variant), C.int(m.VocabSize), C.int(m.MaxSeqLen), m.weights,
		(*C.int)(unsafe.Pointer(&srcTokens[0])), C.int(len(srcTokens)),
		(*C.float)(unsafe.Pointer(&enc[0])))
	return enc, statusErr(rc, "dm.Transformer.Encode")
}

// TransformerLRSchedule computes the paper's warmup learning-rate at a given step.
func TransformerLRSchedule(dModel, step, warmupSteps int) float32 {
	return float32(C.dm_transformer_lr_schedule_raw(C.int(dModel), C.int(step), C.int(warmupSteps)))
}

// TransformerPositionalEncoding returns sinusoidal PE table [maxLen × dModel].
func TransformerPositionalEncoding(maxLen, dModel int) []float32 {
	pe := make([]float32, maxLen*dModel)
	C.dm_transformer_positional_encoding_raw(C.int(maxLen), C.int(dModel),
		(*C.float)(unsafe.Pointer(&pe[0])))
	return pe
}

// TransformerCausalMask returns an upper-triangular causal mask [seq × seq].
func TransformerCausalMask(seq int) []float32 {
	mask := make([]float32, seq*seq)
	C.dm_transformer_causal_mask_raw(C.int(seq), (*C.float)(unsafe.Pointer(&mask[0])))
	return mask
}

// ── Generative ────────────────────────────────────────────────────────────

// VAEModel wraps a Variational Auto-Encoder.
type VAEModel struct {
	InputDim  int
	LatentDim int
	handle    unsafe.Pointer
}

// NewVAE creates a VAE with the given dimensions.
func NewVAE(inputDim, hiddenDim, latentDim int, lr float32) (*VAEModel, error) {
	h := C.dm_vae_create_raw(C.int(inputDim), C.int(hiddenDim),
		C.int(latentDim), C.float(lr))
	if h == nil {
		return nil, errors.New("dm.VAE: allocation failed")
	}
	return &VAEModel{InputDim: inputDim, LatentDim: latentDim, handle: h}, nil
}

// Free releases the VAE.
func (m *VAEModel) Free() {
	if m.handle != nil {
		C.dm_vae_free_raw(m.handle)
		m.handle = nil
	}
}

// TrainStep runs one forward+backward step.  Returns the ELBO loss.
func (m *VAEModel) TrainStep(xBatch []float32) float32 {
	batch := len(xBatch) / m.InputDim
	return float32(C.dm_vae_train_step_raw(m.handle,
		(*C.float)(unsafe.Pointer(&xBatch[0])), C.int(batch)))
}

// Encode returns (mean, logvar) for the input batch.
func (m *VAEModel) Encode(xBatch []float32) ([]float32, []float32) {
	batch := len(xBatch) / m.InputDim
	mean := make([]float32, batch*m.LatentDim)
	logvar := make([]float32, batch*m.LatentDim)
	C.dm_vae_encode_raw(m.handle,
		(*C.float)(unsafe.Pointer(&xBatch[0])), C.int(batch),
		(*C.float)(unsafe.Pointer(&mean[0])),
		(*C.float)(unsafe.Pointer(&logvar[0])))
	return mean, logvar
}

// Decode reconstructs the input from latent vectors.
func (m *VAEModel) Decode(zBatch []float32) []float32 {
	batch := len(zBatch) / m.LatentDim
	out := make([]float32, batch*m.InputDim)
	C.dm_vae_decode_raw(m.handle,
		(*C.float)(unsafe.Pointer(&zBatch[0])), C.int(batch),
		(*C.float)(unsafe.Pointer(&out[0])))
	return out
}

// GANModel wraps a Generative Adversarial Network.
type GANModel struct {
	InputDim int
	NoiseDim int
	handle   unsafe.Pointer
}

// NewGAN creates a GAN.
func NewGAN(inputDim, gHidden, noiseDim, dHidden, maxoutK int,
	dropProb, lr, momentum float32, nesterov bool) (*GANModel, error) {
	n := 0
	if nesterov {
		n = 1
	}
	h := C.dm_gan_create_raw(
		C.int(inputDim), C.int(gHidden), C.int(noiseDim), C.int(dHidden),
		C.int(maxoutK), C.float(dropProb), C.float(lr), C.float(momentum), C.int(n))
	if h == nil {
		return nil, errors.New("dm.GAN: allocation failed")
	}
	return &GANModel{InputDim: inputDim, NoiseDim: noiseDim, handle: h}, nil
}

// Free releases the GAN.
func (m *GANModel) Free() {
	if m.handle != nil {
		C.dm_gan_free_raw(m.handle)
		m.handle = nil
	}
}

// Generate produces fake samples from noise vectors.
func (m *GANModel) Generate(zBatch []float32) []float32 {
	batch := len(zBatch) / m.NoiseDim
	out := make([]float32, batch*m.InputDim)
	C.dm_gan_generate_raw(m.handle,
		(*C.float)(unsafe.Pointer(&zBatch[0])), C.int(batch),
		(*C.float)(unsafe.Pointer(&out[0])))
	return out
}

// TrainDiscriminator runs one D step.  Returns D loss.
func (m *GANModel) TrainDiscriminator(realX, zBatch []float32) float32 {
	batch := len(realX) / m.InputDim
	return float32(C.dm_gan_train_d_step_raw(m.handle,
		(*C.float)(unsafe.Pointer(&realX[0])),
		(*C.float)(unsafe.Pointer(&zBatch[0])), C.int(batch)))
}

// TrainGenerator runs one G step.  Returns G loss.
func (m *GANModel) TrainGenerator(zBatch []float32) float32 {
	batch := len(zBatch) / m.NoiseDim
	return float32(C.dm_gan_train_g_step_raw(m.handle,
		(*C.float)(unsafe.Pointer(&zBatch[0])), C.int(batch)))
}

// ═══════════════════════════════════════════════════════════════════════════
// Tokenizer wrappers — named factory structs (HuggingFace-style)
// ═══════════════════════════════════════════════════════════════════════════

// TokenizerBPE is a Byte-Pair Encoding tokenizer.
type TokenizerBPE struct{ Tokenizer }

func NewTokenizerBPE() (*TokenizerBPE, error) {
	t, err := NewTokenizer("bpe")
	if err != nil {
		return nil, err
	}
	return &TokenizerBPE{*t}, nil
}

// TokenizerBPEDropout is BPE with stochastic dropout.
type TokenizerBPEDropout struct{ Tokenizer }

func NewTokenizerBPEDropout() (*TokenizerBPEDropout, error) {
	t, err := NewTokenizer("bpe_dropout")
	if err != nil {
		return nil, err
	}
	return &TokenizerBPEDropout{*t}, nil
}

// TokenizerUnigram is a unigram language model tokenizer.
type TokenizerUnigram struct{ Tokenizer }

func NewTokenizerUnigram() (*TokenizerUnigram, error) {
	t, err := NewTokenizer("unigram")
	if err != nil {
		return nil, err
	}
	return &TokenizerUnigram{*t}, nil
}

// TokenizerSentencePiece is a SentencePiece-lite tokenizer.
type TokenizerSentencePiece struct{ Tokenizer }

func NewTokenizerSentencePiece() (*TokenizerSentencePiece, error) {
	t, err := NewTokenizer("sentencepiece")
	if err != nil {
		return nil, err
	}
	return &TokenizerSentencePiece{*t}, nil
}

// TokenizerWordPiece is a FastWordPiece (BERT-style) tokenizer.
type TokenizerWordPiece struct{ Tokenizer }

func NewTokenizerWordPiece() (*TokenizerWordPiece, error) {
	t, err := NewTokenizer("wordpiece")
	if err != nil {
		return nil, err
	}
	return &TokenizerWordPiece{*t}, nil
}

// TokenizerGPE is a Grapheme Pair Encoding tokenizer.
type TokenizerGPE struct{ Tokenizer }

func NewTokenizerGPE() (*TokenizerGPE, error) {
	t, err := NewTokenizer("gpe")
	if err != nil {
		return nil, err
	}
	return &TokenizerGPE{*t}, nil
}

// TokenizerVolt is a Vocabulary-via-Optimal-Transport tokenizer.
type TokenizerVolt struct{ Tokenizer }

func NewTokenizerVolt() (*TokenizerVolt, error) {
	t, err := NewTokenizer("volt")
	if err != nil {
		return nil, err
	}
	return &TokenizerVolt{*t}, nil
}

// VoltOptimize runs the Volt vocabulary optimisation procedure.
func VoltOptimize(corpusPath string, minSize, maxSize, nSteps int, outputPath string) error {
	cp := C.CString(corpusPath)
	op := C.CString(outputPath)
	defer C.free(unsafe.Pointer(cp))
	defer C.free(unsafe.Pointer(op))
	return statusErr(
		C.dm_tokenizer_volt_run(cp, C.int(minSize), C.int(maxSize),
			C.int(nSteps), op),
		"dm.VoltOptimize")
}
