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
