/**
 * dm — Node.js bindings for libdm (ffi-napi + ref-napi)
 *
 * Install dependencies:
 *   npm install ffi-napi ref-napi ref-array-di ref-struct-di
 *
 * Set DM_LIB to the full path of libdm.so / libdm.dylib / dm.dll, or place
 * the library in the package directory.
 *
 * Usage:
 *   const dm = require('@pomaieco/dm');
 *   console.log(dm.version());
 *
 * SPDX-License-Identifier: MIT
 */

'use strict';

const path   = require('path');
const ffi    = require('ffi-napi');
const ref    = require('ref-napi');
const ArrayType = require('ref-array-di')(ref);
const StructType = require('ref-struct-di')(ref);

// ── Library path resolution ───────────────────────────────────────────────────

function resolveLib() {
  const env = process.env.DM_LIB;
  if (env) return env;

  const names = {
    linux:  'libdm.so',
    darwin: 'libdm.dylib',
    win32:  'dm.dll',
  };
  const name = names[process.platform] || 'libdm.so';

  // Try: package dir → repo root
  const candidates = [
    path.join(__dirname, name),
    path.join(__dirname, '../../..', name),
    name,
  ];
  for (const c of candidates) {
    try { require('fs').accessSync(c); return c; } catch (_) {}
  }
  return name; // let ffi throw a descriptive error
}

const LIB_PATH = resolveLib();

// ── C type aliases ────────────────────────────────────────────────────────────

const void_p  = ref.refType(ref.types.void);
const uint32  = ref.types.uint32;
const uint32p = ref.refType(uint32);
const int_t   = ref.types.int;
const intp    = ref.refType(int_t);
const size_t  = ref.types.size_t;
const float_t = ref.types.float;
const floatp  = ref.refType(float_t);
const double  = ref.types.double;
const bool_t  = ref.types.bool;
const char_p  = ref.types.CString;
const str_p   = 'string';

// DM_BenchReport struct
const DM_BenchReport = StructType({
  phase_times_ms_0:      double,
  phase_times_ms_1:      double,
  phase_times_ms_2:      double,
  phase_times_ms_3:      double,
  peak_memory_kb:        size_t,
  user_cpu_ms:           double,
  sys_cpu_ms:            double,
  result_ram_bytes:      size_t,
  result_disk_est_bytes: size_t,
  num_patterns:          size_t,
  total_items:           size_t,
  throughput_mb_s:       double,
});

// ── FFI bindings ──────────────────────────────────────────────────────────────

const lib = ffi.Library(LIB_PATH, {
  // § 1  Core
  dm_version:        [ str_p,   [] ],
  dm_version_number: [ 'uint32', [] ],
  dm_init:           [ int_t,   [] ],
  dm_strerror:       [ str_p,   [int_t] ],

  // § 2  Dataset
  dm_dataset_open:    [ void_p, [str_p, str_p] ],
  dm_dataset_count:   [ size_t, [void_p] ],
  dm_dataset_max_id:  [ uint32, [void_p] ],
  dm_dataset_free:    [ 'void', [void_p] ],

  // § 3  Algorithm
  dm_algorithm_create: [ void_p, [str_p] ],
  dm_algorithm_run:    [ int_t,  [void_p, str_p, str_p, double, void_p] ],
  dm_algorithm_list:   [ int_t,  [void_p, int_t] ],
  dm_algorithm_free:   [ 'void', [void_p] ],

  // § 4  Tokenizer
  dm_tokenizer_create:     [ void_p, [str_p] ],
  dm_tokenizer_train:      [ int_t,  [void_p, str_p, int_t, str_p] ],
  dm_tokenizer_load:       [ int_t,  [void_p, str_p] ],
  dm_tokenizer_encode:     [ int_t,  [void_p, str_p, uint32p, intp] ],
  dm_tokenizer_decode:     [ int_t,  [void_p, uint32p, int_t, void_p, int_t] ],
  dm_tokenizer_vocab_size: [ int_t,  [void_p] ],
  dm_tokenizer_token_text: [ str_p,  [void_p, uint32, uint32p] ],
  dm_tokenizer_free:       [ 'void', [void_p] ],
  dm_tokenizer_volt_run:   [ int_t,  [str_p, int_t, int_t, int_t, str_p] ],

  // § 5  Vision
  dm_vision_create:      [ void_p, [str_p] ],
  dm_vision_init:        [ int_t,  [void_p, str_p, int_t, int_t, float_t, float_t] ],
  dm_vision_train:       [ int_t,  [void_p, str_p, int_t, int_t, float_t] ],
  dm_vision_eval:        [ int_t,  [void_p, str_p, floatp, floatp] ],
  dm_vision_predict:     [ int_t,  [void_p, floatp, int_t, int_t, floatp, int_t] ],
  dm_vision_save:        [ int_t,  [void_p] ],
  dm_vision_free:        [ 'void', [void_p] ],
  dm_vision_forward_raw: [ int_t,  [floatp, int_t, int_t, int_t, floatp, int_t] ],

  // § 6  LM
  dm_lm_create:   [ void_p, [str_p] ],
  dm_lm_train:    [ int_t,  [void_p, str_p, str_p, int_t, int_t, float_t] ],
  dm_lm_generate: [ int_t,  [void_p, str_p, int_t, void_p, int_t] ],
  dm_lm_load:     [ int_t,  [void_p, str_p] ],
  dm_lm_free:     [ 'void', [void_p] ],

  // § 8  Engine — dm_engine ops (TFE backend, NCHW float32)
  // Tensor lifecycle
  dm_tensor_alloc: [ int_t, [void_p, int_t, int_t, int_t, int_t] ],
  dm_tensor_free:  [ 'void', [void_p] ],
  dm_tensor_fill:  [ 'void', [void_p, float_t] ],
  dm_tensor_get:   [ float_t, [void_p, int_t, int_t, int_t, int_t] ],
  dm_tensor_set:   [ 'void', [void_p, int_t, int_t, int_t, int_t, float_t] ],
  dm_tensor_count: [ size_t, [void_p] ],
  // Convolutions
  dm_op_conv2d_same:    [ int_t, [void_p, void_p, floatp, floatp, int_t, int_t, int_t] ],
  dm_op_depthwise_conv: [ int_t, [void_p, void_p, floatp, floatp, int_t, int_t] ],
  dm_op_pointwise_conv: [ int_t, [void_p, void_p, floatp, floatp, int_t] ],
  // Linear
  dm_op_linear: [ int_t, [void_p, void_p, floatp, floatp, int_t] ],
  // Pooling
  dm_op_global_avg_pool:  [ int_t, [void_p, void_p] ],
  dm_op_max_pool2d_same:  [ int_t, [void_p, void_p, int_t, int_t] ],
  // Normalisation
  dm_op_batch_norm: [ int_t, [void_p, floatp, floatp, floatp, floatp, float_t] ],
  dm_op_layer_norm: [ int_t, [floatp, int_t, int_t, floatp, floatp, float_t] ],
  // Elementwise
  dm_op_tensor_add: [ int_t, [void_p, void_p] ],
  // Activations
  dm_op_relu:    [ 'void', [void_p] ],
  dm_op_relu6:   [ 'void', [void_p] ],
  dm_op_tanh:    [ 'void', [void_p] ],
  dm_op_sigmoid: [ 'void', [void_p] ],
  dm_op_gelu:    [ 'void', [floatp, int_t] ],
  // Softmax
  dm_op_softmax:      [ 'void', [void_p] ],
  dm_op_softmax_rows: [ 'void', [floatp, int_t, int_t] ],
  // Matrix multiplication
  dm_op_matmul_nt: [ 'void', [floatp, floatp, floatp, int_t, int_t, int_t] ],
  dm_op_matmul_nn: [ 'void', [floatp, floatp, floatp, int_t, int_t, int_t] ],
  // Backward passes
  dm_op_linear_backward: [ int_t, [void_p, void_p, void_p, floatp, floatp, floatp, int_t] ],
  dm_op_relu_backward:   [ 'void', [void_p, void_p, void_p] ],
  dm_op_tanh_backward:   [ 'void', [void_p, void_p, void_p] ],
  // Maxout
  dm_op_maxout:          [ int_t, [void_p, void_p, int_t, intp] ],
  dm_op_maxout_backward: [ int_t, [void_p, void_p, int_t, intp] ],
  // Dropout
  dm_op_dropout:          [ 'void', [void_p, void_p, float_t, intp] ],
  dm_op_dropout_backward: [ 'void', [void_p, void_p, float_t, intp] ],
  // Optimisers
  dm_op_adam_step:         [ 'void', [floatp, floatp, floatp, floatp, int_t, float_t, float_t, float_t, float_t, float_t, int_t] ],
  dm_op_adagrad_step:      [ 'void', [floatp, floatp, floatp, int_t, float_t, float_t, float_t] ],
  dm_op_sgd_momentum_step: [ 'void', [floatp, floatp, floatp, int_t, float_t, float_t, float_t, int_t] ],

  // § 9  Benchmark
  dm_bench_reset:      [ 'void', [] ],
  dm_bench_start:      [ 'void', [int_t] ],
  dm_bench_stop:       [ 'void', [int_t] ],
  dm_bench_record:     [ 'void', [size_t, size_t] ],
  dm_bench_get_report: [ DM_BenchReport, [] ],
  dm_bench_print:      [ 'void', [str_p, str_p] ],

  // § 10  BitSet
  dm_bitset_create:   [ void_p, [size_t] ],
  dm_bitset_copy:     [ void_p, [void_p] ],
  dm_bitset_free:     [ 'void', [void_p] ],
  dm_bitset_set:      [ 'void', [void_p, size_t] ],
  dm_bitset_clear:    [ 'void', [void_p, size_t] ],
  dm_bitset_get:      [ bool_t, [void_p, size_t] ],
  dm_bitset_and:      [ 'void', [void_p, void_p] ],
  dm_bitset_or:       [ 'void', [void_p, void_p] ],
  dm_bitset_not:      [ 'void', [void_p] ],
  dm_bitset_set_all:  [ 'void', [void_p] ],
  dm_bitset_popcount: [ size_t, [void_p] ],

  // § 11  DataGen
  dm_datagen_create: [ void_p, [str_p] ],
  dm_datagen_run:    [ int_t,  [void_p, str_p, str_p, 'uint'] ],
  dm_datagen_free:   [ 'void', [void_p] ],

  // § 13  CLI
  dm_cli_run: [ int_t, [str_p, int_t, void_p] ],

  // § 14  GPU
  dm_gpu_create:      [ void_p, [int_t, str_p] ],
  dm_gpu_free:        [ 'void', [void_p] ],
  dm_gpu_ready:       [ int_t,  [void_p] ],
  dm_gpu_device_name: [ int_t,  [void_p, void_p, size_t] ],

  // § 18  Experiment
  dm_timer_now:                  [ double,  [] ],
  dm_peak_ram_mb:                [ double,  [] ],
  dm_file_size_mb:               [ double,  [str_p] ],
  dm_dir_size_mb:                [ double,  [str_p] ],
  dm_ensure_dir:                 [ int_t,   [str_p] ],
  dm_path_basename:              [ str_p,   [str_p] ],
  dm_experiment_write_header:    [ 'void',  [str_p] ],
  dm_experiment_generate_report: [ 'void',  [str_p] ],
});

// ── Helpers ───────────────────────────────────────────────────────────────────

const DM_OK = 0;

function check(status, ctx = '') {
  if (status !== DM_OK) {
    const msg = lib.dm_strerror(status);
    throw new Error(`${ctx}: ${msg} (code ${status})`);
  }
}

function nullTermArray(strings) {
  /** Build a Buffer of NULL-terminated char* pointers. */
  const ptrs = strings.map(s => ref.allocCString(s));
  ptrs.push(ref.NULL);
  const arr = Buffer.alloc(ptrs.length * ref.sizeof.pointer);
  ptrs.forEach((p, i) => ref.writePointer(arr, i * ref.sizeof.pointer, p));
  return arr;
}

// ── § 1  Core ─────────────────────────────────────────────────────────────────

/** @returns {string} libdm version string */
function version() { return lib.dm_version(); }

/** @returns {{ major: number, minor: number, patch: number }} */
function versionNumber() {
  const v = lib.dm_version_number();
  return { major: (v >>> 16) & 0xFF, minor: (v >>> 8) & 0xFF, patch: v & 0xFF };
}

/** Initialise libdm (idempotent). */
function init() { check(lib.dm_init(), 'dm.init'); }

/** @returns {string} human-readable status message */
function strerror(code) { return lib.dm_strerror(code); }

// ── § 2  Dataset ──────────────────────────────────────────────────────────────

class Dataset {
  /**
   * @param {string} filePath
   * @param {string} type  "transactional" | "utility" | "sequence" | "quantity" | "matrix"
   */
  constructor(filePath, type) {
    this._h = lib.dm_dataset_open(filePath, type);
    if (ref.isNull(this._h))
      throw new Error(`dm.Dataset: failed to open '${filePath}' as '${type}'`);
  }
  close()  { lib.dm_dataset_free(this._h); this._h = null; }
  get count()  { return lib.dm_dataset_count(this._h); }
  get maxId()  { return lib.dm_dataset_max_id(this._h); }
}

// ── § 3  Algorithm ────────────────────────────────────────────────────────────

class Algorithm {
  /** @param {string} id  algorithm ID (see dm.h § 3 for all 132 IDs) */
  constructor(id) {
    this._h = lib.dm_algorithm_create(id);
    if (ref.isNull(this._h))
      throw new Error(`dm.Algorithm: unknown id '${id}'`);
  }
  close() { lib.dm_algorithm_free(this._h); this._h = null; }

  /**
   * @param {string}   datasetPath
   * @param {string}   outputPath
   * @param {number}   minSupport
   * @param {string[]} [extraArgs=[]]
   */
  run(datasetPath, outputPath, minSupport, extraArgs = []) {
    const arr = nullTermArray(extraArgs);
    check(lib.dm_algorithm_run(this._h, datasetPath, outputPath,
                               minSupport, arr), 'dm.Algorithm.run');
  }

  /** @returns {string[]} all registered algorithm IDs */
  static listAll() {
    const buf = Buffer.alloc(32768);
    check(lib.dm_algorithm_list(buf, buf.length), 'dm.Algorithm.listAll');
    return ref.readCString(buf, 0).split('\n').filter(Boolean);
  }
}

// ── § 4  Tokenizer ────────────────────────────────────────────────────────────

class Tokenizer {
  /**
   * @param {string} type
   *   "bpe" | "bpe_dropout" | "unigram" | "sentencepiece" | "wordpiece" |
   *   "gpe" | "parity_bpe" | "volt" | "maximal_munch" | "faro" | "tokenizer_lab"
   */
  constructor(type) {
    this._h = lib.dm_tokenizer_create(type);
    if (ref.isNull(this._h))
      throw new Error(`dm.Tokenizer: unknown type '${type}'`);
  }
  close() { lib.dm_tokenizer_free(this._h); this._h = null; }

  train(corpusPath, vocabSize, outputPath) {
    check(lib.dm_tokenizer_train(this._h, corpusPath, vocabSize, outputPath),
          'dm.Tokenizer.train');
  }

  load(modelPath) {
    check(lib.dm_tokenizer_load(this._h, modelPath), 'dm.Tokenizer.load');
  }

  /** @param {string} text @returns {Uint32Array} */
  encode(text) {
    let capacity = 8192;
    let ids = Buffer.alloc(capacity * 4);
    const lenRef = ref.alloc(int_t, capacity);
    let s = lib.dm_tokenizer_encode(this._h, text, ids, lenRef);
    const needed = lenRef.deref();
    if (s === -3 /* DM_ERR_MEMORY */ || needed > capacity) {
      capacity = needed;
      ids = Buffer.alloc(capacity * 4);
      lenRef.writeInt32LE(capacity, 0);
      s = lib.dm_tokenizer_encode(this._h, text, ids, lenRef);
    }
    check(s, 'dm.Tokenizer.encode');
    const count = lenRef.deref();
    const result = new Uint32Array(count);
    for (let i = 0; i < count; i++) result[i] = ids.readUInt32LE(i * 4);
    return result;
  }

  /** @param {Uint32Array|number[]} ids @returns {string} */
  decode(ids) {
    const idsArr = Buffer.alloc(ids.length * 4);
    for (let i = 0; i < ids.length; i++) idsArr.writeUInt32LE(ids[i], i * 4);
    const outBuf = Buffer.alloc(ids.length * 8 + 64);
    check(lib.dm_tokenizer_decode(this._h, idsArr, ids.length,
                                  outBuf, outBuf.length), 'dm.Tokenizer.decode');
    return ref.readCString(outBuf, 0);
  }

  get vocabSize() { return lib.dm_tokenizer_vocab_size(this._h); }

  tokenText(id) {
    const lenRef = ref.alloc(uint32, 0);
    return lib.dm_tokenizer_token_text(this._h, id, lenRef) || '';
  }

  static voltRun(corpusPath, minSize, maxSize, nSteps, outputPath) {
    check(lib.dm_tokenizer_volt_run(corpusPath, minSize, maxSize, nSteps, outputPath),
          'dm.Tokenizer.voltRun');
  }
}

// ── § 5  Vision ───────────────────────────────────────────────────────────────

class Vision {
  /** @param {string} [modelType='mobilenet_tiny'] */
  constructor(modelType = 'mobilenet_tiny') {
    this._h = lib.dm_vision_create(modelType);
    if (ref.isNull(this._h)) throw new Error('dm.Vision: create failed');
  }
  close() { lib.dm_vision_free(this._h); this._h = null; }

  initModel(savedModelDir, classes, imageSize, widthMult = 1.0, lr = 0.001) {
    check(lib.dm_vision_init(this._h, savedModelDir, classes,
                             imageSize, widthMult, lr), 'dm.Vision.initModel');
  }
  train(manifestPath, epochs, batchSize, lr = 0.001) {
    check(lib.dm_vision_train(this._h, manifestPath, epochs, batchSize, lr),
          'dm.Vision.train');
  }
  /** @returns {{ loss: number, accuracy: number }} */
  eval(manifestPath) {
    const lossRef = ref.alloc(float_t, 0);
    const accRef  = ref.alloc(float_t, 0);
    check(lib.dm_vision_eval(this._h, manifestPath, lossRef, accRef),
          'dm.Vision.eval');
    return { loss: lossRef.deref(), accuracy: accRef.deref() };
  }
  /** @param {Float32Array|number[]} rgb @returns {Float32Array} */
  predict(rgb, h, w, nClasses) {
    const inp = Buffer.alloc(rgb.length * 4);
    for (let i = 0; i < rgb.length; i++) inp.writeFloatLE(rgb[i], i * 4);
    const out = Buffer.alloc(nClasses * 4);
    check(lib.dm_vision_predict(this._h, inp, h, w, out, nClasses),
          'dm.Vision.predict');
    const result = new Float32Array(nClasses);
    for (let i = 0; i < nClasses; i++) result[i] = out.readFloatLE(i * 4);
    return result;
  }
  save() { check(lib.dm_vision_save(this._h), 'dm.Vision.save'); }
}

// ── § 6  LM ───────────────────────────────────────────────────────────────────

class LM {
  /** @param {string} modelType "bert" | "tiny_transformer" | "tinystories" */
  constructor(modelType) {
    this._h = lib.dm_lm_create(modelType);
    if (ref.isNull(this._h))
      throw new Error(`dm.LM: create failed for type '${modelType}'`);
  }
  close() { lib.dm_lm_free(this._h); this._h = null; }

  train(corpusPath, checkpointDir, epochs, batchSize, lr = 0.001) {
    check(lib.dm_lm_train(this._h, corpusPath, checkpointDir,
                          epochs, batchSize, lr), 'dm.LM.train');
  }
  load(checkpointDir) {
    check(lib.dm_lm_load(this._h, checkpointDir), 'dm.LM.load');
  }
  /** @returns {string} */
  generate(prompt, maxTokens = 256) {
    // For BERT, prompt is space-separated token IDs and output is pooled [CLS] values.
    const buf = Buffer.alloc(maxTokens * 4 + 256);
    check(lib.dm_lm_generate(this._h, prompt, maxTokens, buf, buf.length),
          'dm.LM.generate');
    return ref.readCString(buf, 0);
  }
}

// ── § 9  Benchmark ────────────────────────────────────────────────────────────

const BENCH_LOAD  = 0;
const BENCH_ALGO  = 1;
const BENCH_WRITE = 2;
const BENCH_TOTAL = 3;

function benchReset()           { lib.dm_bench_reset(); }
function benchStart(phase)      { lib.dm_bench_start(phase); }
function benchStop(phase)       { lib.dm_bench_stop(phase); }
function benchRecord(n, t)      { lib.dm_bench_record(n, t); }
function benchPrint(algo, ds)   { lib.dm_bench_print(algo, ds); }
function getBenchReport() {
  const r = lib.dm_bench_get_report();
  return {
    phaseMs: [r.phase_times_ms_0, r.phase_times_ms_1,
              r.phase_times_ms_2, r.phase_times_ms_3],
    peakKB:          r.peak_memory_kb,
    userMs:          r.user_cpu_ms,
    sysMs:           r.sys_cpu_ms,
    resultRam:       r.result_ram_bytes,
    resultDisk:      r.result_disk_est_bytes,
    numPatterns:     r.num_patterns,
    totalItems:      r.total_items,
    throughputMbS:   r.throughput_mb_s,
  };
}

// ── § 10  BitSet ──────────────────────────────────────────────────────────────

class BitSet {
  /** @param {number} nBits */
  constructor(nBits) {
    this._h = lib.dm_bitset_create(nBits);
    if (ref.isNull(this._h)) throw new Error('dm.BitSet: allocation failed');
  }
  copy() {
    const h = lib.dm_bitset_copy(this._h);
    if (ref.isNull(h)) throw new Error('dm.BitSet.copy: allocation failed');
    const b = Object.create(BitSet.prototype);
    b._h = h;
    return b;
  }
  close()           { lib.dm_bitset_free(this._h); this._h = null; }
  set(pos)          { lib.dm_bitset_set(this._h, pos); }
  clear(pos)        { lib.dm_bitset_clear(this._h, pos); }
  get(pos)          { return lib.dm_bitset_get(this._h, pos); }
  setAll()          { lib.dm_bitset_set_all(this._h); }
  flip()            { lib.dm_bitset_not(this._h); }
  popcount()        { return lib.dm_bitset_popcount(this._h); }
  and(other)        { lib.dm_bitset_and(this._h, other._h); return this; }
  or(other)         { lib.dm_bitset_or(this._h, other._h); return this; }
}

// ── § 11  DataGen ─────────────────────────────────────────────────────────────

class DataGen {
  /** @param {string} type "medm" | "textbook" */
  constructor(type) {
    this._h = lib.dm_datagen_create(type);
    if (ref.isNull(this._h))
      throw new Error(`dm.DataGen: unknown type '${type}'`);
  }
  close() { lib.dm_datagen_free(this._h); this._h = null; }
  run(spec, outputPath, seed = 0) {
    check(lib.dm_datagen_run(this._h, spec, outputPath, seed),
          'dm.DataGen.run');
  }
}

// ── § 14  GPU ─────────────────────────────────────────────────────────────────

class GpuCtx {
  /**
   * @param {number} [deviceIndex=0]
   * @param {string|null} [shaderDir=null]
   */
  constructor(deviceIndex = 0, shaderDir = null) {
    this._h = lib.dm_gpu_create(deviceIndex, shaderDir);
  }
  close() {
    if (this._h && !ref.isNull(this._h)) {
      lib.dm_gpu_free(this._h); this._h = null;
    }
  }
  get ready() { return this._h && !ref.isNull(this._h) && lib.dm_gpu_ready(this._h) !== 0; }
  get deviceName() {
    if (!this._h || ref.isNull(this._h)) return '(no GPU)';
    const buf = Buffer.alloc(256);
    lib.dm_gpu_device_name(this._h, buf, buf.length);
    return ref.readCString(buf, 0);
  }
}

// ── § 13  CLI ─────────────────────────────────────────────────────────────────

function cliRun(command, ...args) {
  const arr = nullTermArray(args);
  return lib.dm_cli_run(command, args.length, arr);
}

// ── § 18  Experiment ─────────────────────────────────────────────────────────

function timerNow()                     { return lib.dm_timer_now(); }
function peakRamMb()                    { return lib.dm_peak_ram_mb(); }
function fileSizeMb(p)                  { return lib.dm_file_size_mb(p); }
function dirSizeMb(p)                   { return lib.dm_dir_size_mb(p); }
function ensureDir(p) {
  if (lib.dm_ensure_dir(p) !== 0)
    throw new Error(`dm.ensureDir: failed for '${p}'`);
}
function pathBasename(p)                { return lib.dm_path_basename(p); }
function experimentWriteHeader(csv)     { lib.dm_experiment_write_header(csv); }
function experimentGenerateReport(root) { lib.dm_experiment_generate_report(root); }

// ── Exports ───────────────────────────────────────────────────────────────────

module.exports = {
  // constants
  DM_OK, DM_LIB_PATH: LIB_PATH,
  BENCH_LOAD, BENCH_ALGO, BENCH_WRITE, BENCH_TOTAL,

  // § 1
  version, versionNumber, init, strerror,

  // § 2
  Dataset,

  // § 3
  Algorithm,

  // § 4
  Tokenizer,

  // § 5
  Vision,

  // § 6
  LM,

  // § 8  Engine
  Tensor,
  op,

  // § 9
  benchReset, benchStart, benchStop, benchRecord, getBenchReport, benchPrint,

  // § 10
  BitSet,

  // § 11
  DataGen,

  // § 13
  cliRun,

  // § 14
  GpuCtx,

  // § 18
  timerNow, peakRamMb, fileSizeMb, dirSizeMb, ensureDir, pathBasename,
  experimentWriteHeader, experimentGenerateReport,
};

// ── § 8  Engine — Tensor class and op namespace ───────────────────────────────
//
// Build custom models by composing Tensor and op.* primitives, just like
// TensorFlow layers — ops dispatch through TFE so XLA/cuDNN/oneDNN are used.
//
// Layout: NCHW (n, c, h, w), row-major, contiguous float32.
//
// Example:
//   const x = new dm.Tensor(1, 3, 224, 224);
//   const y = new dm.Tensor(1, 64, 112, 112);
//   dm.op.conv2dSame(x, y, weights, bias, 64, 3, 2);
//   dm.op.relu(y);
//   x.free(); y.free();

// DM_Tensor struct layout: int n, c, h, w + float* data
const DM_TENSOR_SIZE = 4 * 4 + ref.sizeof.pointer; // 4 ints + 1 pointer

function _tensorPtr(buf) { return buf; }

/** Allocate a DM_Tensor struct in a Buffer and call dm_tensor_alloc. */
class Tensor {
  constructor(n, c, h, w) {
    // Allocate struct: 4 ints (n,c,h,w) + pointer (data)
    this._buf = Buffer.alloc(DM_TENSOR_SIZE);
    const s = lib.dm_tensor_alloc(this._buf, n, c, h, w);
    check(s, 'dm.Tensor');
    this._freed = false;
  }

  free() {
    if (!this._freed) { lib.dm_tensor_free(this._buf); this._freed = true; }
  }

  get n() { return this._buf.readInt32LE(0); }
  get c() { return this._buf.readInt32LE(4); }
  get h() { return this._buf.readInt32LE(8); }
  get w() { return this._buf.readInt32LE(12); }
  get count() { return lib.dm_tensor_count(this._buf); }

  fill(v)              { lib.dm_tensor_fill(this._buf, v); }
  get_(n, c, y, x)     { return lib.dm_tensor_get(this._buf, n, c, y, x); }
  set_(n, c, y, x, v)  { lib.dm_tensor_set(this._buf, n, c, y, x, v); }
}

/** Convert a JS number[] to a Float32Array Buffer. */
function _floatBuf(arr) {
  const buf = Buffer.alloc(arr.length * 4);
  for (let i = 0; i < arr.length; i++) buf.writeFloatLE(arr[i], i * 4);
  return buf;
}

/** Convert a JS number[] (ints) to an Int32Array Buffer. */
function _intBuf(arr) {
  const buf = Buffer.alloc(arr.length * 4);
  for (let i = 0; i < arr.length; i++) buf.writeInt32LE(arr[i], i * 4);
  return buf;
}

/** Read n floats from a Buffer back into a plain array. */
function _readFloats(buf, n) {
  const out = [];
  for (let i = 0; i < n; i++) out.push(buf.readFloatLE(i * 4));
  return out;
}

/** Read n ints from a Buffer back into a plain array. */
function _readInts(buf, n) {
  const out = [];
  for (let i = 0; i < n; i++) out.push(buf.readInt32LE(i * 4));
  return out;
}

const op = {
  // ── Convolutions ────────────────────────────────────────────────────────────
  conv2dSame(in_, out, w, b, outC, kernel, stride) {
    check(lib.dm_op_conv2d_same(in_._buf, out._buf, _floatBuf(w), _floatBuf(b),
                                outC, kernel, stride), 'dm.op.conv2dSame');
  },
  depthwiseConv(in_, out, w, b, kernel, stride) {
    check(lib.dm_op_depthwise_conv(in_._buf, out._buf, _floatBuf(w), _floatBuf(b),
                                   kernel, stride), 'dm.op.depthwiseConv');
  },
  pointwiseConv(in_, out, w, b, outC) {
    check(lib.dm_op_pointwise_conv(in_._buf, out._buf, _floatBuf(w), _floatBuf(b),
                                   outC), 'dm.op.pointwiseConv');
  },

  // ── Linear ──────────────────────────────────────────────────────────────────
  linear(in_, out, w, b, outC) {
    check(lib.dm_op_linear(in_._buf, out._buf, _floatBuf(w), _floatBuf(b),
                           outC), 'dm.op.linear');
  },

  // ── Pooling ──────────────────────────────────────────────────────────────────
  globalAvgPool(in_, out) {
    check(lib.dm_op_global_avg_pool(in_._buf, out._buf), 'dm.op.globalAvgPool');
  },
  maxPool2dSame(in_, out, kernel, stride) {
    check(lib.dm_op_max_pool2d_same(in_._buf, out._buf, kernel, stride),
          'dm.op.maxPool2dSame');
  },

  // ── Normalisation ────────────────────────────────────────────────────────────
  batchNorm(t, gamma, beta, mean, variance, eps = 1e-5) {
    check(lib.dm_op_batch_norm(t._buf, _floatBuf(gamma), _floatBuf(beta),
                               _floatBuf(mean), _floatBuf(variance), eps),
          'dm.op.batchNorm');
  },
  /** x: Float32 array of length seqLen×dModel, mutated in-place. Returns updated x. */
  layerNorm(x, seqLen, dModel, gamma, beta, eps = 1e-5) {
    const xBuf = _floatBuf(x);
    check(lib.dm_op_layer_norm(xBuf, seqLen, dModel,
                               _floatBuf(gamma), _floatBuf(beta), eps),
          'dm.op.layerNorm');
    return _readFloats(xBuf, x.length);
  },

  // ── Elementwise ───────────────────────────────────────────────────────────────
  add(out, in_) {
    check(lib.dm_op_tensor_add(out._buf, in_._buf), 'dm.op.add');
  },

  // ── Activations ───────────────────────────────────────────────────────────────
  relu   (t) { lib.dm_op_relu(t._buf);    },
  relu6  (t) { lib.dm_op_relu6(t._buf);   },
  tanh   (t) { lib.dm_op_tanh(t._buf);    },
  sigmoid(t) { lib.dm_op_sigmoid(t._buf); },
  /** Applies GELU in-place to a JS number[]. Returns updated array. */
  gelu(x) {
    const xBuf = _floatBuf(x);
    lib.dm_op_gelu(xBuf, x.length);
    return _readFloats(xBuf, x.length);
  },

  // ── Softmax ───────────────────────────────────────────────────────────────────
  softmax(t)               { lib.dm_op_softmax(t._buf); },
  /** Applies softmax row-wise to a flat [rows × cols] array. Returns updated array. */
  softmaxRows(x, rows, cols) {
    const xBuf = _floatBuf(x);
    lib.dm_op_softmax_rows(xBuf, rows, cols);
    return _readFloats(xBuf, x.length);
  },

  // ── Matrix multiplication ──────────────────────────────────────────────────────
  /** C = A × Bᵀ  (A[M×K], B[N×K] → C[M×N]) */
  matmulNT(A, B, M, N, K) {
    const C = Buffer.alloc(M * N * 4);
    lib.dm_op_matmul_nt(_floatBuf(A), _floatBuf(B), C, M, N, K);
    return _readFloats(C, M * N);
  },
  /** C = A × B  (A[M×K], B[K×N] → C[M×N]) */
  matmulNN(A, B, M, K, N) {
    const C = Buffer.alloc(M * N * 4);
    lib.dm_op_matmul_nn(_floatBuf(A), _floatBuf(B), C, M, K, N);
    return _readFloats(C, M * N);
  },

  // ── Backward passes ───────────────────────────────────────────────────────────
  linearBackward(in_, gradOut, gradIn, gradW, gradB, w, outC) {
    const gw = _floatBuf(gradW), gb = _floatBuf(gradB);
    check(lib.dm_op_linear_backward(in_._buf, gradOut._buf, gradIn._buf,
                                    gw, gb, _floatBuf(w), outC),
          'dm.op.linearBackward');
    return { gradW: _readFloats(gw, gradW.length), gradB: _readFloats(gb, gradB.length) };
  },
  reluBackward(in_, gradOut, gradIn)       { lib.dm_op_relu_backward(in_._buf, gradOut._buf, gradIn._buf); },
  tanhBackward(out_, gradOut, gradIn)      { lib.dm_op_tanh_backward(out_._buf, gradOut._buf, gradIn._buf); },

  // ── Maxout ────────────────────────────────────────────────────────────────────
  maxout(in_, out, k) {
    const argmax = Buffer.alloc(in_.n * (in_.c / k) * 4);
    check(lib.dm_op_maxout(in_._buf, out._buf, k, argmax), 'dm.op.maxout');
    return _readInts(argmax, in_.n * (in_.c / k));
  },
  maxoutBackward(gradOut, gradIn, k, argmax) {
    check(lib.dm_op_maxout_backward(gradOut._buf, gradIn._buf, k, _intBuf(argmax)),
          'dm.op.maxoutBackward');
  },

  // ── Dropout ───────────────────────────────────────────────────────────────────
  dropout(in_, out, dropProb) {
    const mask = Buffer.alloc(in_.count * 4);
    lib.dm_op_dropout(in_._buf, out._buf, dropProb, mask);
    return _readInts(mask, in_.count);
  },
  dropoutBackward(gradOut, gradIn, dropProb, mask) {
    lib.dm_op_dropout_backward(gradOut._buf, gradIn._buf, dropProb, _intBuf(mask));
  },

  // ── Optimisers ────────────────────────────────────────────────────────────────
  adamStep(param, grad, m, v, { lr = 1e-3, beta1 = 0.9, beta2 = 0.999,
                                 eps = 1e-8, weightDecay = 0, t = 1 } = {}) {
    const pBuf = _floatBuf(param), gBuf = _floatBuf(grad);
    const mBuf = _floatBuf(m), vBuf = _floatBuf(v);
    lib.dm_op_adam_step(pBuf, gBuf, mBuf, vBuf, param.length,
                        lr, beta1, beta2, eps, weightDecay, t);
    return { param: _readFloats(pBuf, param.length),
             m:     _readFloats(mBuf, m.length),
             v:     _readFloats(vBuf, v.length) };
  },
  adagradStep(param, grad, gSum, { lr = 1e-2, eps = 1e-8, weightDecay = 0 } = {}) {
    const pBuf = _floatBuf(param), gBuf = _floatBuf(grad), gsBuf = _floatBuf(gSum);
    lib.dm_op_adagrad_step(pBuf, gBuf, gsBuf, param.length, lr, eps, weightDecay);
    return { param: _readFloats(pBuf, param.length),
             gSum:  _readFloats(gsBuf, gSum.length) };
  },
  sgdMomentumStep(param, grad, velocity,
                  { lr = 1e-2, momentum = 0.9, weightDecay = 0, nesterov = false } = {}) {
    const pBuf = _floatBuf(param), gBuf = _floatBuf(grad), vBuf = _floatBuf(velocity);
    lib.dm_op_sgd_momentum_step(pBuf, gBuf, vBuf, param.length,
                                lr, momentum, weightDecay, nesterov ? 1 : 0);
    return { param:    _readFloats(pBuf, param.length),
             velocity: _readFloats(vBuf, velocity.length) };
  },
};
