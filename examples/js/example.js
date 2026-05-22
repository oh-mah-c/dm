#!/usr/bin/env node
/**
 * examples/js/example.js — Node.js usage example for dm
 *
 * Install:
 *   npm install ffi-napi ref-napi ref-array-di ref-struct-di
 *   export DM_LIB=/path/to/libdm.so
 *
 * Run:
 *   node example.js
 */

'use strict';

const path = require('path');

// Resolve binding
process.env.DM_LIB = process.env.DM_LIB ||
    path.join(__dirname, '../../libdm.so');

const dm = require(path.join(__dirname, '../../bindings/js/dm'));

const DATASETS = path.join(__dirname, '../../datasets');

function section(title) {
  console.log('\n' + '─'.repeat(60));
  console.log('  ' + title);
  console.log('─'.repeat(60));
}

async function main() {
  // ── § 1  Core ─────────────────────────────────────────────────────────────
  dm.init();
  const v = dm.versionNumber();
  console.log(`libdm ${dm.version()}  (${v.major}.${v.minor}.${v.patch})`);

  // ── Algorithm: FP-Growth ──────────────────────────────────────────────────
  section('Algorithm — FP-Growth');
  {
    const algo = new dm.Algorithm('fpgrowth');
    try {
      algo.run(
        `${DATASETS}/itemsets/mushrooms.txt`,
        '/tmp/fpgrowth_js_out.txt',
        0.05,
        ['min_length=2']
      );
      console.log('FP-Growth complete → /tmp/fpgrowth_js_out.txt');
    } catch (e) {
      console.error('FP-Growth error:', e.message);
    }
    algo.close();
  }

  // ── Algorithm: EFIM ───────────────────────────────────────────────────────
  section('Algorithm — EFIM (high-utility)');
  {
    const efim = new dm.Algorithm('efim');
    try {
      efim.run(
        `${DATASETS}/utilities/foodmart.txt`,
        '/tmp/efim_js_out.txt',
        50.0
      );
      console.log('EFIM complete → /tmp/efim_js_out.txt');
    } catch (e) {
      console.error('EFIM error:', e.message);
    }
    efim.close();
  }

  // ── Algorithm list ────────────────────────────────────────────────────────
  section('Algorithm registry (first 5)');
  const ids = dm.Algorithm.listAll();
  ids.slice(0, 5).forEach(id => console.log(' ', id));
  console.log(`  … (${ids.length} total)`);

  // ── Tokenizer ─────────────────────────────────────────────────────────────
  section('Tokenizer — BPE train + encode/decode');
  {
    const tok = new dm.Tokenizer('bpe');
    try {
      tok.train(
        `${DATASETS}/tokenizer/real_corpus.txt`,
        2000,
        '/tmp/bpe_js_model'
      );
      console.log(`Trained BPE, vocab_size=${tok.vocabSize}`);

      const text = 'data mining and machine learning';
      const ids2 = tok.encode(text);
      console.log(`Encoded '${text}' → ${ids2.length} tokens`);

      const decoded = tok.decode(ids2);
      console.log(`Decoded: '${decoded}'`);

      // Show first 3 token texts
      for (let i = 0; i < Math.min(3, ids2.length); i++) {
        console.log(`  token[${ids2[i]}] = '${tok.tokenText(ids2[i])}'`);
      }
    } catch (e) {
      console.error('Tokenizer error:', e.message);
    }
    tok.close();
  }

  // ── BitSet ────────────────────────────────────────────────────────────────
  section('BitSet operations');
  {
    const a = new dm.BitSet(128);
    const b = new dm.BitSet(128);
    [1, 7, 42, 100].forEach(p => a.set(p));
    [7, 42, 99].forEach(p => b.set(p));
    a.and(b); // {7, 42}
    console.log(`popcount after AND = ${a.popcount()} (expected 2)`);
    a.close();
    b.close();
  }

  // ── DataGen ───────────────────────────────────────────────────────────────
  section('DataGen — MEDM synthetic');
  {
    const gen = new dm.DataGen('medm');
    try {
      gen.run('nItems=100,nTxn=1000,avgLen=10', '/tmp/syn_js.txt', 42);
      console.log(`Synthetic dataset → /tmp/syn_js.txt ` +
                  `(${dm.fileSizeMb('/tmp/syn_js.txt').toFixed(2)} MB)`);
    } catch (e) {
      console.error('DataGen error:', e.message);
    }
    gen.close();
  }

  // ── GPU context ───────────────────────────────────────────────────────────
  section('GPU context');
  {
    const gpu = new dm.GpuCtx(0, null);
    console.log(`GPU ready: ${gpu.ready}`);
    console.log(`GPU device: ${gpu.deviceName}`);
    gpu.close();
  }

  // ── Benchmark ─────────────────────────────────────────────────────────────
  section('Benchmark — Apriori on T10I4D100K');
  {
    dm.benchReset();
    dm.benchStart(dm.BENCH_TOTAL);

    const algo = new dm.Algorithm('apriori');
    try {
      algo.run(
        `${DATASETS}/itemsets/T10I4D100K.txt`,
        '/tmp/apriori_js.txt',
        0.1
      );
    } catch (e) {
      console.error('Apriori error:', e.message);
    }
    algo.close();

    dm.benchStop(dm.BENCH_TOTAL);
    dm.benchPrint('apriori', 'T10I4D100K');

    const r = dm.getBenchReport();
    console.log(`total=${r.phaseMs[3].toFixed(1)}ms  patterns=${r.numPatterns}  ` +
                `throughput=${r.throughputMbS.toFixed(2)}MB/s`);
  }

  // ── Experiment helpers ────────────────────────────────────────────────────
  section('Experiment helpers');
  console.log(`timer_now   = ${dm.timerNow().toFixed(3)} s`);
  console.log(`peak_ram_mb = ${dm.peakRamMb().toFixed(1)} MB`);

  // ── CLI passthrough ───────────────────────────────────────────────────────
  section('CLI passthrough — dm version');
  const code = dm.cliRun('version');
  console.log(`exit code: ${code}`);
}

main().catch(err => {
  console.error('Unhandled error:', err);
  process.exit(1);
});
