# Algorithm Interface Audit

This audit checks whether existing algorithms are already aligned with the next-generation plug-and-play direction.

## Current State

- Algorithm C files scanned: 136
- Files with old `DM_Algorithm` registry signs: 126
- Files not registered through the old registry: 10
- Algorithm files with direct file I/O: 15

## What This Means

Most classic algorithms already use the older `DM_Algorithm` interface:

```c
DM_Status (*run)(DM_Dataset *ds, void *params);
```

That is useful, but it is not yet the new universal connector contract. The old path still depends on `DM_Dataset` types such as transactional, utility, sequence utility, or quantity. Standalone tools and several newer research prototypes bypass the registry and call direct functions such as `*_mine_file`, `*_mine_path`, or `*_mine_dataset`.

The new target interface is:

```c
int (*run)(const DM_PluginInput *input, DM_PluginResult *result);
```

where `input->flat` is a `DM_FlatDataset` produced by `spmf`, `text`, or `graph` connectors.

## Unified Runner Path

Added:

- `include/core/dm_plugin.h`
- `src/core/dm_plugin.c`
- `src/core/dm_builtin_plugins.c`
- `include/core/dm_flat_adapter.h`
- `src/core/dm_flat_adapter.c`
- `src/core/dm_algorithm_adapters.c`
- `src/tools/dm_run.c`

The runner now has three unified paths behind one CLI:

- Native flat plugins consume `DM_FlatDataset` directly.
- Legacy registry algorithms consume `DM_Dataset` produced from `DM_FlatDataset` by family adapters.
- Standalone research miners are exposed as plugins; dataset-based ones use the flat adapters, while raw-parser papers keep their exact parser to avoid semantic loss.

Flat/native and adapter plugins include:

- `flat_stats`: reports statistics for any connector-backed flat dataset.
- `topk_items`: mines top-k item counts from any connector-backed flat dataset.
- `chuo`: adapts flat rows to utility transactions and runs CHUO.
- `tku_miner`: adapts flat rows to utility transactions and runs TKU.
- `tku_pso`: adapts flat rows to utility transactions and runs TKU-PSO.
- `huciminer`: exposed through `dm_run` with the exact utility parser because HUCI requires per-item utility values that are not present in `DM_FlatDataset`.
- `htk_miner`, `hupp`, `kclotree_miner`, `tipn_houi`, `topkphm`: exposed through `dm_run` with their exact raw-format parsers.
- `pso_classifier`: exposed through `dm_run` as a raw-only folder adapter because its input is a class folder, not a single flat itemset file.

Example:

```bash
bin/dm_run --algorithm topk_items --input datasets/itemsets/mushrooms.txt --connector spmf --k 5
bin/dm_run --algorithm topk_items --input README.md --connector text --window 32 --stride 16 --k 5
bin/dm_run --algorithm flat_stats --input graph_edges.txt --connector graph --undirected
bin/dm_run --algorithm fpgrowth --input datasets/synthetic/syn_sparse.txt --connector spmf
bin/dm_run --algorithm tku_miner --input datasets/synthetic/syn_sparse.txt --connector spmf --k 5
bin/dm_run --algorithm huciminer --input datasets/utilities/foodmart.txt --minutil 5000 --minconf 0.8
```

## Files Not Yet Registered In The Old Registry

- `chuo_miner.c`
- `htk_miner.c`
- `hupp.c`
- `kclotree_miner.c`
- `pso_classifier.c`
- `tipn_houi.c`
- `tku_miner.c`
- `tku_pso.c`
- `topkphm.c`

Most of these are now reachable from `dm_run` through `src/core/dm_algorithm_adapters.c`. HUCI-Miner is now registered as `huciminer` and also has a raw utility plugin adapter for parameterized `dm_run` use.

## Algorithm Files With Direct File I/O

- `clh_miner.c`
- `feacp.c`
- `fhuqi_miner.c`
- `hiep.c`
- `htk_miner.c`
- `hupp.c`
- `kclotree_miner.c`
- `mfhoi_common.c`
- `mheinu.c`
- `mhoui.c`
- `mlhui_miner.c`
- `pso_classifier.c`
- `tipn_houi.c`
- `tkq.c`
- `topkphm.c`

Some file I/O is legitimate for optional taxonomy/profit/model files or pattern output, but data ingestion should migrate toward connectors. HIEP is closest conceptually to the new design because it already consumes raw text/token windows, but it still owns file reading internally.

## Verdict

The existing repository now has a single primary runner for connector-backed mining:

```text
raw file -> mmap connector/tokenizer -> DM_FlatDataset -> DM_Plugin
                                             -> DM_FlatDataset adapter -> DM_Dataset family -> DM_Algorithm
```

New algorithms should target `DM_Plugin` first. Existing `DM_Algorithm` implementations are no longer isolated from the connector layer: `dm_run` now adapts `DM_FlatDataset` into the legacy family structures automatically.
