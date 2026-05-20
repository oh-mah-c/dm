# dm Core Next: Embedded Knowledge Discovery Engine

This layer is the shared infrastructure for modern dm algorithms. It separates data ingestion from mining logic so algorithms can consume one normalized representation:

```c
uint32_t *items;
size_t   *row_offsets;
```

Each row is a transaction, token window, sequence fragment, or graph-derived neighborhood. The original source may be SPMF transactions, raw text, or graph edges.

## Components

- `DM_Arena`: durable bump allocator for algorithm/plugin memory. All allocations are released with one `dm_arena_reset()` or `dm_arena_free()`.
- `DM_MMap`: zero-copy file mapping for connector input.
- `DM_FlatDataset`: flat integer dataset shared by algorithms.
- `dm_connect`: smoke-test CLI for universal connectors.
- `DM_Plugin`: plug-and-play algorithm interface over `DM_FlatDataset`.
- `DM_FlatDataset` adapters: convert the flat connector output into the internal family structures expected by older algorithms (`DM_Trans_Simple`, `DM_Trans_Utility`, `DM_Sequence_Utility`, `DM_Trans_Quantity`, and `DM_Matrix_Row`).
- `dm_run`: unified connector runner for native flat plugins, standalone adapters, and old `DM_Algorithm` registry entries.

## Connectors

- `spmf`: reads classic integer transaction files.
- `text`: tokenizes raw text with the built-in FARO tokenizer into sliding token windows. This is the current lightweight tokenizer path for language mining and HIEP-style entropy mining.
- `graph`: reads an edge list and emits adjacency rows. Use `--undirected` for symmetric neighborhoods.

## Example

```bash
bin/dm_connect --input datasets/itemsets/mushrooms.txt --connector spmf
bin/dm_connect --input README.md --connector text --window 64 --stride 32
bin/dm_connect --input graph_edges.txt --connector graph --undirected
bin/dm_run --algorithm topk_items --input README.md --connector text --k 20
bin/dm_run --algorithm fpgrowth --input datasets/synthetic/syn_sparse.txt --connector spmf
bin/dm_run --algorithm tku_miner --input datasets/synthetic/syn_sparse.txt --connector spmf --k 20
bin/dm_run --algorithm huciminer --input datasets/utilities/foodmart.txt --minutil 5000 --minconf 0.8
```

No connector writes an intermediate dataset file. The pipeline maps source bytes, builds the flat integer view in an arena, adapts it to the target algorithm family when needed, prints statistics, and releases the arena in O(1). Algorithms with paper-specific raw parsers are still invoked through `dm_run`, so the user-facing entry point is unified while exact input semantics are preserved. Utility algorithms such as HUCI-Miner use their raw utility parser when the algorithm requires per-item utilities that cannot be represented by the current item-only flat view.
