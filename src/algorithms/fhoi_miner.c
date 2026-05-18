#include "algorithms/mfhoi.h"
#include "core/dm_algorithm.h"
#include "core/dm_benchmark.h"
#include "core/dm_dataset_types.h"
#include <math.h>
#include <stdio.h>

extern DM_Status mfhoi_adapter_run(DM_Dataset *ds, void *params);

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_MFHOI_Params p = params ? *(DM_MFHOI_Params *)params : (DM_MFHOI_Params){0.01, 0.3, false, MFHOI_ALGO_FHOI};
    p.strong = false;
    p.output_algorithm = MFHOI_ALGO_FHOI;
    return mfhoi_adapter_run(ds, &p);
}

static DM_Algorithm algo = {
    .id = "fhoi_miner",
    .name = "FHOI-Miner",
    .description = "Frequent High-Occupancy Itemset Miner baseline.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
