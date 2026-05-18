#include "algorithms/mfhoi.h"
#include "core/dm_algorithm.h"

extern DM_Status mfhoi_adapter_run(DM_Dataset *ds, void *params);

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_MFHOI_Params p = params ? *(DM_MFHOI_Params *)params : (DM_MFHOI_Params){0.01, 0.3, true, MFHOI_ALGO_STRONG};
    p.strong = true;
    p.output_algorithm = MFHOI_ALGO_STRONG;
    return mfhoi_adapter_run(ds, &p);
}

static DM_Algorithm algo = {
    .id = "strong_mfhoi_miner",
    .name = "Strong MFHOI-Miner",
    .description = "Strong occupancy-dominance maximal FHOI miner.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
