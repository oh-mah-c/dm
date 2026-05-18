#include "algorithms/mfhoi.h"
#include "core/dm_algorithm.h"

extern DM_Status mfhoi_adapter_run(DM_Dataset *ds, void *params);

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_MFHOI_Params p = params ? *(DM_MFHOI_Params *)params : (DM_MFHOI_Params){0.01, 0.3, false, MFHOI_ALGO_WEAK};
    p.strong = false;
    p.output_algorithm = MFHOI_ALGO_WEAK;
    return mfhoi_adapter_run(ds, &p);
}

static DM_Algorithm algo = {
    .id = "weak_mfhoi_miner",
    .name = "Weak MFHOI-Miner",
    .description = "Weak occupancy-dominance maximal FHOI miner.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
