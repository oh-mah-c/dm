#include "core/dm_algorithm.h"
#include "core/dm_benchmark.h"
#include "algorithms/ais.h"
#include "algorithms/apriori.h"
#include "algorithms/eclat.h"
#include "algorithms/fpgrowth.h"
#include "algorithms/aclose.h"
#include "algorithms/closet.h"
#include "algorithms/closetplus.h"
#include "algorithms/fpclose.h"
#include "algorithms/charm.h"
#include "algorithms/dci_closed.h"
#include "algorithms/max_miner.h"
#include "algorithms/tree_projection.h"
#include "algorithms/genmax.h"
#include "algorithms/fpmax.h"
#include "algorithms/lcm.h"
#include "algorithms/nafcp.h"
#include "algorithms/fcfia.h"
#include "algorithms/prepost.h"
#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <algo_id> <dataset_path> [type_id] [min_support]\n", argv[0]);
        printf("Types: 0=Transactional, 1=Utility, 2=Matrix\n");
        dm_list_algorithms();
        return 1;
    }

    const char *algo_id = argv[1];
    const char *path = argv[2];
    DM_DatasetType type = (argc >= 4) ? (DM_DatasetType)atoi(argv[3]) : DM_TYPE_TRANSACTIONAL;
    double min_support = (argc >= 5) ? atof(argv[4]) : 0.01;

    DM_Algorithm *algo = dm_get_algorithm(algo_id);
    if (!algo) {
        printf("Error: Algorithm '%s' not found.\n", algo_id);
        dm_list_algorithms();
        return 1;
    }

    dm_bench_reset();
    dm_bench_start(DM_PHASE_TOTAL);

    printf("Loading dataset (%d): %s\n", type, path);
    dm_bench_start(DM_PHASE_LOAD);
    DM_Dataset *ds = dm_dataset_load(path, type);
    dm_bench_stop(DM_PHASE_LOAD);
    
    if (!ds) {
        printf("Error: Could not load dataset at %s\n", path);
        return 1;
    }

    // Set up parameters
    void *params = NULL;
    DM_AIS_Params ais_params;
    DM_APRIORI_Params apriori_params;
    DM_ECLAT_Params eclat_params;
    DM_FPGROWTH_Params fpgrowth_params;
    DM_ACLOSE_Params aclose_params;
    DM_CLOSET_Params closet_params;
    DM_CLOSETPLUS_Params closetplus_params;
    DM_FPCLOSE_Params fpclose_params;
    DM_CHARM_Params charm_params;
    DM_DCI_CLOSED_Params dci_closed_params;
    DM_MAX_MINER_Params max_miner_params;
    DM_TREE_PROJECTION_Params tree_projection_params;
    DM_GENMAX_Params genmax_params;
    DM_FPMAX_Params fpmax_params;
    DM_LCM_Params lcm_params;
    DM_NAFCP_Params nafcp_params;
    DM_FCFIA_Params fcfia_params;
    DM_PrePost_Params prepost_params;
    if (strcmp(algo_id, "ais") == 0) {
        ais_params.min_support = min_support;
        params = &ais_params;
    } else if (strcmp(algo_id, "apriori") == 0) {
        apriori_params.min_support = min_support;
        params = &apriori_params;
    } else if (strcmp(algo_id, "eclat") == 0) {
        eclat_params.min_support = min_support;
        params = &eclat_params;
    } else if (strcmp(algo_id, "fpgrowth") == 0) {
        fpgrowth_params.min_support = min_support;
        params = &fpgrowth_params;
    } else if (strcmp(algo_id, "aclose") == 0) {
        aclose_params.min_support = min_support;
        params = &aclose_params;
    } else if (strcmp(algo_id, "closet") == 0) {
        closet_params.min_support = min_support;
        params = &closet_params;
    } else if (strcmp(algo_id, "closetplus") == 0) {
        closetplus_params.min_support = min_support;
        params = &closetplus_params;
    } else if (strcmp(algo_id, "fpclose") == 0) {
        fpclose_params.min_support = min_support;
        params = &fpclose_params;
    } else if (strcmp(algo_id, "charm") == 0) {
        charm_params.min_support = min_support;
        params = &charm_params;
    } else if (strcmp(algo_id, "dci_closed") == 0) {
        dci_closed_params.min_support = min_support;
        params = &dci_closed_params;
    } else if (strcmp(algo_id, "max_miner") == 0) {
        max_miner_params.min_support = min_support;
        params = &max_miner_params;
    } else if (strcmp(algo_id, "tree_projection") == 0) {
        tree_projection_params.min_support = min_support;
        params = &tree_projection_params;
    } else if (strcmp(algo_id, "genmax") == 0) {
        genmax_params.min_support = min_support;
        params = &genmax_params;
    } else if (strcmp(algo_id, "fpmax") == 0) {
        fpmax_params.min_support = min_support;
        params = &fpmax_params;
    } else if (strcmp(algo_id, "lcm") == 0) {
        lcm_params.min_support = min_support;
        params = &lcm_params;
    } else if (strcmp(algo_id, "nafcp") == 0) {
        nafcp_params.min_support = min_support;
        params = &nafcp_params;
    } else if (strcmp(algo_id, "fcfia") == 0) {
        fcfia_params.min_support = min_support;
        params = &fcfia_params;
    } else if (strcmp(algo_id, "prepost") == 0) {
        prepost_params.min_support = min_support;
        params = &prepost_params;
    }

    printf("Executing %s...\n", algo->name);
    dm_bench_start(DM_PHASE_ALGO);
    DM_Status status = algo->run(ds, params);
    dm_bench_stop(DM_PHASE_ALGO);
    
    if (status != DM_SUCCESS) {
        printf("Execution failed with code %d\n", status);
    }

    // Dummy phase for write (until output module is built)
    dm_bench_start(DM_PHASE_WRITE);
    dm_bench_stop(DM_PHASE_WRITE);

    dm_dataset_free(ds);
    
    dm_bench_stop(DM_PHASE_TOTAL);
    
    dm_bench_print_report(algo->name, path);
    return 0;
}

