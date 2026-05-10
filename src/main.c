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
#include "algorithms/mafia.h"
#include "algorithms/hep.h"
#include "algorithms/dfhoi.h"
#include "algorithms/fhoi.h"
#include "algorithms/tkhoim.h"
#include "algorithms/hoimto.h"
#include "algorithms/cfi_stream.h"
#include "algorithms/fin.h"
#include "algorithms/finplus.h"
#include "algorithms/negfin.h"
#include "algorithms/prepostplus.h"
#include "algorithms/lcmver2.h"
#include "algorithms/dic.h"
#include "algorithms/ltm.h"
#include "algorithms/sam.h"
#include "algorithms/carpenter.h"
#include "algorithms/dbv_miner.h"
#include "algorithms/defme.h"
#include "algorithms/talky_g.h"
#include "algorithms/pascal.h"
#include "algorithms/zart.h"
#include "algorithms/apriori_rare.h"
#include "algorithms/apriori_inverse.h"
#include "algorithms/cori.h"
#include "algorithms/rp_tree.h"
#include "algorithms/clostream.h"
#include "algorithms/estdec.h"
#include "algorithms/uapriori.h"
#include "algorithms/msapriori.h"
#include "algorithms/ffiminer.h"
#include "algorithms/ubmffp.h"
#include "algorithms/close.h"
#include "algorithms/dfi_growth.h"
#include "algorithms/opus_miner.h"
#include "algorithms/apriori_tid.h"
#include "algorithms/apriori_hybrid.h"
#include "algorithms/krimp.h"
#include "algorithms/slim.h"
#include "algorithms/two_phase.h"
#include "algorithms/fhm.h"
#include "algorithms/efim.h"
#include "algorithms/hui_miner.h"
#include "algorithms/up_growth.h"
#include "algorithms/ihup.h"
#include "algorithms/huim_su.h"
#include "algorithms/ulb_miner.h"
#include "algorithms/ufh.h"
#include "algorithms/huci_miner.h"
#include "algorithms/up_hist.h"
#include "algorithms/r_miner.h"
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
    DM_MAFIA_Params mafia_params;
    DM_HEP_Params hep_params;
    DM_DFHOI_Params dfhoi_params;
    DM_FHOI_Params fhoi_params;
    DM_TKHOIM_Params tkhoim_params;
    DM_HOIMTO_Params hoimto_params;
    DM_CFI_STREAM_Params cfi_stream_params;
    DM_FIN_Params fin_params;
    DM_FINPLUS_Params finplus_params;
    DM_NEGFIN_Params negfin_params;
    DM_PrePostPlus_Params prepostplus_params;
    DM_LCMVER2_Params lcmver2_params;
    DM_DIC_Params dic_params;
    DM_LTM_Params ltm_params;
    DM_SAM_Params sam_params;
    DM_CARPENTER_Params carpenter_params;
    DM_DBV_MINER_Params dbv_miner_params;
    DM_DEFME_Params defme_params;
    DM_TALKY_G_Params talky_g_params;
    DM_PASCAL_Params pascal_params;
    DM_ZART_Params zart_params;
    DM_APRIORI_RARE_Params apriori_rare_params;
    DM_APRIORI_INVERSE_Params apriori_inverse_params;
    DM_CORI_Params cori_params;
    DM_RP_TREE_Params rp_tree_params;
    DM_CLOSTREAM_Params clostream_params;
    DM_ESTDEC_Params estdec_params;
    DM_UAPRIORI_Params uapriori_params;
    DM_MSAPRIORI_Params msapriori_params;
    DM_FFIMINER_Params ffiminer_params;
    DM_UBMFFP_Params ubmffp_params;
    DM_CLOSE_Params close_params;
    DM_DFI_GROWTH_Params dfigrowth_params;
    DM_OPUS_MINER_Params opus_miner_params;
    DM_APRIORI_TID_Params apriori_tid_params;
    DM_APRIORI_HYBRID_Params apriori_hybrid_params;
    DM_KRIMP_Params krimp_params;
    DM_SLIM_Params slim_params;
    DM_Two_Phase_Params twophase_params;
    DM_FHM_Params fhm_params;
    DM_EFIM_Params efim_params;
    DM_HUI_Miner_Params huiminer_params;
    DM_UP_Growth_Params upgrowth_params;
    DM_IHUP_Params ihup_params;
    DM_HUIM_SU_Params huimsu_params;
    DM_ULB_Miner_Params ulbminer_params;
    DM_UFH_Params ufh_params;
    DM_HUCI_Miner_Params huciminer_params;
    DM_UP_Hist_Params uphist_params;
    DM_R_Miner_Params rminer_params;
    double min_bond = (argc >= 6) ? atof(argv[5]) : 0.2; // Default bond or 6th arg
    if (strcmp(algo_id, "ais") == 0) {
        ais_params.min_support = min_support;
        params = &ais_params;
    } else if (strcmp(algo_id, "apriori") == 0) {
        apriori_params.min_support = min_support;
        apriori_params.min_confidence = (argc >= 6) ? atof(argv[5]) : 0.8;
        params = &apriori_params;
    } else if (strcmp(algo_id, "apriori_tid") == 0) {
        apriori_tid_params.min_support = min_support;
        apriori_tid_params.min_confidence = (argc >= 6) ? atof(argv[5]) : 0.8;
        params = &apriori_tid_params;
    } else if (strcmp(algo_id, "apriori_hybrid") == 0) {
        apriori_hybrid_params.min_support = min_support;
        apriori_hybrid_params.min_confidence = (argc >= 6) ? atof(argv[5]) : 0.8;
        params = &apriori_hybrid_params;
    } else if (strcmp(algo_id, "krimp") == 0) {
        krimp_params.min_support = min_support;
        krimp_params.prune = (argc >= 6) ? (atoi(argv[5]) != 0) : true;
        params = &krimp_params;
    } else if (strcmp(algo_id, "slim") == 0) {
        slim_params.prune = (argc >= 6) ? (atoi(argv[5]) != 0) : true;
        params = &slim_params;
    } else if (strcmp(algo_id, "twophase") == 0) {
        twophase_params.min_utility = min_support; // User uses min_support arg as min_utility
        params = &twophase_params;
    } else if (strcmp(algo_id, "fhm") == 0) {
        fhm_params.min_utility = min_support;
        params = &fhm_params;
    } else if (strcmp(algo_id, "efim") == 0) {
        efim_params.min_utility = min_support;
        params = &efim_params;
    } else if (strcmp(algo_id, "huiminer") == 0) {
        huiminer_params.min_utility = min_support;
        params = &huiminer_params;
    } else if (strcmp(algo_id, "upgrowth") == 0) {
        upgrowth_params.min_utility = min_support;
        params = &upgrowth_params;
    } else if (strcmp(algo_id, "ihup") == 0) {
        ihup_params.min_utility = min_support;
        params = &ihup_params;
    } else if (strcmp(algo_id, "huimsu") == 0) {
        huimsu_params.min_utility = min_support;
        params = &huimsu_params;
    } else if (strcmp(algo_id, "ulbminer") == 0) {
        ulbminer_params.min_utility = min_support;
        ulbminer_params.buffer_size = 1000000;
        params = &ulbminer_params;
    } else if (strcmp(algo_id, "ufh") == 0) {
        ufh_params.min_utility = min_support;
        params = &ufh_params;
    } else if (strcmp(algo_id, "huciminer") == 0) {
        huciminer_params.min_utility = min_support;
        huciminer_params.min_confidence = 0.8;
        params = &huciminer_params;
    } else if (strcmp(algo_id, "uphist") == 0) {
        uphist_params.min_utility = min_support;
        params = &uphist_params;
    } else if (strcmp(algo_id, "rminer") == 0) {
        rminer_params.min_utility = min_support;
        params = &rminer_params;
    } else if (strcmp(algo_id, "eclat") == 0) {
        eclat_params.min_support = min_support;
        params = &eclat_params;
    } else if (strcmp(algo_id, "fpgrowth") == 0) {
        fpgrowth_params.min_support = min_support;
        params = &fpgrowth_params;
    } else if (strcmp(algo_id, "aclose") == 0) {
        aclose_params.min_support = min_support;
        params = &aclose_params;
    } else if (strcmp(algo_id, "close") == 0) {
        close_params.min_support = min_support;
        close_params.min_confidence = (argc >= 6) ? atof(argv[5]) : 0.8;
        params = &close_params;
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
    } else if (strcmp(algo_id, "mafia") == 0) {
        mafia_params.min_support = min_support;
        params = &mafia_params;
    } else if (strcmp(algo_id, "hep") == 0) {
        hep_params.min_occupancy = min_support;
        params = &hep_params;
    } else if (strcmp(algo_id, "dfhoi") == 0) {
        dfhoi_params.min_occupancy = min_support;
        params = &dfhoi_params;
    } else if (strcmp(algo_id, "fhoi") == 0) {
        fhoi_params.min_occupancy = min_support;
        params = &fhoi_params;
    } else if (strcmp(algo_id, "tkhoim") == 0) {
        tkhoim_params.k = (size_t)min_support;
        params = &tkhoim_params;
    } else if (strcmp(algo_id, "hoimto") == 0) {
        hoimto_params.min_sup = min_support;
        hoimto_params.min_io = (argc >= 6) ? atof(argv[5]) : min_support;
        params = &hoimto_params;
    } else if (strcmp(algo_id, "cfi_stream") == 0) {
        cfi_stream_params.min_support = min_support;
        params = &cfi_stream_params;
    } else if (strcmp(algo_id, "fin") == 0) {
        fin_params.min_support = min_support;
        params = &fin_params;
    } else if (strcmp(algo_id, "finplus") == 0) {
        finplus_params.min_support = min_support;
        params = &finplus_params;
    } else if (strcmp(algo_id, "negfin") == 0) {
        negfin_params.min_support = min_support;
        params = &negfin_params;
    } else if (strcmp(algo_id, "prepostplus") == 0) {
        prepostplus_params.min_support = min_support;
        params = &prepostplus_params;
    } else if (strcmp(algo_id, "lcmver2") == 0) {
        lcmver2_params.min_support = min_support;
        lcmver2_params.mode = 0; // Default to ALL
        params = &lcmver2_params;
    } else if (strcmp(algo_id, "dic") == 0) {
        dic_params.min_support = min_support;
        dic_params.block_size = 1000; // Default M=1000
        params = &dic_params;
    } else if (strcmp(algo_id, "ltm") == 0) {
        ltm_params.min_support = min_support;
        params = &ltm_params;
    } else if (strcmp(algo_id, "sam") == 0) {
        sam_params.min_support = min_support;
        params = &sam_params;
    } else if (strcmp(algo_id, "carpenter") == 0) {
        carpenter_params.min_support = min_support;
        params = &carpenter_params;
    } else if (strcmp(algo_id, "dbv_miner") == 0) {
        dbv_miner_params.min_support = min_support;
        params = &dbv_miner_params;
    } else if (strcmp(algo_id, "defme") == 0) {
        defme_params.min_support = min_support;
        params = &defme_params;
    } else if (strcmp(algo_id, "talky_g") == 0) {
        talky_g_params.min_support = min_support;
        params = &talky_g_params;
    } else if (strcmp(algo_id, "pascal") == 0) {
        pascal_params.min_support = min_support;
        params = &pascal_params;
    } else if (strcmp(algo_id, "zart") == 0) {
        zart_params.min_support = min_support;
        params = &zart_params;
    } else if (strcmp(algo_id, "apriori_rare") == 0) {
        apriori_rare_params.min_support = min_support;
        params = &apriori_rare_params;
    } else if (strcmp(algo_id, "apriori_inverse") == 0) {
        apriori_inverse_params.max_support = min_support; // use min_support flag for max_support
        apriori_inverse_params.min_abs_support = 5;
        params = &apriori_inverse_params;
    } else if (strcmp(algo_id, "cori") == 0) {
        cori_params.min_support = min_support;
        cori_params.min_bond = min_bond;
        params = &cori_params;
    } else if (strcmp(algo_id, "rp_tree") == 0) {
        rp_tree_params.min_freq_support = min_support;
        rp_tree_params.min_rare_support = (argc >= 6) ? atof(argv[5]) : 0.001;
        params = &rp_tree_params;
    } else if (strcmp(algo_id, "clostream") == 0) {
        clostream_params.min_support = min_support;
        params = &clostream_params;
    } else if (strcmp(algo_id, "estdec") == 0) {
        estdec_params.min_support = min_support;
        estdec_params.ins_threshold = (argc >= 6) ? atof(argv[5]) : min_support * 0.5;
        estdec_params.prn_threshold = (argc >= 7) ? atof(argv[6]) : min_support * 0.1;
        estdec_params.decay_base = (argc >= 8) ? atof(argv[7]) : 2.0;
        estdec_params.decay_life = (argc >= 9) ? atof(argv[8]) : 10000.0;
        params = &estdec_params;
    } else if (strcmp(algo_id, "uapriori") == 0) {
        uapriori_params.min_support = min_support;
        params = &uapriori_params;
    } else if (strcmp(algo_id, "msapriori") == 0) {
        msapriori_params.LS = min_support;
        msapriori_params.beta = (argc >= 6) ? atof(argv[5]) : 1.0;
        msapriori_params.mis_values = NULL;
    } else if (strcmp(algo_id, "ffiminer") == 0) {
        ffiminer_params.min_support = min_support;
        params = &ffiminer_params;
    } else if (strcmp(algo_id, "ubmffp") == 0) {
        ubmffp_params.min_support = min_support;
        params = &ubmffp_params;
    } else if (strcmp(algo_id, "dfigrowth") == 0) {
        dfigrowth_params.min_support = min_support;
        params = &dfigrowth_params;
    } else if (strcmp(algo_id, "opus_miner") == 0) {
        opus_miner_params.k = (int)min_support; // k using min_support flag
        opus_miner_params.alpha = (argc >= 6) ? atof(argv[5]) : 0.05;
        opus_miner_params.measure = DM_OPUS_MEASURE_LEVERAGE;
        opus_miner_params.check_indep = true;
        params = &opus_miner_params;
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

