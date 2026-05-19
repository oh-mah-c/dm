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
#include "algorithms/mfhoi.h"
#include "algorithms/mhoui.h"
#include "algorithms/vifp.h"
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
#include "algorithms/vhuqi.h"
#include "algorithms/fhuqi_miner.h"
#include "algorithms/tkq.h"
#include "algorithms/efim.h"
#include "algorithms/dphim.h"
#include "algorithms/hui_miner.h"
#include "algorithms/up_growth.h"
#include "algorithms/ihup.h"
#include "algorithms/huim_su.h"
#include "algorithms/ulb_miner.h"
#include "algorithms/ufh.h"
#include "algorithms/huci_miner.h"
#include "algorithms/up_hist.h"
#include "algorithms/r_miner.h"
#include "algorithms/sum.h"
#include "algorithms/clh_miner.h"
#include "algorithms/feacp.h"
#include "algorithms/mlhui_miner.h"
#include "algorithms/fchm.h"
#include "algorithms/foshu.h"
#include "algorithms/tshoun.h"
#include "algorithms/eihi.h"
#include "algorithms/hui_list_ins.h"
#include "algorithms/efim_closed.h"
#include "algorithms/chui_miner.h"
#include "algorithms/chuimine.h"
#include "algorithms/cls_miner.h"
#include "algorithms/ghui_miner.h"
#include "algorithms/fhim.h"
#include "algorithms/minfhm.h"
#include "algorithms/skymine.h"
#include "algorithms/sfui_uf.h"
#include "algorithms/sfu_ce.h"
#include "algorithms/uspan.h"
#include "algorithms/hupspm.h"
#include "algorithms/hup_miner.h"
#include "algorithms/huim_bpso.h"
#include "algorithms/hupe_garm.h"
#include "algorithms/huim_aco.h"
#include "algorithms/huim_hc.h"
#include "algorithms/huim_sa.h"
#include "algorithms/huim_bpso_tree.h"
#include "algorithms/bio_huif.h"
#include "algorithms/huim_afsa.h"
#include "algorithms/huim_abc.h"
#include "algorithms/skyline_miner.h"
#include "algorithms/thui.h"
#include "algorithms/tku_ce.h"
#include "algorithms/tku_ce_plus.h"
#include "algorithms/fhn.h"
#include "algorithms/fhmds.h"
#include "algorithms/haui_miner.h"
#include "algorithms/ehaupm.h"
#include "algorithms/hauim_gmu.h"
#include "algorithms/nam_hep.h"
#include "algorithms/memu.h"
#include "algorithms/mheinu.h"
#include "algorithms/closed_fhuim_kinana.h"
#include "algorithms/regular_mine.h"
#include "algorithms/tmku.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern DM_Algorithm bio_huif_ga_algo;
extern DM_Algorithm tmku_algo;
extern DM_Algorithm bio_huif_pso_algo;
extern DM_Algorithm bio_huif_ba_algo;
extern DM_Algorithm huim_afsa_algo;
extern DM_Algorithm huim_abc_algo;
extern DM_Algorithm skyline_miner_algo;
extern DM_Algorithm thui_algo;
extern DM_Algorithm tku_ce_algo;
extern DM_Algorithm tku_ce_plus_algo;
extern DM_Algorithm fhmds_algo;
extern DM_Algorithm haui_miner_algo;
extern DM_Algorithm ehaupm_algo;
extern DM_Algorithm hauim_gmu_algo;
extern DM_Algorithm nam_hep_algo;
extern DM_Algorithm memu_algo;
extern DM_Algorithm mheinu_algo;
extern DM_Algorithm dphim_algo;
extern DM_Algorithm closed_fhuim_kinana_algo;
extern DM_Algorithm hup_miner_algo;
extern DM_Algorithm regular_mine_algo;
extern DM_Algorithm mhoui_algo;
extern DM_Algorithm vifp_algo;

int main(int argc, char **argv) {
    dm_register_algorithm(&bio_huif_ga_algo);
    dm_register_algorithm(&bio_huif_pso_algo);
    dm_register_algorithm(&bio_huif_ba_algo);
    dm_register_algorithm(&huim_afsa_algo);
    dm_register_algorithm(&huim_abc_algo);
    dm_register_algorithm(&skyline_miner_algo);
    dm_register_algorithm(&thui_algo);
    dm_register_algorithm(&tku_ce_algo);
    dm_register_algorithm(&tku_ce_plus_algo);
    dm_register_algorithm(&fhmds_algo);
    dm_register_algorithm(&haui_miner_algo);
    dm_register_algorithm(&ehaupm_algo);
    dm_register_algorithm(&hauim_gmu_algo);
    dm_register_algorithm(&nam_hep_algo);
    dm_register_algorithm(&memu_algo);
    dm_register_algorithm(&mheinu_algo);
    dm_register_algorithm(&dphim_algo);
    dm_register_algorithm(&closed_fhuim_kinana_algo);
    dm_register_algorithm(&hup_miner_algo);
    dm_register_algorithm(&regular_mine_algo);
    dm_register_algorithm(&mhoui_algo);
    dm_register_algorithm(&vifp_algo);
    dm_register_algorithm(&tmku_algo);

    if (argc < 3) {
        printf("Usage: %s <algo_id> <dataset_path> [type_id] [min_support]\n", argv[0]);
        printf("MFHOI: %s mfhoi <dataset_path> 0 <min_support> <min_occupancy> [strong:0|1]\n", argv[0]);
        printf("MHOUI: %s mhoui <utility_dataset> 1 <min_support> <min_occupancy> <min_utility> [strong:0|1] [direct:0|1]\n", argv[0]);
        printf("MHEINU: %s mheinu <utility_dataset> 1 <min_efficiency> [investment_file]\n", argv[0]);
        printf("DPHIM: %s dphim <utility_dataset> 1 <min_utility> [threads]\n", argv[0]);
        printf("Closed-FHUIM-Kinana: %s closed_fhuim_kinana <utility_dataset> 1 <min_utility> <min_support> [min_owl]\n", argv[0]);
        printf("HUP-Miner: %s hup_miner <sequence_utility_dataset> 3 <min_average_utility> [max_length] [max_candidates]\n", argv[0]);
        printf("RegularMine: %s regular_mine <transactional_dataset> 0 <min_support>\n", argv[0]);
        printf("VIFP: %s vifp <transactional_dataset> 0 <min_support> [mode:plaintext|smpc|fhe]\n", argv[0]);
        printf("TMKU: %s tmku <utility_dataset> 1 [k] [min_utility] [target_pattern]\n", argv[0]);
        printf("Types: 0=Transactional, 1=Utility, 2=Matrix, 3=SequenceUtility, 4=Quantity\n");
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
    DM_MFHOI_Params mfhoi_params;
    DM_MHOUI_Params mhoui_params;
    DM_VIFP_Params vifp_params;
    DM_TKHOIM_Params tkhoim_params;
    DM_HOIMTO_Params hoimto_params;
    DM_CFI_STREAM_Params cfi_stream_params;
    DM_HUIM_SA_Params huim_sa_params;
    DM_TMKU_Params tmku_params;
    DM_HUIM_BPSO_Tree_Params huim_bpso_tree_params;
    DM_BioHUIF_Params bio_huif_params;
    DM_HUIM_AFSA_Params afsa_params;
    DM_HUIM_ABC_Params abc_params;
    DM_Skyline_Miner_Params skyline_miner_params;
    DM_THUI_Params thui_params;
    DM_TKU_CE_Params tku_ce_params;
    DM_TKU_CE_Plus_Params tku_ce_plus_params;
    DM_FHN_Params fhn_params;
    DM_DPHIM_Params dphim_params;
    DM_FHMDS_Params fhmds_params;
    DM_HAUI_Miner_Params haui_miner_params;
    DM_EHAUPM_Params ehaupm_params;
    DM_HAUIM_GMU_Params hauim_gmu_params;
    DM_NAM_HEP_Params nam_hep_params;
    DM_MEMU_Params memu_params;
    DM_MHEINU_Params mheinu_params;
    DM_ClosedFHUIMKinana_Params closed_fhuim_kinana_params;
    DM_REGULAR_MINE_Params regular_mine_params;
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
    DM_VHUQI_Params vhuqi_params;
    DM_FHUQI_Miner_Params fhuqi_miner_params;
    DM_TKQ_Params tkq_params;
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
    static DM_SUM_Params sum_params;
    static DM_CLH_Miner_Params clhminer_params;
    static DM_FEACP_Params feacp_params;
    static DM_MLHUI_Miner_Params mlhuiminer_params;
    static DM_FCHM_Params fchm_params;
    DM_FOSHU_Params foshu_params = {0.43, 5};
    static DM_TSHOUN_Params tshoun_params;
    static DM_EIHI_Params eihi_params;
    static DM_HUI_LIST_INS_Params hli_params;
    static DM_EFIM_Closed_Params efc_params;
    static DM_CHUI_Miner_Params chui_params;
    static DM_CHUIMINE_Params cm_params;
    static DM_CLS_Miner_Params clsm_params;
    static DM_GHUI_Miner_Params ghuim_params;
    static DM_FHIM_Params fhim_params;
    static DM_MinFHM_Params minfhm_params;
    static DM_SKYMINE_Params skymine_params;
    static DM_SFUI_UF_Params sfui_uf_params;
    static DM_SFU_CE_Params sfu_ce_params = { .sample_size = 1000, .max_iterations = 100, .quantile = 0.1, .mutation_factor = 0.2 };
    static DM_USPAN_Params uspan_params;
    static DM_HUPSPM_Params hupspm_params;
    static DM_HUP_Miner_Params hup_miner_params;
    static DM_HUIM_BPSO_Params huim_bpso_params;
    static DM_HUPE_GARM_Params hupe_garm_params;
    static DM_HUIM_ACO_Params huim_aco_params;
    DM_HUIM_HC_Params huim_hc_params;
    static double min_uconf = 0.8;
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
    } else if (strcmp(algo_id, "vhuqi") == 0) {
        vhuqi_params.min_abs_utility = min_support;
        vhuqi_params.qrc = (argc >= 6) ? atof(argv[5]) : 3.0;
        params = &vhuqi_params;
    } else if (strcmp(algo_id, "fhuqi_miner") == 0) {
        fhuqi_miner_params.min_utility = min_support;
        fhuqi_miner_params.qrc = (argc >= 6) ? atof(argv[5]) : 3.0;
        fhuqi_miner_params.combine_method = DM_FHUQI_COMBINE_ALL;
        if (argc >= 7) {
            if (strcmp(argv[6], "min") == 0) fhuqi_miner_params.combine_method = DM_FHUQI_COMBINE_MIN;
            else if (strcmp(argv[6], "max") == 0) fhuqi_miner_params.combine_method = DM_FHUQI_COMBINE_MAX;
        }
        fhuqi_miner_params.profit_path = (argc >= 8) ? argv[7] : NULL;
        params = &fhuqi_miner_params;
    } else if (strcmp(algo_id, "tkq") == 0) {
        tkq_params.k = (size_t)min_support;
        tkq_params.qrc = (argc >= 6) ? atof(argv[5]) : 3.0;
        tkq_params.combine_method = DM_FHUQI_COMBINE_ALL;
        if (argc >= 7) {
            if (strcmp(argv[6], "min") == 0) tkq_params.combine_method = DM_FHUQI_COMBINE_MIN;
            else if (strcmp(argv[6], "max") == 0) tkq_params.combine_method = DM_FHUQI_COMBINE_MAX;
        }
        tkq_params.profit_path = (argc >= 8) ? argv[7] : NULL;
        params = &tkq_params;
    } else if (strcmp(algo_id, "efim") == 0) {
        efim_params.min_utility = min_support;
        params = &efim_params;
    } else if (strcmp(algo_id, "dphim") == 0) {
        dphim_params.min_utility = min_support;
        dphim_params.threads = (argc >= 6) ? atoi(argv[5]) : 0;
        params = &dphim_params;
    } else if (strcmp(algo_id, "closed_fhuim_kinana") == 0) {
        closed_fhuim_kinana_params.min_utility = min_support;
        closed_fhuim_kinana_params.min_support = (argc >= 6) ? atoi(argv[5]) : 20;
        closed_fhuim_kinana_params.min_owl = (argc >= 7) ? atof(argv[6]) : 0.0;
        params = &closed_fhuim_kinana_params;
    } else if (strcmp(algo_id, "regular_mine") == 0) {
        regular_mine_params.min_support = min_support;
        params = &regular_mine_params;
    } else if (strcmp(algo_id, "vifp") == 0) {
        vifp_params.min_support = min_support;
        if (vifp_parse_mode((argc >= 6) ? argv[5] : "plaintext", &vifp_params.mode) != 0) {
            vifp_params.mode = VIFP_MODE_PLAINTEXT;
        }
        vifp_params.max_itemsets = 0;
        vifp_params.max_seconds = 0.0;
        params = &vifp_params;
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
    } else if (strcmp(algo_id, "fhoi_miner") == 0) {
        mfhoi_params.min_support = min_support;
        mfhoi_params.min_occupancy = (argc >= 6) ? atof(argv[5]) : 0.3;
        mfhoi_params.strong = false;
        mfhoi_params.output_algorithm = MFHOI_ALGO_FHOI;
        params = &mfhoi_params;
    } else if (strcmp(algo_id, "weak_mfhoi_miner") == 0) {
        mfhoi_params.min_support = min_support;
        mfhoi_params.min_occupancy = (argc >= 6) ? atof(argv[5]) : 0.3;
        mfhoi_params.strong = false;
        mfhoi_params.output_algorithm = MFHOI_ALGO_WEAK;
        params = &mfhoi_params;
    } else if (strcmp(algo_id, "strong_mfhoi_miner") == 0) {
        mfhoi_params.min_support = min_support;
        mfhoi_params.min_occupancy = (argc >= 6) ? atof(argv[5]) : 0.3;
        mfhoi_params.strong = true;
        mfhoi_params.output_algorithm = MFHOI_ALGO_STRONG;
        params = &mfhoi_params;
    } else if (strcmp(algo_id, "mfhoi") == 0) {
        mfhoi_params.min_support = min_support;
        mfhoi_params.min_occupancy = (argc >= 6) ? atof(argv[5]) : 0.3;
        mfhoi_params.strong = (argc >= 7) ? (atoi(argv[6]) != 0) : false;
        mfhoi_params.output_algorithm = mfhoi_params.strong ? MFHOI_ALGO_STRONG : MFHOI_ALGO_WEAK;
        params = &mfhoi_params;
    } else if (strcmp(algo_id, "houi_miner") == 0 || strcmp(algo_id, "weak_mhoui_miner") == 0 || strcmp(algo_id, "strong_mhoui_miner") == 0 || strcmp(algo_id, "direct_mhoui_miner") == 0 || strcmp(algo_id, "mhoui") == 0) {
        mhoui_params.min_support = min_support;
        mhoui_params.min_occupancy = (argc >= 6) ? atof(argv[5]) : 0.4;
        mhoui_params.min_utility = (argc >= 7) ? atof(argv[6]) : 1000.0;
        mhoui_params.strong = (argc >= 8) ? (atoi(argv[7]) != 0) : 1;
        mhoui_params.direct = (argc >= 9) ? (atoi(argv[8]) != 0) : (strcmp(algo_id, "direct_mhoui_miner") == 0);
        mhoui_params.max_patterns = 200000;
        mhoui_params.max_seconds = 0.0;
        if (strcmp(algo_id, "houi_miner") == 0) mhoui_params.output_algorithm = MHOUI_ALGO_HOUI;
        else if (strcmp(algo_id, "weak_mhoui_miner") == 0) mhoui_params.output_algorithm = MHOUI_ALGO_WEAK;
        else if (strcmp(algo_id, "strong_mhoui_miner") == 0) mhoui_params.output_algorithm = MHOUI_ALGO_STRONG;
        else mhoui_params.output_algorithm = mhoui_params.strong ? MHOUI_ALGO_DIRECT_STRONG : MHOUI_ALGO_DIRECT_WEAK;
        params = &mhoui_params;
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
    } else if (strcmp(algo_id, "sum") == 0) {
        sum_params.min_utility = min_support;
        sum_params.window_size = (argc >= 6) ? atoi(argv[5]) : 150;
        sum_params.increment_size = (argc >= 7) ? atoi(argv[6]) : 0;
        sum_params.dynamic_threshold = (argc >= 8) ? (atoi(argv[7]) != 0) : true;
        params = &sum_params;
    } else if (strcmp(algo_id, "clhminer") == 0) {
        clhminer_params.min_utility = min_support;
        clhminer_params.taxonomy_path = (argc >= 6) ? argv[5] : NULL;
        params = &clhminer_params;
    } else if (strcmp(algo_id, "feacp") == 0) {
        feacp_params.min_utility = min_support;
        if (argc >= 6) strncpy(feacp_params.taxonomy_path, argv[5], 1024);
        else feacp_params.taxonomy_path[0] = '\0';
        params = &feacp_params;
    } else if (strcmp(algo_id, "mlhui_miner") == 0) {
        mlhuiminer_params.min_utility = min_support;
        mlhuiminer_params.taxonomy_path = (argc >= 6) ? argv[5] : NULL;
        params = &mlhuiminer_params;
    } else if (strcmp(algo_id, "fchm") == 0) {
        fchm_params.min_utility = min_support;
        fchm_params.min_bond = (argc >= 6) ? atof(argv[5]) : 0.2;
        params = &fchm_params;
    } else if (strcmp(algo_id, "fhn") == 0) {
        fhn_params.min_utility = min_support;
        params = &fhn_params;
    } else if (strcmp(algo_id, "foshu") == 0) {
        foshu_params.min_utility = min_support;
        foshu_params.num_periods = (argc >= 6) ? atoi(argv[5]) : 5;
        params = &foshu_params;
    } else if (strcmp(algo_id, "tshoun") == 0) {
        tshoun_params.min_utility = min_support;
        tshoun_params.num_periods = (argc >= 6) ? atoi(argv[5]) : 5;
        params = &tshoun_params;
    } else if (strcmp(algo_id, "eihi") == 0) {
        eihi_params.min_utility = min_support;
        eihi_params.batch_size = (argc >= 6) ? atoi(argv[5]) : -1;
        params = &eihi_params;
    } else if (strcmp(algo_id, "hui_list_ins") == 0) {
        hli_params.min_utility = min_support;
        hli_params.batch_size = (argc >= 6) ? atoi(argv[5]) : -1;
        params = &hli_params;
    } else if (strcmp(algo_id, "efim_closed") == 0) {
        efc_params.min_utility = min_support;
        params = &efc_params;
    } else if (strcmp(algo_id, "chui_miner") == 0) {
        chui_params.min_utility = min_support;
        params = &chui_params;
    } else if (strcmp(algo_id, "chuimine_closed") == 0) {
        cm_params.min_utility = min_support;
        cm_params.find_maximal = false;
        params = &cm_params;
    } else if (strcmp(algo_id, "chuimine_maximal") == 0) {
        cm_params.min_utility = min_support;
        cm_params.find_maximal = true;
        params = &cm_params;
    } else if (strcmp(algo_id, "cls_miner") == 0) {
        clsm_params.min_utility = min_support;
        params = &clsm_params;
    } else if (strcmp(algo_id, "hug_miner") == 0) {
        ghuim_params.min_utility = min_support;
        ghuim_params.mine_ghui = false;
        params = &ghuim_params;
    } else if (strcmp(algo_id, "fhim") == 0) {
        fhim_params.min_utility = min_support;
        params = &fhim_params;
    } else if (strcmp(algo_id, "fhim_rules") == 0) {
        fhim_params.min_utility = min_support;
        min_uconf = (argc >= 6) ? atof(argv[5]) : 0.8;
        params = &fhim_params;
    } else if (strcmp(algo_id, "minfhm") == 0) {
        minfhm_params.min_utility = min_support;
        params = &minfhm_params;
    } else if (strcmp(algo_id, "skymine") == 0) {
        params = &skymine_params;
    } else if (strcmp(algo_id, "sfui_uf") == 0) {
        params = &sfui_uf_params;
    } else if (strcmp(algo_id, "sfu_ce") == 0) {
        params = &sfu_ce_params;
    } else if (strcmp(algo_id, "uspan") == 0) {
        uspan_params.min_utility = min_support;
        params = &uspan_params;
    } else if (strcmp(algo_id, "hupspm") == 0) {
        hupspm_params.min_utility = min_support;
        hupspm_params.min_probability = (argc >= 6) ? atof(argv[5]) : 0.5;
        params = &hupspm_params;
    } else if (strcmp(algo_id, "hup_miner") == 0) {
        hup_miner_params.min_average_utility = min_support;
        hup_miner_params.max_pattern_length = (argc >= 6) ? (size_t)atoi(argv[5]) : 3;
        hup_miner_params.max_candidates = (argc >= 7) ? (size_t)atoll(argv[6]) : 200000;
        params = &hup_miner_params;
    } else if (strcmp(algo_id, "huim_bpso") == 0) {
        huim_bpso_params.min_utility = min_support;
        huim_bpso_params.pop_size = 20;
        huim_bpso_params.max_iter = 1000;
        huim_bpso_params.w = 0.8;
        huim_bpso_params.c1 = 2.0;
        huim_bpso_params.c2 = 2.0;
        params = &huim_bpso_params;
    } else if (strcmp(algo_id, "hupe_garm") == 0) {
        hupe_garm_params.min_utility = min_support;
        hupe_garm_params.pop_size = 50;
        hupe_garm_params.max_iter = 100;
        hupe_garm_params.p_max = 0.1;
        hupe_garm_params.p_min = 0.01;
        params = &hupe_garm_params;
    } else if (strcmp(algo_id, "huim_aco") == 0) {
        huim_aco_params.min_utility = min_support;
        huim_aco_params.pop_size = 2000;
        huim_aco_params.max_iter = 25;
        huim_aco_params.alpha = 0.1;
        huim_aco_params.beta = 3.0;
        huim_aco_params.gamma = 5.0;
        huim_aco_params.lambda = 1000.0;
        huim_aco_params.tau = 0.8;
        params = &huim_aco_params;
    } else if (strcmp(algo_id, "huim_hc") == 0) {
        huim_hc_params.min_utility = min_support;
        huim_hc_params.pop_size = (argc >= 6) ? atoi(argv[5]) : 30;
        huim_hc_params.max_gen = (argc >= 7) ? atoi(argv[6]) : 10000;
        params = &huim_hc_params;
    } else if (strcmp(algo_id, "huim_sa") == 0) {
        huim_sa_params.min_utility = min_support;
        huim_sa_params.pop_size = (argc >= 6) ? atoi(argv[5]) : 30;
        huim_sa_params.temp = 100000.0;
        huim_sa_params.min_temp = 0.00001;
        huim_sa_params.alpha = 0.9993;
        params = &huim_sa_params;
    } else if (strcmp(algo_id, "huim_bpso_tree") == 0) {
        huim_bpso_tree_params.min_utility = min_support;
        huim_bpso_tree_params.pop_size = (argc >= 6) ? atoi(argv[5]) : 20;
        huim_bpso_tree_params.max_iter = (argc >= 7) ? atoi(argv[6]) : 1000;
        huim_bpso_tree_params.w = (argc >= 8) ? atof(argv[7]) : 0.9;
        huim_bpso_tree_params.c1 = (argc >= 9) ? atof(argv[8]) : 2.0;
        huim_bpso_tree_params.c2 = (argc >= 10) ? atof(argv[9]) : 2.0;
        params = &huim_bpso_tree_params;
    } else if (strcmp(algo_id, "bio_huif_ga") == 0 || strcmp(algo_id, "bio_huif_pso") == 0 || strcmp(algo_id, "bio_huif_ba") == 0) {
        bio_huif_params.min_utility = min_support;
        bio_huif_params.pop_size = (argc >= 6) ? atoi(argv[5]) : 30;
        bio_huif_params.max_iter = (argc >= 7) ? atoi(argv[6]) : 1000;
        params = &bio_huif_params;
    } else if (strcmp(algo_id, "huim_afsa") == 0) {
        afsa_params.min_utility = min_support;
        afsa_params.pop_size = (argc >= 6) ? atoi(argv[5]) : 30;
        afsa_params.max_iter = (argc >= 7) ? atoi(argv[6]) : 1000;
        afsa_params.visual = (argc >= 8) ? atoi(argv[7]) : 10;
        afsa_params.step = (argc >= 9) ? atoi(argv[8]) : 2;
        afsa_params.try_number = (argc >= 10) ? atoi(argv[9]) : 5;
        afsa_params.delta = (argc >= 11) ? atof(argv[10]) : 0.618;
        params = &afsa_params;
    } else if (strcmp(algo_id, "huim_abc") == 0) {
        abc_params.min_utility = min_support;
        abc_params.pop_size = (argc >= 6) ? atoi(argv[5]) : 30;
        abc_params.max_iter = (argc >= 7) ? atoi(argv[6]) : 1000;
        abc_params.limit = (argc >= 8) ? atoi(argv[7]) : 100;
        abc_params.step = (argc >= 9) ? atoi(argv[8]) : 2;
        params = &abc_params;
    } else if (strcmp(algo_id, "skyline_miner") == 0) {
        skyline_miner_params.min_utility = min_support;
        skyline_miner_params.min_support = (argc >= 6) ? atoi(argv[5]) : 0;
        params = &skyline_miner_params;
    } else if (strcmp(algo_id, "thui") == 0) {
        thui_params.k = (argc >= 5) ? atoi(argv[4]) : 10;
        params = &thui_params;
    } else if (strcmp(algo_id, "tku_ce") == 0) {
        tku_ce_params.k = (argc >= 5) ? atoi(argv[4]) : 10;
        tku_ce_params.n = (argc >= 6) ? atoi(argv[5]) : 2000;
        tku_ce_params.rho = (argc >= 7) ? atof(argv[6]) : 0.2;
        tku_ce_params.max_iter = (argc >= 8) ? atoi(argv[7]) : 2000;
        params = &tku_ce_params;
    } else if (strcmp(algo_id, "tku_ce_plus") == 0) {
        tku_ce_plus_params.k = (argc >= 5) ? atoi(argv[4]) : 10;
        tku_ce_plus_params.n = (argc >= 6) ? atoi(argv[5]) : 2000;
        tku_ce_plus_params.rho = (argc >= 7) ? atof(argv[6]) : 0.2;
        tku_ce_plus_params.max_iter = (argc >= 8) ? atoi(argv[7]) : 2000;
        params = &tku_ce_plus_params;
    } else if (strcmp(algo_id, "fhmds") == 0) {
        fhmds_params.k = (argc >= 5) ? atoi(argv[4]) : 10;
        fhmds_params.batch_size = (argc >= 6) ? atoi(argv[5]) : 1000;
        fhmds_params.window_size = (argc >= 7) ? atoi(argv[6]) : 5;
        params = &fhmds_params;
    } else if (strcmp(algo_id, "haui_miner") == 0) {
        haui_miner_params.min_utility_ratio = min_support; // use min_support flag as ratio
        params = &haui_miner_params;
    } else if (strcmp(algo_id, "ehaupm") == 0) {
        ehaupm_params.min_utility_ratio = min_support;
        params = &ehaupm_params;
    } else if (strcmp(algo_id, "hauim_gmu") == 0) {
        hauim_gmu_params.min_utility_ratio = min_support;
        params = &hauim_gmu_params;
    } else if (strcmp(algo_id, "nam_hep") == 0) {
        nam_hep_params.support_threshold = (argc >= 5) ? atof(argv[4]) : -1.0;
        nam_hep_params.occupancy_threshold = (argc >= 6) ? atof(argv[5]) : -1.0;
        params = &nam_hep_params;
    } else if (strcmp(algo_id, "mheinu") == 0) {
        mheinu_params.min_efficiency = min_support;
        mheinu_params.investment_path = (argc >= 6) ? argv[5] : NULL;
        params = &mheinu_params;
    } else if (strcmp(algo_id, "memu") == 0) {
        double glmau = (argc >= 5) ? atof(argv[4]) : 1000.0;
        memu_params.item_count = ds->max_id + 1;
        memu_params.mau_table = malloc(sizeof(double) * memu_params.item_count);
        for(uint32_t i=0; i < memu_params.item_count; i++) {
            memu_params.mau_table[i] = glmau;
        }
        params = &memu_params;
    } else if (strcmp(algo_id, "tmku") == 0) {
        tmku_params.k = (argc >= 5) ? (size_t)atoi(argv[4]) : 10;
        tmku_params.min_utility = (argc >= 6) ? atof(argv[5]) : 0.0;
        tmku_params.max_seconds = 0.0;
        const char *t_str = (argc >= 7) ? argv[6] : "";
        size_t commas = 0;
        for (size_t i = 0; t_str[i] != '\0'; i++) {
            if (t_str[i] == ',') commas++;
        }
        tmku_params.target_len = 0;
        tmku_params.target_pattern = NULL;
        if (strlen(t_str) > 0) {
            tmku_params.target_pattern = malloc(sizeof(uint32_t) * (commas + 1));
            char *str_copy = strdup(t_str);
            char *tok = strtok(str_copy, ",");
            while (tok != NULL) {
                tmku_params.target_pattern[tmku_params.target_len++] = (uint32_t)strtoul(tok, NULL, 10);
                tok = strtok(NULL, ",");
            }
            free(str_copy);
        }
        params = &tmku_params;
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
