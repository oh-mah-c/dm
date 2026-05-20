#ifndef DM_MEDM_GEN_H
#define DM_MEDM_GEN_H

#include "core/dm_common.h"
#include "core/dm_dataset.h"

typedef enum {
    MODALITY_TRANSACTIONAL,
    MODALITY_UTILITY,
    MODALITY_SEQUENCE
} DM_Modality;

typedef struct {
    uint32_t *items;
    size_t length;
    
    // For sequence modality, we can plant sequential motifs:
    // We represent them as a sequence of itemsets (delimited by -1 in C) or simple sequence.
    // For simplicity, we also support itemset sequence format.
    
    double target_support;
    double inject_prob;
    double measured_support;
} DM_PlantedPattern;

typedef struct {
    DM_PlantedPattern *patterns;
    size_t count;
    size_t capacity;
} DM_Ledger;

typedef struct {
    DM_Modality modality;
    size_t size;             // N: Number of records
    uint32_t item_count;     // Max items
    double avg_len;          // Average length (lambda for Poisson)
    char length_dist[16];    // "poisson" or "normal"
    double noise_rate;       // Controlled noise
    double epsilon;          // Differential privacy epsilon (<= 0 to disable)
    
    // Constraints
    size_t max_len;
    bool allow_duplicates;
    
    // Evaluation/Feedback Thresholds
    double support_error_threshold;
    int max_iterations;
} DM_Spec;

// API functions
DM_Spec* dm_spec_new(void);
void dm_spec_free(DM_Spec *spec);
DM_Ledger* dm_ledger_new(void);
void dm_ledger_free(DM_Ledger *ledger);

int dm_spec_parse(const char *path, DM_Spec *spec, DM_Ledger *ledger);
int dm_medm_compile(DM_Spec *spec, DM_Ledger *ledger, const char *real_dataset_path);
DM_Dataset* dm_medm_generate(const DM_Spec *spec, DM_Ledger *ledger, unsigned int seed);
int dm_medm_repair_and_feedback(DM_Spec *spec, DM_Ledger *ledger, const char *output_base_path, const char *real_dataset_path, unsigned int seed);

#endif // DM_MEDM_GEN_H
