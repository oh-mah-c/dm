#include "algorithms/medm_gen.h"
#include <math.h>
#include <time.h>

// Thread-safe and reproducible PRNG (Xorshift32)
static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    if (x == 0) x = 1; // 0 is a bad state for xorshift
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static double rand_double(uint32_t *state) {
    return (double)xorshift32(state) / (double)UINT32_MAX;
}

static int sample_poisson(double lambda, uint32_t *state) {
    double L = exp(-lambda);
    int k = 0;
    double p = 1.0;
    do {
        k++;
        p *= rand_double(state);
    } while (p > L && k < 1000);
    return k - 1;
}

// Helper to trim leading/trailing whitespace
static void trim(char *str) {
    char *end;
    // Trim leading
    char *start = str;
    while (*start == ' ' || *start == '\t' || *start == '\r' || *start == '\n') {
        start++;
    }
    if (start != str) {
        memmove(str, start, strlen(start) + 1);
    }
    // Trim trailing
    end = str + strlen(str) - 1;
    while (end >= str && (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n')) {
        *end = '\0';
        end--;
    }
}

DM_Spec* dm_spec_new(void) {
    DM_Spec *spec = (DM_Spec*)malloc(sizeof(DM_Spec));
    if (!spec) return NULL;
    spec->modality = MODALITY_TRANSACTIONAL;
    spec->size = 1000;
    spec->item_count = 100;
    spec->avg_len = 5.0;
    strcpy(spec->length_dist, "poisson");
    spec->noise_rate = 0.0;
    spec->epsilon = 0.0;
    spec->max_len = 100;
    spec->allow_duplicates = false;
    spec->support_error_threshold = 0.01;
    spec->max_iterations = 20;
    return spec;
}

void dm_spec_free(DM_Spec *spec) {
    free(spec);
}

DM_Ledger* dm_ledger_new(void) {
    DM_Ledger *ledger = (DM_Ledger*)malloc(sizeof(DM_Ledger));
    if (!ledger) return NULL;
    ledger->patterns = NULL;
    ledger->count = 0;
    ledger->capacity = 0;
    return ledger;
}

void dm_ledger_free(DM_Ledger *ledger) {
    if (ledger) {
        for (size_t i = 0; i < ledger->count; i++) {
            free(ledger->patterns[i].items);
        }
        free(ledger->patterns);
        free(ledger);
    }
}

int dm_spec_parse(const char *path, DM_Spec *spec, DM_Ledger *ledger) {
    FILE *file = fopen(path, "r");
    if (!file) return DM_ERROR_IO;

    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        trim(line);
        if (line[0] == '\0' || line[0] == '#' || line[0] == ';') {
            continue;
        }

        char key[128], val[512];
        if (sscanf(line, "%127[^=]= %511[^\n]", key, val) != 2) {
            continue;
        }
        trim(key);
        trim(val);

        if (strcmp(key, "modality") == 0) {
            if (strcmp(val, "transactional") == 0) spec->modality = MODALITY_TRANSACTIONAL;
            else if (strcmp(val, "utility") == 0) spec->modality = MODALITY_UTILITY;
            else if (strcmp(val, "sequence") == 0) spec->modality = MODALITY_SEQUENCE;
        } else if (strcmp(key, "size") == 0) {
            spec->size = (size_t)strtoul(val, NULL, 10);
        } else if (strcmp(key, "item_count") == 0) {
            spec->item_count = (uint32_t)strtoul(val, NULL, 10);
        } else if (strcmp(key, "avg_len") == 0) {
            spec->avg_len = atof(val);
        } else if (strcmp(key, "length_dist") == 0) {
            strncpy(spec->length_dist, val, sizeof(spec->length_dist) - 1);
        } else if (strcmp(key, "noise_rate") == 0) {
            spec->noise_rate = atof(val);
        } else if (strcmp(key, "epsilon") == 0) {
            spec->epsilon = atof(val);
        } else if (strcmp(key, "max_len") == 0) {
            spec->max_len = (size_t)strtoul(val, NULL, 10);
        } else if (strcmp(key, "allow_duplicates") == 0) {
            spec->allow_duplicates = (strcmp(val, "true") == 0 || strcmp(val, "1") == 0);
        } else if (strcmp(key, "support_error_threshold") == 0) {
            spec->support_error_threshold = atof(val);
        } else if (strcmp(key, "max_iterations") == 0) {
            spec->max_iterations = atoi(val);
        } else if (strcmp(key, "pattern") == 0) {
            // Parse pattern e.g., "10,20,30:0.15"
            char items_str[256], support_str[64];
            if (sscanf(val, "%255[^:]:%63s", items_str, support_str) == 2) {
                trim(items_str);
                trim(support_str);

                // Count items
                size_t item_count = 1;
                for (char *c = items_str; *c; c++) {
                    if (*c == ',') item_count++;
                }

                uint32_t *items = malloc(sizeof(uint32_t) * item_count);
                size_t idx = 0;
                char *tok = strtok(items_str, ",");
                while (tok) {
                    trim(tok);
                    items[idx++] = (uint32_t)strtoul(tok, NULL, 10);
                    tok = strtok(NULL, ",");
                }

                if (ledger->count >= ledger->capacity) {
                    ledger->capacity = ledger->capacity == 0 ? 8 : ledger->capacity * 2;
                    ledger->patterns = realloc(ledger->patterns, sizeof(DM_PlantedPattern) * ledger->capacity);
                }

                DM_PlantedPattern *p = &ledger->patterns[ledger->count++];
                p->items = items;
                p->length = item_count;
                p->target_support = atof(support_str);
                p->inject_prob = p->target_support; // Init projection to target
                p->measured_support = 0.0;
            }
        }
    }

    fclose(file);
    return DM_SUCCESS;
}

int dm_medm_compile(DM_Spec *spec, DM_Ledger *ledger, const char *real_dataset_path) {
    if (!real_dataset_path || strlen(real_dataset_path) == 0) {
        // No seed dataset: use spec properties directly
        return DM_SUCCESS;
    }

    // Load real dataset to calibrate specs (Algorithm 1 step 3)
    DM_DatasetType ds_type = DM_TYPE_TRANSACTIONAL;
    if (spec->modality == MODALITY_UTILITY) ds_type = DM_TYPE_UTILITY;
    else if (spec->modality == MODALITY_SEQUENCE) ds_type = DM_TYPE_SEQUENCE_UTILITY;

    DM_Dataset *ds = dm_dataset_load(real_dataset_path, ds_type);
    if (!ds) {
        printf("Warning: Could not load real seed dataset '%s' for compilation calibration. Proceeding with spec values.\n", real_dataset_path);
        return DM_SUCCESS;
    }

    printf("[Algorithm 1] Compiling specification using real seed dataset characteristics...\n");
    spec->size = ds->count;
    if (ds->max_id + 1 > spec->item_count) {
        spec->item_count = ds->max_id + 1;
    }

    double total_len = 0.0;
    if (spec->modality == MODALITY_TRANSACTIONAL) {
        DM_Trans_Simple *payload = (DM_Trans_Simple*)ds->payload;
        for (size_t i = 0; i < ds->count; i++) {
            total_len += payload[i].count;
        }
    } else if (spec->modality == MODALITY_UTILITY) {
        DM_Trans_Utility *payload = (DM_Trans_Utility*)ds->payload;
        for (size_t i = 0; i < ds->count; i++) {
            total_len += payload[i].count;
        }
    } else if (spec->modality == MODALITY_SEQUENCE) {
        DM_Sequence_Utility *payload = (DM_Sequence_Utility*)ds->payload;
        for (size_t i = 0; i < ds->count; i++) {
            size_t seq_items = 0;
            for (size_t j = 0; j < payload[i].count; j++) {
                seq_items += payload[i].itemsets[j].count;
            }
            total_len += seq_items;
        }
    }
    spec->avg_len = total_len / ds->count;
    printf("  Calibrated average length = %.2f, Max item ID = %u, Size = %zu\n", spec->avg_len, spec->item_count - 1, spec->size);

    dm_dataset_free(ds);
    return DM_SUCCESS;
}

// Generate an item from Zipf distribution
static uint32_t sample_zipf(uint32_t item_count, double s, double *zipf_cdf, uint32_t *state) {
    if (!zipf_cdf) {
        // Fallback to uniform if cdf not built
        return xorshift32(state) % item_count;
    }
    double u = rand_double(state);
    // Binary search on cdf
    uint32_t low = 0;
    uint32_t high = item_count - 1;
    while (low < high) {
        uint32_t mid = (low + high) / 2;
        if (zipf_cdf[mid] >= u) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return low;
}

// Helper to check if an item is already present in a simple transaction
static bool contains_item(uint32_t *items, size_t count, uint32_t item) {
    for (size_t i = 0; i < count; i++) {
        if (items[i] == item) return true;
    }
    return false;
}

// Helper to check if an item is already present in a utility transaction
static bool contains_item_util(DM_Item *items, size_t count, uint32_t item) {
    for (size_t i = 0; i < count; i++) {
        if (items[i].id == item) return true;
    }
    return false;
}

static int compare_uint32(const void *a, const void *b) {
    uint32_t val_a = *(const uint32_t*)a;
    uint32_t val_b = *(const uint32_t*)b;
    return (val_a > val_b) - (val_a < val_b);
}

static int compare_dm_item(const void *a, const void *b) {
    uint32_t val_a = ((const DM_Item*)a)->id;
    uint32_t val_b = ((const DM_Item*)b)->id;
    return (val_a > val_b) - (val_a < val_b);
}

// Memory freeing helpers
static void free_medm_trans(void *payload, size_t count) {
    DM_Trans_Simple *data = (DM_Trans_Simple*)payload;
    if (data) {
        for (size_t i = 0; i < count; i++) {
            free(data[i].items);
        }
        free(data);
    }
}

static void free_medm_util(void *payload, size_t count) {
    DM_Trans_Utility *data = (DM_Trans_Utility*)payload;
    if (data) {
        for (size_t i = 0; i < count; i++) {
            free(data[i].items);
        }
        free(data);
    }
}

static void free_medm_seq(void *payload, size_t count) {
    DM_Sequence_Utility *data = (DM_Sequence_Utility*)payload;
    if (data) {
        for (size_t i = 0; i < count; i++) {
            for (size_t j = 0; j < data[i].count; j++) {
                free(data[i].itemsets[j].items);
            }
            free(data[i].itemsets);
        }
        free(data);
    }
}

DM_Dataset* dm_medm_generate(const DM_Spec *spec, DM_Ledger *ledger, unsigned int seed) {
    uint32_t prng_state = seed;

    DM_Dataset *ds = (DM_Dataset*)malloc(sizeof(DM_Dataset));
    if (!ds) return NULL;

    ds->count = spec->size;
    ds->max_id = spec->item_count - 1;

    // Precalculate total pattern items length to avoid buffer overflow
    size_t total_p_len = 0;
    for (size_t k = 0; k < ledger->count; k++) {
        total_p_len += ledger->patterns[k].length;
    }

    // Precalculate Zipf CDF for background distribution (exponent s = 1.0)
    double *zipf_cdf = malloc(sizeof(double) * spec->item_count);
    double sum = 0.0;
    for (uint32_t i = 0; i < spec->item_count; i++) {
        sum += 1.0 / (double)(i + 1);
    }
    double accum = 0.0;
    for (uint32_t i = 0; i < spec->item_count; i++) {
        accum += (1.0 / (double)(i + 1)) / sum;
        zipf_cdf[i] = accum;
    }

    if (spec->modality == MODALITY_TRANSACTIONAL) {
        ds->type = DM_TYPE_TRANSACTIONAL;
        ds->free_payload = free_medm_trans;
        DM_Trans_Simple *payload = malloc(sizeof(DM_Trans_Simple) * spec->size);
        ds->payload = payload;

        for (size_t i = 0; i < spec->size; i++) {
            // Sample length
            int L = sample_poisson(spec->avg_len, &prng_state);
            if (L < 1) L = 1;
            if ((size_t)L > spec->max_len) L = spec->max_len;

            // Safe size to avoid buffer overflow
            size_t safe_size = L + total_p_len + 10;
            payload[i].items = malloc(sizeof(uint32_t) * safe_size);
            payload[i].count = 0;

            // Inject ledger patterns
            for (size_t p_idx = 0; p_idx < ledger->count; p_idx++) {
                DM_PlantedPattern *p = &ledger->patterns[p_idx];
                if (rand_double(&prng_state) < p->inject_prob) {
                    for (size_t item_idx = 0; item_idx < p->length; item_idx++) {
                        if (spec->allow_duplicates || !contains_item(payload[i].items, payload[i].count, p->items[item_idx])) {
                            payload[i].items[payload[i].count++] = p->items[item_idx];
                        }
                    }
                }
            }

            // Fill remaining with Zipf items
            while (payload[i].count < (size_t)L) {
                uint32_t item = sample_zipf(spec->item_count, 1.0, zipf_cdf, &prng_state);
                if (spec->allow_duplicates || !contains_item(payload[i].items, payload[i].count, item)) {
                    payload[i].items[payload[i].count++] = item;
                }
            }

            // Apply Differential Privacy / noise (Algorithm 2 step 2.e)
            if (spec->epsilon > 0.0 && spec->noise_rate > 0.0) {
                double eff_noise = spec->noise_rate * (1.0 + 1.0 / spec->epsilon);
                if (rand_double(&prng_state) < eff_noise) {
                    if (payload[i].count > 1 && rand_double(&prng_state) < 0.5) {
                        // delete
                        payload[i].count--;
                    } else {
                        // add
                        uint32_t item = sample_zipf(spec->item_count, 1.0, zipf_cdf, &prng_state);
                        if (spec->allow_duplicates || !contains_item(payload[i].items, payload[i].count, item)) {
                            payload[i].items[payload[i].count++] = item;
                        }
                    }
                }
            }

            // Sort items for standard transactional format
            qsort(payload[i].items, payload[i].count, sizeof(uint32_t), compare_uint32);
        }
    } else if (spec->modality == MODALITY_UTILITY) {
        ds->type = DM_TYPE_UTILITY;
        ds->free_payload = free_medm_util;
        DM_Trans_Utility *payload = malloc(sizeof(DM_Trans_Utility) * spec->size);
        ds->payload = payload;

        for (size_t i = 0; i < spec->size; i++) {
            int L = sample_poisson(spec->avg_len, &prng_state);
            if (L < 1) L = 1;
            if ((size_t)L > spec->max_len) L = spec->max_len;

            size_t safe_size = L + total_p_len + 10;
            payload[i].items = malloc(sizeof(DM_Item) * safe_size);
            payload[i].count = 0;

            // Inject ledger patterns
            for (size_t p_idx = 0; p_idx < ledger->count; p_idx++) {
                DM_PlantedPattern *p = &ledger->patterns[p_idx];
                if (rand_double(&prng_state) < p->inject_prob) {
                    for (size_t item_idx = 0; item_idx < p->length; item_idx++) {
                        if (spec->allow_duplicates || !contains_item_util(payload[i].items, payload[i].count, p->items[item_idx])) {
                            payload[i].items[payload[i].count].id = p->items[item_idx];
                            // Generate log-normal-ish utility (1 to 100)
                            payload[i].items[payload[i].count].utility = (double)(xorshift32(&prng_state) % 10 + 1) * (double)(xorshift32(&prng_state) % 5 + 1);
                            payload[i].count++;
                        }
                    }
                }
            }

            // Fill remaining with Zipf items
            while (payload[i].count < (size_t)L) {
                uint32_t item = sample_zipf(spec->item_count, 1.0, zipf_cdf, &prng_state);
                if (spec->allow_duplicates || !contains_item_util(payload[i].items, payload[i].count, item)) {
                    payload[i].items[payload[i].count].id = item;
                    payload[i].items[payload[i].count].utility = (double)(xorshift32(&prng_state) % 10 + 1) * (double)(xorshift32(&prng_state) % 5 + 1);
                    payload[i].count++;
                }
            }

            // Sort items by ID
            qsort(payload[i].items, payload[i].count, sizeof(DM_Item), compare_dm_item);

            // Compute total utility
            double total_u = 0.0;
            for (size_t j = 0; j < payload[i].count; j++) {
                total_u += payload[i].items[j].utility;
            }
            payload[i].total_utility = total_u;
        }
    } else {
        // MODALITY_SEQUENCE
        ds->type = DM_TYPE_SEQUENCE_UTILITY;
        ds->free_payload = free_medm_seq;
        DM_Sequence_Utility *payload = malloc(sizeof(DM_Sequence_Utility) * spec->size);
        ds->payload = payload;

        for (size_t i = 0; i < spec->size; i++) {
            // Sequence of itemsets. We assume 2-4 itemsets per sequence.
            size_t itemsets_count = (xorshift32(&prng_state) % 3) + 2; 
            payload[i].itemsets = malloc(sizeof(DM_Trans_Sequence_Utility) * itemsets_count);
            payload[i].count = itemsets_count;
            payload[i].total_utility = 0.0;
            payload[i].probability = 1.0;

            for (size_t s = 0; s < itemsets_count; s++) {
                int L = sample_poisson(spec->avg_len / itemsets_count, &prng_state);
                if (L < 1) L = 1;

                size_t safe_size = L + total_p_len + 10;
                payload[i].itemsets[s].items = malloc(sizeof(DM_Item) * safe_size);
                payload[i].itemsets[s].count = 0;

                // Inject ledger patterns to first itemset or randomly
                for (size_t p_idx = 0; p_idx < ledger->count; p_idx++) {
                    DM_PlantedPattern *p = &ledger->patterns[p_idx];
                    if (rand_double(&prng_state) < p->inject_prob) {
                        for (size_t item_idx = 0; item_idx < p->length; item_idx++) {
                            if (!contains_item_util(payload[i].itemsets[s].items, payload[i].itemsets[s].count, p->items[item_idx])) {
                                size_t c = payload[i].itemsets[s].count;
                                payload[i].itemsets[s].items[c].id = p->items[item_idx];
                                payload[i].itemsets[s].items[c].utility = (double)(xorshift32(&prng_state) % 5 + 1);
                                payload[i].itemsets[s].count++;
                            }
                        }
                    }
                }

                // Fill rest
                while (payload[i].itemsets[s].count < (size_t)L) {
                    uint32_t item = sample_zipf(spec->item_count, 1.0, zipf_cdf, &prng_state);
                    if (!contains_item_util(payload[i].itemsets[s].items, payload[i].itemsets[s].count, item)) {
                        size_t c = payload[i].itemsets[s].count;
                        payload[i].itemsets[s].items[c].id = item;
                        payload[i].itemsets[s].items[c].utility = (double)(xorshift32(&prng_state) % 5 + 1);
                        payload[i].itemsets[s].count++;
                    }
                }

                // Sort by ID
                qsort(payload[i].itemsets[s].items, payload[i].itemsets[s].count, sizeof(DM_Item), compare_dm_item);

                // Add to total sequence utility
                for (size_t j = 0; j < payload[i].itemsets[s].count; j++) {
                    payload[i].total_utility += payload[i].itemsets[s].items[j].utility;
                }
            }
        }
    }

    free(zipf_cdf);
    return ds;
}

int dm_medm_evaluate(const DM_Dataset *synthetic, const DM_Ledger *ledger, double *max_error) {
    *max_error = 0.0;

    for (size_t p_idx = 0; p_idx < ledger->count; p_idx++) {
        DM_PlantedPattern *p = &ledger->patterns[p_idx];
        size_t match_count = 0;

        if (synthetic->type == DM_TYPE_TRANSACTIONAL) {
            DM_Trans_Simple *payload = (DM_Trans_Simple*)synthetic->payload;
            for (size_t i = 0; i < synthetic->count; i++) {
                bool contains_all = true;
                for (size_t item_idx = 0; item_idx < p->length; item_idx++) {
                    if (!contains_item(payload[i].items, payload[i].count, p->items[item_idx])) {
                        contains_all = false;
                        break;
                    }
                }
                if (contains_all) match_count++;
            }
        } else if (synthetic->type == DM_TYPE_UTILITY) {
            DM_Trans_Utility *payload = (DM_Trans_Utility*)synthetic->payload;
            for (size_t i = 0; i < synthetic->count; i++) {
                bool contains_all = true;
                for (size_t item_idx = 0; item_idx < p->length; item_idx++) {
                    if (!contains_item_util(payload[i].items, payload[i].count, p->items[item_idx])) {
                        contains_all = false;
                        break;
                    }
                }
                if (contains_all) match_count++;
            }
        } else if (synthetic->type == DM_TYPE_SEQUENCE_UTILITY) {
            DM_Sequence_Utility *payload = (DM_Sequence_Utility*)synthetic->payload;
            for (size_t i = 0; i < synthetic->count; i++) {
                // For sequence, count if any itemset in the sequence contains the pattern
                bool contains_all = false;
                for (size_t s = 0; s < payload[i].count; s++) {
                    bool itemset_contains = true;
                    for (size_t item_idx = 0; item_idx < p->length; item_idx++) {
                        if (!contains_item_util(payload[i].itemsets[s].items, payload[i].itemsets[s].count, p->items[item_idx])) {
                            itemset_contains = false;
                            break;
                        }
                    }
                    if (itemset_contains) {
                        contains_all = true;
                        break;
                    }
                }
                if (contains_all) match_count++;
            }
        }

        p->measured_support = (double)match_count / (double)synthetic->count;
        double error = fabs(p->measured_support - p->target_support);
        if (error > *max_error) {
            *max_error = error;
        }
    }

    return DM_SUCCESS;
}

int dm_medm_repair_and_feedback(DM_Spec *spec, DM_Ledger *ledger, const char *output_base_path, const char *real_dataset_path, unsigned int seed) {
    // Step 1: Spec compilation
    int status = dm_medm_compile(spec, ledger, real_dataset_path);
    if (status != DM_SUCCESS) return status;

    DM_Dataset *synthetic = NULL;
    double max_error = 1.0;
    int iteration = 0;

    printf("[Algorithm 3] Starting constraint repair and closed-loop feedback correction...\n");
    
    while (iteration < spec->max_iterations) {
        if (synthetic) {
            dm_dataset_free(synthetic);
            synthetic = NULL;
        }

        printf("  Iteration %d:\n", iteration + 1);
        synthetic = dm_medm_generate(spec, ledger, seed + iteration);
        if (!synthetic) {
            return DM_ERROR_MEMORY;
        }

        // Evaluate actual support and get max deviation
        dm_medm_evaluate(synthetic, ledger, &max_error);
        
        printf("    Max Support Deviation = %.4f (threshold = %.4f)\n", max_error, spec->support_error_threshold);
        for (size_t p_idx = 0; p_idx < ledger->count; p_idx++) {
            DM_PlantedPattern *p = &ledger->patterns[p_idx];
            printf("      Pattern {");
            for (size_t item_idx = 0; item_idx < p->length; item_idx++) {
                printf("%u%s", p->items[item_idx], item_idx == p->length - 1 ? "" : ",");
            }
            printf("}: Target=%.3f, Actual=%.3f, Current Inject Prob=%.3f\n", 
                   p->target_support, p->measured_support, p->inject_prob);
        }

        if (max_error <= spec->support_error_threshold) {
            printf("  Converged within threshold at iteration %d!\n", iteration + 1);
            break;
        }

        // Closed-loop controller adjustment
        for (size_t p_idx = 0; p_idx < ledger->count; p_idx++) {
            DM_PlantedPattern *p = &ledger->patterns[p_idx];
            if (p->measured_support == 0.0) {
                // If support is completely zero, increase injection probability
                p->inject_prob = p->inject_prob * 1.5;
            } else {
                // Proportional feedback step
                p->inject_prob = p->inject_prob * (p->target_support / p->measured_support);
            }
            // Clamp injection probability to [0.0, 1.0]
            if (p->inject_prob > 1.0) p->inject_prob = 1.0;
            if (p->inject_prob < 0.0) p->inject_prob = 0.0;
        }

        iteration++;
    }

    if (iteration >= spec->max_iterations) {
        printf("  Reached max iterations (%d) without full convergence. Using final iteration output.\n", spec->max_iterations);
    }

    // Export output files
    char synthetic_file[1024];
    char ledger_file[1024];
    char report_file[1024];
    sprintf(synthetic_file, "%s.txt", output_base_path);
    sprintf(ledger_file, "%s_ledger.txt", output_base_path);
    sprintf(report_file, "%s_report.txt", output_base_path);

    // Save synthetic dataset
    FILE *f_syn = fopen(synthetic_file, "w");
    if (!f_syn) {
        dm_dataset_free(synthetic);
        return DM_ERROR_IO;
    }

    if (synthetic->type == DM_TYPE_TRANSACTIONAL) {
        DM_Trans_Simple *payload = (DM_Trans_Simple*)synthetic->payload;
        for (size_t i = 0; i < synthetic->count; i++) {
            for (size_t j = 0; j < payload[i].count; j++) {
                fprintf(f_syn, "%u%s", payload[i].items[j], j == payload[i].count - 1 ? "" : " ");
            }
            fprintf(f_syn, "\n");
        }
    } else if (synthetic->type == DM_TYPE_UTILITY) {
        DM_Trans_Utility *payload = (DM_Trans_Utility*)synthetic->payload;
        for (size_t i = 0; i < synthetic->count; i++) {
            // item1 item2 ... | total_utility | item1_util item2_util ...
            for (size_t j = 0; j < payload[i].count; j++) {
                fprintf(f_syn, "%u%s", payload[i].items[j].id, j == payload[i].count - 1 ? "" : " ");
            }
            fprintf(f_syn, ":%.0f:", payload[i].total_utility); // or HUI miner format: items : total_utility : utilities
            for (size_t j = 0; j < payload[i].count; j++) {
                fprintf(f_syn, "%.0f%s", payload[i].items[j].utility, j == payload[i].count - 1 ? "" : " ");
            }
            fprintf(f_syn, "\n");
        }
    } else if (synthetic->type == DM_TYPE_SEQUENCE_UTILITY) {
        DM_Sequence_Utility *payload = (DM_Sequence_Utility*)synthetic->payload;
        for (size_t i = 0; i < synthetic->count; i++) {
            // Sequence of itemsets format
            // e.g., itemset1 -1 itemset2 -1 ... -2
            for (size_t s = 0; s < payload[i].count; s++) {
                for (size_t j = 0; j < payload[i].itemsets[s].count; j++) {
                    fprintf(f_syn, "%u ", payload[i].itemsets[s].items[j].id);
                }
                fprintf(f_syn, "-1 ");
            }
            fprintf(f_syn, "-2\n");
        }
    }
    fclose(f_syn);

    // Save ledger
    FILE *f_led = fopen(ledger_file, "w");
    if (f_led) {
        fprintf(f_led, "# MeDM-Gen Ground-Truth Pattern Ledger\n");
        fprintf(f_led, "# Format: pattern_items : target_support : actual_support\n");
        for (size_t p_idx = 0; p_idx < ledger->count; p_idx++) {
            DM_PlantedPattern *p = &ledger->patterns[p_idx];
            for (size_t item_idx = 0; item_idx < p->length; item_idx++) {
                fprintf(f_led, "%u%s", p->items[item_idx], item_idx == p->length - 1 ? "" : ",");
            }
            fprintf(f_led, ":%.3f:%.3f\n", p->target_support, p->measured_support);
        }
        fclose(f_led);
    }

    // Save report
    FILE *f_rep = fopen(report_file, "w");
    if (f_rep) {
        fprintf(f_rep, "MeDM-Gen Benchmark Report\n");
        fprintf(f_rep, "=========================\n");
        fprintf(f_rep, "Modality: %s\n", spec->modality == MODALITY_TRANSACTIONAL ? "Transactional" : 
                                         (spec->modality == MODALITY_UTILITY ? "Utility" : "Sequence"));
        fprintf(f_rep, "Database Size: %zu\n", spec->size);
        fprintf(f_rep, "Average Transaction Length: %.2f\n", spec->avg_len);
        fprintf(f_rep, "Epsilon (DP): %.2f\n", spec->epsilon);
        fprintf(f_rep, "Iterations to Convergence: %d\n", iteration + 1);
        fprintf(f_rep, "Final Max Support Deviation: %.4f\n", max_error);
        fprintf(f_rep, "Target Error Threshold: %.4f\n", spec->support_error_threshold);
        fprintf(f_rep, "Status: %s\n", max_error <= spec->support_error_threshold ? "CONVERGED" : "FAILED_TO_CONVERGE");
        fclose(f_rep);
    }

    printf("[Algorithm 3] Successfully compiled, generated, evaluated, repaired, and exported dataset, ledger, and report files.\n");

    if (synthetic) {
        dm_dataset_free(synthetic);
    }

    return DM_SUCCESS;
}
