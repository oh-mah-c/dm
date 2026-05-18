#include "core/experiment.h"
#include <stdio.h>

void dm_dataset_stats_write(const char *csv_path, const char **paths, int count) {
    FILE *fp = fopen(csv_path, "w");
    if (!fp) return;
    fprintf(fp, "dataset,transactions,max_item_id,avg_transaction_length,max_transaction_length\n");
    for (int i = 0; i < count; i++) {
        TransactionDB *db = mfhoi_load_transaction_db(paths[i]);
        if (!db) continue;
        long total_len = 0;
        int max_len = 0;
        for (int t = 0; t < db->transaction_count; t++) {
            total_len += db->transaction_lengths[t];
            if (db->transaction_lengths[t] > max_len) max_len = db->transaction_lengths[t];
        }
        fprintf(fp, "%s,%d,%d,%.6f,%d\n", dm_path_basename(paths[i]), db->transaction_count, db->max_item_id,
                db->transaction_count ? (double)total_len / (double)db->transaction_count : 0.0, max_len);
        mfhoi_free_transaction_db(db);
    }
    fclose(fp);
}
