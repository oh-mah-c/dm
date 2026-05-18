#include "core/experiment.h"

void dm_pattern_output_write(const char *path, const PatternList *patterns, int transaction_count) {
    mfhoi_write_patterns(path, patterns, transaction_count);
}
