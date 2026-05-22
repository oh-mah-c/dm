#ifndef DM_PLUGIN_H
#define DM_PLUGIN_H

#include "core/dm_arena.h"
#include "core/dm_flat.h"

#include <stddef.h>

typedef struct {
    const char *key;
    const char *value;
} DM_PluginArg;

typedef struct {
    const DM_FlatDataset *flat;
    DM_Arena *arena;
    const char *source_path;
    DM_ConnectorKind connector;
    const DM_PluginArg *args;
    size_t arg_count;
} DM_PluginInput;

typedef struct {
    size_t patterns;
    size_t total_items;
    double best_score;
    double runtime_sec;
} DM_PluginResult;

typedef struct DM_Plugin {
    const char *id;
    const char *name;
    const char *description;
    unsigned accepts_connectors;
    int (*run)(const DM_PluginInput *input, DM_PluginResult *result);
} DM_Plugin;

#define DM_PLUGIN_ACCEPT_SPMF  (1u << DM_CONNECTOR_SPMF)
#define DM_PLUGIN_ACCEPT_TEXT  (1u << DM_CONNECTOR_TEXT)
#define DM_PLUGIN_ACCEPT_GRAPH (1u << DM_CONNECTOR_GRAPH)
#define DM_PLUGIN_ACCEPT_ALL   (DM_PLUGIN_ACCEPT_SPMF | DM_PLUGIN_ACCEPT_TEXT | DM_PLUGIN_ACCEPT_GRAPH)

int dm_plugin_register(DM_Plugin *plugin);
DM_Plugin *dm_plugin_get(const char *id);
void dm_plugin_print_all(void);
const char *dm_plugin_arg(const DM_PluginInput *input, const char *key, const char *fallback);
long dm_plugin_arg_long(const DM_PluginInput *input, const char *key, long fallback);
double dm_plugin_arg_double(const DM_PluginInput *input, const char *key, double fallback);

#if defined(__GNUC__) || defined(__clang__)
#define DM_REGISTER_PLUGIN(plugin_var) \
    __attribute__((constructor)) static void _dm_reg_plugin_##plugin_var(void) { \
        dm_plugin_register(&(plugin_var)); \
    }
#else
#define DM_REGISTER_PLUGIN(plugin_var)
#endif

#endif
