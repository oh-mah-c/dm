#include "core/dm_plugin.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DM_MAX_PLUGINS 256

static DM_Plugin *plugins[DM_MAX_PLUGINS];
static size_t plugin_count = 0;

int dm_plugin_register(DM_Plugin *plugin) {
    if (!plugin || !plugin->id || !plugin->run) return -1;
    for (size_t i = 0; i < plugin_count; i++) {
        if (strcmp(plugins[i]->id, plugin->id) == 0) return 0;
    }
    if (plugin_count >= DM_MAX_PLUGINS) return -1;
    plugins[plugin_count++] = plugin;
    return 0;
}

DM_Plugin *dm_plugin_get(const char *id) {
    if (!id) return NULL;
    for (size_t i = 0; i < plugin_count; i++) {
        if (strcmp(plugins[i]->id, id) == 0) return plugins[i];
    }
    return NULL;
}

void dm_plugin_print_all(void) {
    printf("Available dm plugins/adapters:\n");
    for (size_t i = 0; i < plugin_count; i++) {
        printf("  %s - %s\n", plugins[i]->id, plugins[i]->description ? plugins[i]->description : plugins[i]->name);
    }
}

const char *dm_plugin_arg(const DM_PluginInput *input, const char *key, const char *fallback) {
    if (!input || !key) return fallback;
    for (size_t i = 0; i < input->arg_count; i++) {
        if (input->args[i].key && strcmp(input->args[i].key, key) == 0) return input->args[i].value ? input->args[i].value : fallback;
    }
    return fallback;
}

long dm_plugin_arg_long(const DM_PluginInput *input, const char *key, long fallback) {
    const char *v = dm_plugin_arg(input, key, NULL);
    return v ? strtol(v, NULL, 10) : fallback;
}

double dm_plugin_arg_double(const DM_PluginInput *input, const char *key, double fallback) {
    const char *v = dm_plugin_arg(input, key, NULL);
    return v ? strtod(v, NULL) : fallback;
}
