#ifndef DM_ROLE_H
#define DM_ROLE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DM_ROLE_UNKNOWN,
    DM_ROLE_INPUT,
    DM_ROLE_LABEL,
    DM_ROLE_FEATURE,
    DM_ROLE_TOKEN_IDS,
    DM_ROLE_ATTENTION_MASK,
    DM_ROLE_IMAGE,
    DM_ROLE_PATCH,
    DM_ROLE_EMBEDDING,
    DM_ROLE_LOGITS,
    DM_ROLE_PARAMETER,
    DM_ROLE_GRADIENT,
    DM_ROLE_TRANSACTION,
    DM_ROLE_SEQUENCE,
    DM_ROLE_GRAPH,
    DM_ROLE_DISTRIBUTION
} DM_Role;

const char* dm_role_name(DM_Role role);

#ifdef __cplusplus
}
#endif

#endif // DM_ROLE_H
