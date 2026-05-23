#include "dm_role.h"

const char* dm_role_name(DM_Role role) {
    switch (role) {
        case DM_ROLE_UNKNOWN:        return "UNKNOWN";
        case DM_ROLE_INPUT:          return "INPUT";
        case DM_ROLE_LABEL:          return "LABEL";
        case DM_ROLE_FEATURE:        return "FEATURE";
        case DM_ROLE_TOKEN_IDS:      return "TOKEN_IDS";
        case DM_ROLE_ATTENTION_MASK: return "ATTENTION_MASK";
        case DM_ROLE_IMAGE:          return "IMAGE";
        case DM_ROLE_PATCH:          return "PATCH";
        case DM_ROLE_EMBEDDING:      return "EMBEDDING";
        case DM_ROLE_LOGITS:         return "LOGITS";
        case DM_ROLE_PARAMETER:      return "PARAMETER";
        case DM_ROLE_GRADIENT:       return "GRADIENT";
        case DM_ROLE_TRANSACTION:    return "TRANSACTION";
        case DM_ROLE_SEQUENCE:       return "SEQUENCE";
        case DM_ROLE_GRAPH:          return "GRAPH";
        case DM_ROLE_DISTRIBUTION:   return "DISTRIBUTION";
        default:                     return "UNKNOWN";
    }
}
