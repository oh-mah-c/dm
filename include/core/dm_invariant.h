#ifndef DM_INVARIANT_H
#define DM_INVARIANT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char name[64];
    char expression[256];
} DM_Invariant;

#ifdef __cplusplus
}
#endif

#endif // DM_INVARIANT_H
