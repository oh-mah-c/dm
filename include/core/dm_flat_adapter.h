#ifndef DM_FLAT_ADAPTER_H
#define DM_FLAT_ADAPTER_H

#include "core/dm_dataset.h"
#include "core/dm_flat.h"

DM_Dataset *dm_flat_to_dataset(const DM_FlatDataset *flat, DM_DatasetType type);
DM_DatasetType dm_flat_choose_legacy_type(uint32_t supported_types, DM_ConnectorKind connector, int requested_type);

#endif
