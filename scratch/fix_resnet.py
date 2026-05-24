with open("include/models/vision/resnet.h", "r") as f: text = f.read()
text = text.replace(
    'int dm_resnet_basic_block(const DM_Block *in, DM_Block *out, int out_c, int stride, unsigned int seed);',
    '#include "core/dm_weight_cache.h"\nint dm_resnet_basic_block(DM_WeightCache *cache, const DM_Block *in, DM_Block *out, int out_c, int stride, unsigned int seed);'
)
with open("include/models/vision/resnet.h", "w") as f: f.write(text)

with open("src/models/vision/resnet.c", "r") as f: text = f.read()
text = text.replace('#include "models/vision/resnet.h"', '#include "models/vision/resnet.h"\n#include "core/dm_weight_cache.h"')
with open("src/models/vision/resnet.c", "w") as f: f.write(text)
