CC = gcc
VULKAN_HEADERS_DIR = third_party/Vulkan-Hpp/Vulkan-Headers
VULKAN_CFLAGS = -I$(VULKAN_HEADERS_DIR)/include
CFLAGS = -D_POSIX_C_SOURCE=200809L -Iinclude -Iinclude/core $(VULKAN_CFLAGS) -Wall -Wextra -O2 -pthread -fPIC
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

GPU_SOURCES = $(SRC_DIR)/gpu/dm_gpu.c \
              $(SRC_DIR)/gpu/gpu_bpe.c \
              $(SRC_DIR)/gpu/gpu_sinkhorn.c \
              $(SRC_DIR)/gpu/gpu_unigram_em.c
RUNTIME_SOURCES = $(wildcard $(SRC_DIR)/runtime/*.c)
GENERATOR_SOURCES = $(wildcard $(SRC_DIR)/generator/*.c)
ENCODING_SOURCES = $(wildcard $(SRC_DIR)/encoding/*.c)
MODEL_SOURCES = $(wildcard $(SRC_DIR)/models/*.c) \
                $(wildcard $(SRC_DIR)/models/language/*.c) \
                $(wildcard $(SRC_DIR)/models/vision/*.c) \
                $(wildcard $(SRC_DIR)/models/multimodal/*.c)

# Source files
SOURCES = $(SRC_DIR)/main.c \
          $(wildcard $(SRC_DIR)/core/*.c) \
          $(SRC_DIR)/tokenizer/faro_tokenizer.c \
          $(SRC_DIR)/tokenizer/tokenizer_variants.c \
          $(SRC_DIR)/tokenizer/maximal_munch.c \
          $(SRC_DIR)/tokenizer/bpe_subword.c \
          $(SRC_DIR)/tokenizer/bpe_dropout.c \
          $(SRC_DIR)/tokenizer/fast_wordpiece.c \
          $(SRC_DIR)/tokenizer/grapheme_pair_encoding.c \
          $(SRC_DIR)/tokenizer/parity_bpe.c \
          $(SRC_DIR)/tokenizer/sentencepiece_lite.c \
          $(SRC_DIR)/tokenizer/tokenizer_lab.c \
          $(SRC_DIR)/tokenizer/unigram_subword.c \
          $(SRC_DIR)/tokenizer/volt.c \
          $(RUNTIME_SOURCES) \
          $(GENERATOR_SOURCES) \
          $(ENCODING_SOURCES) \
          $(MODEL_SOURCES) \
          $(GPU_SOURCES) \
          $(wildcard $(SRC_DIR)/algorithms/*.c)
MFHOI_COMMON_SOURCES = $(SRC_DIR)/algorithms/mfhoi_common.c \
                       $(SRC_DIR)/core/experiment.c \
                       $(SRC_DIR)/core/metrics.c \
                       $(SRC_DIR)/core/csv_writer.c \
                       $(SRC_DIR)/core/timer.c \
                       $(SRC_DIR)/core/memory_usage.c \
                       $(SRC_DIR)/core/dataset_stats.c \
                       $(SRC_DIR)/core/pattern_output.c

# Convert .c paths to .o paths in obj/ directory
# Use substitution to handle subdirectories
OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SOURCES))
TARGET = $(BIN_DIR)/dm.exe
MFHOI_MINER_TARGET = $(BIN_DIR)/mfhoi_miner
MHOUI_MINER_TARGET = $(BIN_DIR)/mhoui_miner
VIFP_MINER_TARGET = $(BIN_DIR)/vifp_miner
HUPP_MINER_TARGET = $(BIN_DIR)/hupp_miner
CHUO_MINER_TARGET = $(BIN_DIR)/chuo_miner
PSO_CLASSIFIER_TARGET = $(BIN_DIR)/pso_classifier
TKU_MINER_TARGET = $(BIN_DIR)/tku_miner
KCLOTREE_MINER_TARGET = $(BIN_DIR)/kclotree_miner
TIPN_HOUI_MINER_TARGET = $(BIN_DIR)/tipn_houi_miner
HTK_MINER_TARGET = $(BIN_DIR)/htk_miner
TOPKPHM_MINER_TARGET = $(BIN_DIR)/topkphm_miner
TKU_PSO_MINER_TARGET = $(BIN_DIR)/tku_pso_miner
TMKU_MINER_TARGET = $(BIN_DIR)/tmku_miner
ITEMSET_BENCH_TARGET = $(BIN_DIR)/itemset_mining_bench
HUST_TARGET = $(BIN_DIR)/hust_tokenize
BPE_TARGET = $(BIN_DIR)/dm_bpe
BPE_DROPOUT_TARGET = $(BIN_DIR)/dm_bpe_dropout
UNIGRAM_TARGET = $(BIN_DIR)/dm_unigram
SENTENCEPIECE_TARGET = $(BIN_DIR)/dm_sentencepiece
TOKENIZER_LAB_TARGET = $(BIN_DIR)/dm_tokenizer_lab
GPE_TARGET = $(BIN_DIR)/dm_gpe
PARITY_BPE_TARGET = $(BIN_DIR)/dm_parity_bpe
FAST_WORDPIECE_TARGET = $(BIN_DIR)/dm_fast_wordpiece
VOLT_TARGET = $(BIN_DIR)/dm_volt
TINYSTORIES_TARGET = $(BIN_DIR)/dm_tinystories
TINY_TRANSFORMER_TARGET = $(BIN_DIR)/dm_tiny_transformer
TEXTBOOK_GENERATOR_TARGET = $(BIN_DIR)/dm_textbook_generator
MOBILENET_TINY_TARGET = $(BIN_DIR)/dm_mobilenet_tiny
CONNECT_TARGET = $(BIN_DIR)/dm_connect
RUN_TARGET = $(BIN_DIR)/dm_run
MFHOI_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/mfhoi_miner.c $(MFHOI_COMMON_SOURCES))
TMKU_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/tmku_miner.c $(SRC_DIR)/algorithms/tmku.c $(SRC_DIR)/core/dm_dataset.c $(SRC_DIR)/core/dm_registry.c $(SRC_DIR)/core/dm_benchmark.c)
MHOUI_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/mhoui_miner.c $(SRC_DIR)/algorithms/mhoui.c $(SRC_DIR)/core/dm_dataset.c $(SRC_DIR)/core/dm_registry.c $(SRC_DIR)/core/dm_benchmark.c)
VIFP_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/vifp_miner.c $(SRC_DIR)/algorithms/vifp.c $(SRC_DIR)/core/dm_dataset.c $(SRC_DIR)/core/dm_registry.c $(SRC_DIR)/core/dm_benchmark.c)
HUPP_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/hupp_miner.c $(SRC_DIR)/algorithms/hupp.c $(SRC_DIR)/core/dm_benchmark.c)
CHUO_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/chuo_miner.c $(SRC_DIR)/algorithms/chuo_miner.c $(SRC_DIR)/core/dm_dataset.c $(SRC_DIR)/core/dm_benchmark.c)
PSO_CLASSIFIER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/pso_classifier.c $(SRC_DIR)/algorithms/pso_classifier.c $(SRC_DIR)/core/dm_benchmark.c)
TKU_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/tku_miner.c $(SRC_DIR)/algorithms/tku_miner.c $(SRC_DIR)/core/dm_dataset.c $(SRC_DIR)/core/dm_benchmark.c $(SRC_DIR)/core/dm_arena.c $(SRC_DIR)/algorithms/laga.c $(SRC_DIR)/tokenizer/faro_tokenizer.c $(SRC_DIR)/tokenizer/tokenizer_variants.c)
KCLOTREE_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/kclotree_miner.c $(SRC_DIR)/algorithms/kclotree_miner.c $(SRC_DIR)/core/dm_benchmark.c)
TIPN_HOUI_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/tipn_houi_miner.c $(SRC_DIR)/algorithms/tipn_houi.c $(SRC_DIR)/core/dm_benchmark.c)
HTK_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/htk_miner.c $(SRC_DIR)/algorithms/htk_miner.c $(SRC_DIR)/core/dm_benchmark.c)
TOPKPHM_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/topkphm_miner.c $(SRC_DIR)/algorithms/topkphm.c $(SRC_DIR)/core/dm_benchmark.c)
TKU_PSO_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/tku_pso_miner.c $(SRC_DIR)/algorithms/tku_pso.c $(SRC_DIR)/core/dm_dataset.c $(SRC_DIR)/core/dm_benchmark.c)
EXPERIMENT_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/experiments/run_mfhoi_experiments.c $(MFHOI_COMMON_SOURCES))
ITEMSET_BENCH_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/itemset_mining_bench.c)
CONNECT_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/dm_connect.c $(SRC_DIR)/core/dm_flat.c $(SRC_DIR)/core/dm_arena.c $(SRC_DIR)/core/dm_mmap.c $(SRC_DIR)/core/dm_benchmark.c $(SRC_DIR)/tokenizer/faro_tokenizer.c $(SRC_DIR)/tokenizer/tokenizer_variants.c)
RUN_SOURCES = $(SRC_DIR)/tools/dm_run.c \
              $(wildcard $(SRC_DIR)/core/*.c) \
              $(SRC_DIR)/tokenizer/faro_tokenizer.c \
              $(SRC_DIR)/tokenizer/tokenizer_variants.c \
              $(SRC_DIR)/tokenizer/maximal_munch.c \
              $(SRC_DIR)/tokenizer/bpe_subword.c \
              $(SRC_DIR)/tokenizer/bpe_dropout.c \
              $(SRC_DIR)/tokenizer/fast_wordpiece.c \
              $(SRC_DIR)/tokenizer/grapheme_pair_encoding.c \
              $(SRC_DIR)/tokenizer/parity_bpe.c \
              $(SRC_DIR)/tokenizer/sentencepiece_lite.c \
              $(SRC_DIR)/tokenizer/tokenizer_lab.c \
              $(SRC_DIR)/tokenizer/unigram_subword.c \
              $(SRC_DIR)/tokenizer/volt.c \
              $(RUNTIME_SOURCES) \
              $(GENERATOR_SOURCES) \
              $(ENCODING_SOURCES) \
              $(MODEL_SOURCES) \
              $(GPU_SOURCES) \
              $(wildcard $(SRC_DIR)/algorithms/*.c)
RUN_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(RUN_SOURCES))

SHADER_DIR = shaders/vulkan/tokenizer
SHADER_OUT_DIR = $(BIN_DIR)/shaders
SUBMODULE_GLSLANG = third_party/Vulkan-Hpp/glslang/build/StandAlone/glslang
SUBMODULE_GLSLANG_VALIDATOR = third_party/Vulkan-Hpp/glslang/build/StandAlone/glslangValidator
SHADER_SOURCES = $(SHADER_DIR)/bpe_pair_count.glsl \
                 $(SHADER_DIR)/sinkhorn_spmv.glsl \
                 $(SHADER_DIR)/sinkhorn_rowsum.glsl \
                 $(SHADER_DIR)/unigram_word.glsl
SPV_SHADERS = $(patsubst $(SHADER_DIR)/%.glsl,$(SHADER_OUT_DIR)/%.spv,$(SHADER_SOURCES))

# ── Shared library (libdm.so / libdm.dylib) ────────────────────────────────
# All sources from the main build minus main.c, plus the public API facade.
LIB_SOURCES = $(filter-out $(SRC_DIR)/main.c, $(SOURCES)) \
              $(SRC_DIR)/lib/dm_lib.c
LIB_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(LIB_SOURCES))
LIBDM_SO    = libdm.so
LIBDM_DYLIB = libdm.dylib

LDFLAGS = -lm -pthread
ICU_LDFLAGS = -licuuc
VULKAN_LDFLAGS =
ifneq ($(OS),Windows_NT)
    VULKAN_LDFLAGS += -ldl
endif
ifeq ($(OS),Windows_NT)
    LDFLAGS += -lpsapi
endif

all: $(TARGET) $(LIBDM_SO) $(MFHOI_MINER_TARGET) $(MHOUI_MINER_TARGET) $(VIFP_MINER_TARGET) $(HUPP_MINER_TARGET) $(CHUO_MINER_TARGET) $(PSO_CLASSIFIER_TARGET) $(TKU_MINER_TARGET) $(KCLOTREE_MINER_TARGET) $(TIPN_HOUI_MINER_TARGET) $(HTK_MINER_TARGET) $(TOPKPHM_MINER_TARGET) $(TKU_PSO_MINER_TARGET) $(TMKU_MINER_TARGET) $(ITEMSET_BENCH_TARGET) $(HUST_TARGET) $(BPE_TARGET) $(BPE_DROPOUT_TARGET) $(UNIGRAM_TARGET) $(SENTENCEPIECE_TARGET) $(TOKENIZER_LAB_TARGET) $(GPE_TARGET) $(PARITY_BPE_TARGET) $(FAST_WORDPIECE_TARGET) $(VOLT_TARGET) $(TINYSTORIES_TARGET) $(TINY_TRANSFORMER_TARGET) $(TEXTBOOK_GENERATOR_TARGET) $(MOBILENET_TINY_TARGET) $(CONNECT_TARGET) $(RUN_TARGET)

vulkan: $(TARGET) $(BPE_TARGET) $(UNIGRAM_TARGET) $(SENTENCEPIECE_TARGET) $(TOKENIZER_LAB_TARGET) $(GPE_TARGET) $(VOLT_TARGET) $(TINYSTORIES_TARGET) $(TINY_TRANSFORMER_TARGET) $(TEXTBOOK_GENERATOR_TARGET) $(MOBILENET_TINY_TARGET) shaders


# ─── Shared library targets ────────────────────────────────────────────────
$(LIBDM_SO): $(LIB_OBJECTS)
	$(CC) -shared -fPIC -DDM_BUILDING_LIB $(LIB_OBJECTS) \
	    -o $@ $(LDFLAGS) $(ICU_LDFLAGS) $(VULKAN_LDFLAGS) \
	    -L.venv/lib/python3.12/site-packages/tensorflow \
	    -ltensorflow_cc -ltensorflow_framework \
	    -Wl,-rpath,.venv/lib/python3.12/site-packages/tensorflow

$(LIBDM_DYLIB): $(LIB_OBJECTS)
	$(CC) -shared -DDM_BUILDING_LIB $(LIB_OBJECTS) \
	    -o $@ $(LDFLAGS) $(ICU_LDFLAGS) $(VULKAN_LDFLAGS)

# Pattern rule to compile the lib facade with -DDM_BUILDING_LIB
$(OBJ_DIR)/lib/dm_lib.o: $(SRC_DIR)/lib/dm_lib.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -DDM_BUILDING_LIB -c $< -o $@
# ─── Main executable ────────────────────────────────────────────────────────
$(TARGET): $(OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS) $(ICU_LDFLAGS) $(VULKAN_LDFLAGS) -L.venv/lib/python3.12/site-packages/tensorflow -ltensorflow_cc -ltensorflow_framework -Wl,-rpath,.venv/lib/python3.12/site-packages/tensorflow

$(MFHOI_MINER_TARGET): $(MFHOI_MINER_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(MFHOI_MINER_OBJECTS) -o $@ $(LDFLAGS)

$(MHOUI_MINER_TARGET): $(MHOUI_MINER_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(MHOUI_MINER_OBJECTS) -o $@ $(LDFLAGS)

$(VIFP_MINER_TARGET): $(VIFP_MINER_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(VIFP_MINER_OBJECTS) -o $@ $(LDFLAGS)

$(HUPP_MINER_TARGET): $(HUPP_MINER_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(HUPP_MINER_OBJECTS) -o $@ $(LDFLAGS)

$(CHUO_MINER_TARGET): $(CHUO_MINER_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CHUO_MINER_OBJECTS) -o $@ $(LDFLAGS)

$(PSO_CLASSIFIER_TARGET): $(PSO_CLASSIFIER_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(PSO_CLASSIFIER_OBJECTS) -o $@ $(LDFLAGS)

$(TKU_MINER_TARGET): $(TKU_MINER_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(TKU_MINER_OBJECTS) -o $@ $(LDFLAGS)

$(KCLOTREE_MINER_TARGET): $(KCLOTREE_MINER_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(KCLOTREE_MINER_OBJECTS) -o $@ $(LDFLAGS)

$(TIPN_HOUI_MINER_TARGET): $(TIPN_HOUI_MINER_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(TIPN_HOUI_MINER_OBJECTS) -o $@ $(LDFLAGS)

$(HTK_MINER_TARGET): $(HTK_MINER_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(HTK_MINER_OBJECTS) -o $@ $(LDFLAGS)

$(TOPKPHM_MINER_TARGET): $(TOPKPHM_MINER_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(TOPKPHM_MINER_OBJECTS) -o $@ $(LDFLAGS)

$(TKU_PSO_MINER_TARGET): $(TKU_PSO_MINER_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(TKU_PSO_MINER_OBJECTS) -o $@ $(LDFLAGS)

$(TMKU_MINER_TARGET): $(TMKU_MINER_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(TMKU_MINER_OBJECTS) -o $@ $(LDFLAGS)

$(ITEMSET_BENCH_TARGET): $(ITEMSET_BENCH_OBJECTS) $(TARGET)
	@mkdir -p $(BIN_DIR)
	$(CC) $(ITEMSET_BENCH_OBJECTS) -o $@ $(LDFLAGS)

$(CONNECT_TARGET): $(CONNECT_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CONNECT_OBJECTS) -o $@ $(LDFLAGS)

$(RUN_TARGET): $(RUN_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(RUN_OBJECTS) -o $@ $(LDFLAGS) $(ICU_LDFLAGS) $(VULKAN_LDFLAGS)

$(HUST_TARGET): $(SRC_DIR)/tokenizer/hust_tokenize.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BPE_TARGET): $(SRC_DIR)/tools/dm_bpe.c $(SRC_DIR)/tokenizer/bpe_subword.c include/tokenizer/bpe_subword.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(SRC_DIR)/tools/dm_bpe.c $(SRC_DIR)/tokenizer/bpe_subword.c $(GPU_SOURCES) -o $@ $(LDFLAGS) $(VULKAN_LDFLAGS)

$(BPE_DROPOUT_TARGET): $(SRC_DIR)/tools/dm_bpe_dropout.c $(SRC_DIR)/tokenizer/bpe_dropout.c include/tokenizer/bpe_dropout.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(SRC_DIR)/tools/dm_bpe_dropout.c $(SRC_DIR)/tokenizer/bpe_dropout.c -o $@ $(LDFLAGS)

$(UNIGRAM_TARGET): $(SRC_DIR)/tools/dm_unigram.c $(SRC_DIR)/tokenizer/unigram_subword.c include/tokenizer/unigram_subword.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(SRC_DIR)/tools/dm_unigram.c $(SRC_DIR)/tokenizer/unigram_subword.c $(GPU_SOURCES) -o $@ $(LDFLAGS) $(VULKAN_LDFLAGS)

$(SENTENCEPIECE_TARGET): $(SRC_DIR)/tools/dm_sentencepiece.c $(SRC_DIR)/tokenizer/sentencepiece_lite.c include/tokenizer/sentencepiece_lite.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(SRC_DIR)/tools/dm_sentencepiece.c $(SRC_DIR)/tokenizer/sentencepiece_lite.c $(GPU_SOURCES) -o $@ $(LDFLAGS) $(ICU_LDFLAGS) $(VULKAN_LDFLAGS)

$(TOKENIZER_LAB_TARGET): $(SRC_DIR)/tools/dm_tokenizer_lab.c $(SRC_DIR)/tokenizer/tokenizer_lab.c include/tokenizer/tokenizer_lab.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(SRC_DIR)/tools/dm_tokenizer_lab.c $(SRC_DIR)/tokenizer/tokenizer_lab.c $(GPU_SOURCES) -o $@ $(LDFLAGS) $(VULKAN_LDFLAGS)

$(GPE_TARGET): $(SRC_DIR)/tools/dm_gpe.c $(SRC_DIR)/tokenizer/grapheme_pair_encoding.c include/tokenizer/grapheme_pair_encoding.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(SRC_DIR)/tools/dm_gpe.c $(SRC_DIR)/tokenizer/grapheme_pair_encoding.c $(GPU_SOURCES) -o $@ $(LDFLAGS) $(VULKAN_LDFLAGS)

$(PARITY_BPE_TARGET): $(SRC_DIR)/tools/dm_parity_bpe.c $(SRC_DIR)/tokenizer/parity_bpe.c include/tokenizer/parity_bpe.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(SRC_DIR)/tools/dm_parity_bpe.c $(SRC_DIR)/tokenizer/parity_bpe.c -o $@ $(LDFLAGS)

$(FAST_WORDPIECE_TARGET): $(SRC_DIR)/tools/dm_fast_wordpiece.c $(SRC_DIR)/tokenizer/fast_wordpiece.c include/tokenizer/fast_wordpiece.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(SRC_DIR)/tools/dm_fast_wordpiece.c $(SRC_DIR)/tokenizer/fast_wordpiece.c -o $@ $(LDFLAGS)

$(VOLT_TARGET): $(SRC_DIR)/tools/dm_volt.c $(SRC_DIR)/tokenizer/volt.c include/tokenizer/volt.h include/gpu/dm_gpu.h $(GPU_SOURCES)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(SRC_DIR)/tools/dm_volt.c $(SRC_DIR)/tokenizer/volt.c $(GPU_SOURCES) -o $@ $(LDFLAGS) $(VULKAN_LDFLAGS)

$(TINYSTORIES_TARGET): $(SRC_DIR)/tools/dm_tinystories.c $(SRC_DIR)/models/tinystories.c include/models/tinystories.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(SRC_DIR)/tools/dm_tinystories.c $(SRC_DIR)/models/tinystories.c -o $@ $(LDFLAGS)

$(TINY_TRANSFORMER_TARGET): $(SRC_DIR)/tools/dm_tiny_transformer.c $(SRC_DIR)/models/language/tiny_transformer.c include/models/language/tiny_transformer.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(SRC_DIR)/tools/dm_tiny_transformer.c $(SRC_DIR)/models/language/tiny_transformer.c -o $@ $(LDFLAGS)

$(TEXTBOOK_GENERATOR_TARGET): $(SRC_DIR)/tools/dm_textbook_generator.c $(SRC_DIR)/generator/textbook_generator.c include/generator/textbook_generator.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(SRC_DIR)/tools/dm_textbook_generator.c $(SRC_DIR)/generator/textbook_generator.c -o $@ $(LDFLAGS)

$(MOBILENET_TINY_TARGET): $(SRC_DIR)/tools/dm_mobilenet_tiny.c $(SRC_DIR)/models/vision/mobilenet_tiny.c $(SRC_DIR)/models/tensor.c $(SRC_DIR)/encoding/image_patchify.c include/models/vision/mobilenet_tiny.h include/models/tensor.h include/encoding/image_patchify.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -Ithird_party/tensorflow -Ithird_party/tensorflow/third_party/xla -Ithird_party/tensorflow/third_party/xla/third_party/tsl $(SRC_DIR)/tools/dm_mobilenet_tiny.c $(SRC_DIR)/models/vision/mobilenet_tiny.c $(SRC_DIR)/models/tensor.c $(SRC_DIR)/encoding/image_patchify.c -o $@ $(LDFLAGS) -L.venv/lib/python3.12/site-packages/tensorflow -ltensorflow_cc -ltensorflow_framework -Wl,-rpath,.venv/lib/python3.12/site-packages/tensorflow

shaders: $(SPV_SHADERS)

$(SHADER_OUT_DIR)/%.spv: $(SHADER_DIR)/%.glsl
	@mkdir -p $(SHADER_OUT_DIR)
	@if command -v glslc >/dev/null 2>&1; then \
		glslc --target-env=vulkan1.1 -O $< -o $@; \
	elif command -v glslangValidator >/dev/null 2>&1; then \
		glslangValidator -V -S comp --target-env vulkan1.1 $< -o $@; \
	elif [ -x "$(SUBMODULE_GLSLANG_VALIDATOR)" ]; then \
		$(SUBMODULE_GLSLANG_VALIDATOR) -V -S comp --target-env vulkan1.1 $< -o $@; \
	elif [ -x "$(SUBMODULE_GLSLANG)" ]; then \
		$(SUBMODULE_GLSLANG) -V -S comp --target-env vulkan1.1 $< -o $@; \
	elif command -v glslang >/dev/null 2>&1; then \
		glslang -V -S comp --target-env vulkan1.1 $< -o $@; \
	else \
		echo "error: no shader compiler found. Install Vulkan SDK tools or run: cmake -S third_party/Vulkan-Hpp/glslang -B third_party/Vulkan-Hpp/glslang/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_OPT=OFF && cmake --build third_party/Vulkan-Hpp/glslang/build --target glslang-standalone" >&2; \
		exit 127; \
	fi

TF_IFLAGS = -Ithird_party/tensorflow \
            -Ithird_party/tensorflow/third_party/xla \
            -Ithird_party/tensorflow/third_party/xla/third_party/tsl

$(OBJ_DIR)/models/vision/mobilenet_tiny.o: $(SRC_DIR)/models/vision/mobilenet_tiny.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(TF_IFLAGS) -c $< -o $@

$(OBJ_DIR)/models/vision/tinyvit.o: $(SRC_DIR)/models/vision/tinyvit.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(TF_IFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@


libdm: $(LIBDM_SO)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(LIBDM_SO) $(LIBDM_DYLIB)

plots:
	MPLCONFIGDIR=/tmp/mpl python3 scripts/plot_results.py results

report: experiments
	@echo "Report generated at results/reports/experiment_report.txt"

.PHONY: all clean experiments plots report vulkan shaders libdm
