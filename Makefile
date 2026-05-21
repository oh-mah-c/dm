CC = gcc
CFLAGS = -Iinclude -Iinclude/core -Wall -Wextra -O2 -pthread
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Source files
SOURCES = $(SRC_DIR)/main.c \
          $(wildcard $(SRC_DIR)/core/*.c) \
          $(SRC_DIR)/tokenizer/faro_tokenizer.c \
          $(SRC_DIR)/tokenizer/tokenizer_variants.c \
          $(SRC_DIR)/tokenizer/maximal_munch.c \
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
              $(wildcard $(SRC_DIR)/algorithms/*.c)
RUN_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(RUN_SOURCES))

LDFLAGS = -lm -pthread
ifeq ($(OS),Windows_NT)
    LDFLAGS += -lpsapi
endif

all: $(TARGET) $(MFHOI_MINER_TARGET) $(MHOUI_MINER_TARGET) $(VIFP_MINER_TARGET) $(HUPP_MINER_TARGET) $(CHUO_MINER_TARGET) $(PSO_CLASSIFIER_TARGET) $(TKU_MINER_TARGET) $(KCLOTREE_MINER_TARGET) $(TIPN_HOUI_MINER_TARGET) $(HTK_MINER_TARGET) $(TOPKPHM_MINER_TARGET) $(TKU_PSO_MINER_TARGET) $(TMKU_MINER_TARGET) $(ITEMSET_BENCH_TARGET) $(HUST_TARGET) $(BPE_TARGET) $(BPE_DROPOUT_TARGET) $(UNIGRAM_TARGET) $(SENTENCEPIECE_TARGET) $(TOKENIZER_LAB_TARGET) $(GPE_TARGET) $(PARITY_BPE_TARGET) $(FAST_WORDPIECE_TARGET) $(CONNECT_TARGET) $(RUN_TARGET)


$(TARGET): $(OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

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
	$(CC) $(RUN_OBJECTS) -o $@ $(LDFLAGS)

$(HUST_TARGET): $(SRC_DIR)/tokenizer/hust_tokenize.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BPE_TARGET): scripts/tokenizer/dm_bpe.py src/tokenizer/nlp/bpe_subword.py
	@mkdir -p $(BIN_DIR)
	@printf '%s\n' '#!/usr/bin/env sh' 'ROOT=$$(CDPATH= cd -- "$$(dirname -- "$$0")/.." && pwd)' 'exec python3 "$$ROOT/scripts/tokenizer/dm_bpe.py" "$$@"' > $@
	@chmod +x $@

$(BPE_DROPOUT_TARGET): scripts/tokenizer/dm_bpe_dropout.py src/tokenizer/nlp/bpe_dropout.py src/tokenizer/nlp/bpe_subword.py
	@mkdir -p $(BIN_DIR)
	@printf '%s\n' '#!/usr/bin/env sh' 'ROOT=$$(CDPATH= cd -- "$$(dirname -- "$$0")/.." && pwd)' 'exec python3 "$$ROOT/scripts/tokenizer/dm_bpe_dropout.py" "$$@"' > $@
	@chmod +x $@

$(UNIGRAM_TARGET): scripts/tokenizer/dm_unigram.py src/tokenizer/nlp/unigram_subword.py
	@mkdir -p $(BIN_DIR)
	@printf '%s\n' '#!/usr/bin/env sh' 'ROOT=$$(CDPATH= cd -- "$$(dirname -- "$$0")/.." && pwd)' 'exec python3 "$$ROOT/scripts/tokenizer/dm_unigram.py" "$$@"' > $@
	@chmod +x $@

$(SENTENCEPIECE_TARGET): scripts/tokenizer/dm_sentencepiece.py src/tokenizer/nlp/sentencepiece_lite.py src/tokenizer/nlp/unigram_subword.py
	@mkdir -p $(BIN_DIR)
	@printf '%s\n' '#!/usr/bin/env sh' 'ROOT=$$(CDPATH= cd -- "$$(dirname -- "$$0")/.." && pwd)' 'exec python3 "$$ROOT/scripts/tokenizer/dm_sentencepiece.py" "$$@"' > $@
	@chmod +x $@

$(TOKENIZER_LAB_TARGET): scripts/tokenizer/dm_tokenizer_lab.py src/tokenizer/nlp/tokenizer_lab.py
	@mkdir -p $(BIN_DIR)
	@printf '%s\n' '#!/usr/bin/env sh' 'ROOT=$$(CDPATH= cd -- "$$(dirname -- "$$0")/.." && pwd)' 'exec python3 "$$ROOT/scripts/tokenizer/dm_tokenizer_lab.py" "$$@"' > $@
	@chmod +x $@

$(GPE_TARGET): scripts/tokenizer/dm_gpe.py src/tokenizer/nlp/grapheme_pair_encoding.py
	@mkdir -p $(BIN_DIR)
	@printf '%s\n' '#!/usr/bin/env sh' 'ROOT=$$(CDPATH= cd -- "$$(dirname -- "$$0")/.." && pwd)' 'exec python3 "$$ROOT/scripts/tokenizer/dm_gpe.py" "$$@"' > $@
	@chmod +x $@

$(PARITY_BPE_TARGET): scripts/tokenizer/dm_parity_bpe.py src/tokenizer/nlp/parity_bpe.py
	@mkdir -p $(BIN_DIR)
	@printf '%s\n' '#!/usr/bin/env sh' 'ROOT=$$(CDPATH= cd -- "$$(dirname -- "$$0")/.." && pwd)' 'exec python3 "$$ROOT/scripts/tokenizer/dm_parity_bpe.py" "$$@"' > $@
	@chmod +x $@

$(FAST_WORDPIECE_TARGET): scripts/tokenizer/dm_fast_wordpiece.py src/tokenizer/nlp/fast_wordpiece.py
	@mkdir -p $(BIN_DIR)
	@printf '%s\n' '#!/usr/bin/env sh' 'ROOT=$$(CDPATH= cd -- "$$(dirname -- "$$0")/.." && pwd)' 'exec python3 "$$ROOT/scripts/tokenizer/dm_fast_wordpiece.py" "$$@"' > $@
	@chmod +x $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@


clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

plots:
	MPLCONFIGDIR=/tmp/mpl python3 scripts/plot_results.py results

report: experiments
	@echo "Report generated at results/reports/experiment_report.txt"

.PHONY: all clean experiments plots report
