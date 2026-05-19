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
EXPERIMENT_TARGET = $(BIN_DIR)/run_mfhoi_experiments
ITEMSET_BENCH_TARGET = $(BIN_DIR)/itemset_mining_bench
MFHOI_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/mfhoi_miner.c $(MFHOI_COMMON_SOURCES))
TMKU_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/tmku_miner.c $(SRC_DIR)/algorithms/tmku.c $(SRC_DIR)/core/dm_dataset.c $(SRC_DIR)/core/dm_registry.c $(SRC_DIR)/core/dm_benchmark.c)
MHOUI_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/mhoui_miner.c $(SRC_DIR)/algorithms/mhoui.c $(SRC_DIR)/core/dm_dataset.c $(SRC_DIR)/core/dm_registry.c $(SRC_DIR)/core/dm_benchmark.c)
VIFP_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/vifp_miner.c $(SRC_DIR)/algorithms/vifp.c $(SRC_DIR)/core/dm_dataset.c $(SRC_DIR)/core/dm_registry.c $(SRC_DIR)/core/dm_benchmark.c)
HUPP_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/hupp_miner.c $(SRC_DIR)/algorithms/hupp.c $(SRC_DIR)/core/dm_benchmark.c)
CHUO_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/chuo_miner.c $(SRC_DIR)/algorithms/chuo_miner.c $(SRC_DIR)/core/dm_dataset.c $(SRC_DIR)/core/dm_benchmark.c)
PSO_CLASSIFIER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/pso_classifier.c $(SRC_DIR)/algorithms/pso_classifier.c $(SRC_DIR)/core/dm_benchmark.c)
TKU_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/tku_miner.c $(SRC_DIR)/algorithms/tku_miner.c $(SRC_DIR)/core/dm_dataset.c $(SRC_DIR)/core/dm_benchmark.c)
KCLOTREE_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/kclotree_miner.c $(SRC_DIR)/algorithms/kclotree_miner.c $(SRC_DIR)/core/dm_benchmark.c)
TIPN_HOUI_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/tipn_houi_miner.c $(SRC_DIR)/algorithms/tipn_houi.c $(SRC_DIR)/core/dm_benchmark.c)
HTK_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/htk_miner.c $(SRC_DIR)/algorithms/htk_miner.c $(SRC_DIR)/core/dm_benchmark.c)
TOPKPHM_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/topkphm_miner.c $(SRC_DIR)/algorithms/topkphm.c $(SRC_DIR)/core/dm_benchmark.c)
TKU_PSO_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/tku_pso_miner.c $(SRC_DIR)/algorithms/tku_pso.c $(SRC_DIR)/core/dm_dataset.c $(SRC_DIR)/core/dm_benchmark.c)
EXPERIMENT_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/experiments/run_mfhoi_experiments.c $(MFHOI_COMMON_SOURCES))
ITEMSET_BENCH_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/itemset_mining_bench.c)

LDFLAGS = -lm -pthread
ifeq ($(OS),Windows_NT)
    LDFLAGS += -lpsapi
endif

all: $(TARGET) $(MFHOI_MINER_TARGET) $(MHOUI_MINER_TARGET) $(VIFP_MINER_TARGET) $(HUPP_MINER_TARGET) $(CHUO_MINER_TARGET) $(PSO_CLASSIFIER_TARGET) $(TKU_MINER_TARGET) $(KCLOTREE_MINER_TARGET) $(TIPN_HOUI_MINER_TARGET) $(HTK_MINER_TARGET) $(TOPKPHM_MINER_TARGET) $(TKU_PSO_MINER_TARGET) $(TMKU_MINER_TARGET) $(EXPERIMENT_TARGET) $(ITEMSET_BENCH_TARGET)

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

$(EXPERIMENT_TARGET): $(EXPERIMENT_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(EXPERIMENT_OBJECTS) -o $@ $(LDFLAGS)

$(ITEMSET_BENCH_TARGET): $(ITEMSET_BENCH_OBJECTS) $(TARGET)
	@mkdir -p $(BIN_DIR)
	$(CC) $(ITEMSET_BENCH_OBJECTS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

experiments: $(EXPERIMENT_TARGET)
	./$(EXPERIMENT_TARGET) --datasets datasets --out results --runs 3

plots:
	MPLCONFIGDIR=/tmp/mpl python3 scripts/plot_results.py results

report: experiments
	@echo "Report generated at results/reports/experiment_report.txt"

.PHONY: all clean experiments plots report
