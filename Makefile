CC = gcc
CFLAGS = -Iinclude -Iinclude/core -Wall -Wextra -O2 -pthread
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Source files
SOURCES = $(SRC_DIR)/main.c \
          $(wildcard $(SRC_DIR)/core/*.c) \
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
EXPERIMENT_TARGET = $(BIN_DIR)/run_mfhoi_experiments
ITEMSET_BENCH_TARGET = $(BIN_DIR)/itemset_mining_bench
MFHOI_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/mfhoi_miner.c $(MFHOI_COMMON_SOURCES))
MHOUI_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/mhoui_miner.c $(SRC_DIR)/algorithms/mhoui.c $(SRC_DIR)/core/dm_dataset.c $(SRC_DIR)/core/dm_registry.c $(SRC_DIR)/core/dm_benchmark.c)
VIFP_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/vifp_miner.c $(SRC_DIR)/algorithms/vifp.c $(SRC_DIR)/core/dm_dataset.c $(SRC_DIR)/core/dm_registry.c $(SRC_DIR)/core/dm_benchmark.c)
HUPP_MINER_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/hupp_miner.c $(SRC_DIR)/algorithms/hupp.c $(SRC_DIR)/core/dm_benchmark.c)
EXPERIMENT_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/experiments/run_mfhoi_experiments.c $(MFHOI_COMMON_SOURCES))
ITEMSET_BENCH_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_DIR)/tools/itemset_mining_bench.c)

LDFLAGS = -lm -pthread
ifeq ($(OS),Windows_NT)
    LDFLAGS += -lpsapi
endif

all: $(TARGET) $(MFHOI_MINER_TARGET) $(MHOUI_MINER_TARGET) $(VIFP_MINER_TARGET) $(HUPP_MINER_TARGET) $(EXPERIMENT_TARGET) $(ITEMSET_BENCH_TARGET)

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
