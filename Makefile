CC = gcc
CFLAGS = -Iinclude -Iinclude/core -Wall -Wextra -O2
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Source files
SOURCES = $(SRC_DIR)/main.c \
          $(wildcard $(SRC_DIR)/core/*.c) \
          $(wildcard $(SRC_DIR)/algorithms/*.c)

# Convert .c paths to .o paths in obj/ directory
# Use substitution to handle subdirectories
OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SOURCES))
TARGET = $(BIN_DIR)/dm.exe

LDFLAGS =
ifeq ($(OS),Windows_NT)
    LDFLAGS += -lpsapi
endif

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@if not exist $(BIN_DIR) mkdir $(BIN_DIR)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@if not exist $(dir $@) powershell -Command "New-Item -ItemType Directory -Force -Path (Split-Path '$@')" > nul
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	@if exist $(OBJ_DIR) powershell -Command "Remove-Item -Recurse -Force $(OBJ_DIR)"
	@if exist $(BIN_DIR) powershell -Command "Remove-Item -Recurse -Force $(BIN_DIR)"

.PHONY: all clean
