CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c99 -I../../include
TARGET = bin/tokenizer_miner

SRCS = faro_tokenizer.c main.c
OBJS = $(SRCS:%.c=obj/%.o)

.PHONY: all clean run test

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $^

obj/%.o: %.c
	@mkdir -p obj
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -rf obj bin test_out.txt test_input.txt

run: all
	@echo "Running FARO-Tokenizer on Adware dataset..."
	./$(TARGET) -i ../../datasets/Malware_types_in_SPMF_Format/Adwaretranslated.txt -m doc --benchmark

test: all
	@echo "Creating simple test input..."
	@echo -e "FARO-Tokenizer is a zero-dependency high-performance tokenizer.\nUTF-8 test: Hello, 世界! This is a test contraction: don't. And hyphen-word: high-performance." > test_input.txt
	@echo "Running tokenizer in Document (Line) mode..."
	./$(TARGET) -i test_input.txt -m doc -o test_out.txt
	@echo "==================== TEST INPUT ===================="
	@cat test_input.txt
	@echo "================== TOKENIZED OUT =================="
	@cat test_out.txt
	@echo "===================================================="
	@echo "Testing completed successfully!"
