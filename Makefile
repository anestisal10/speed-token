CC = gcc
CFLAGS = -O3 -Wall -Wextra -Iinclude -Isrc
LDFLAGS = -shared

TARGET = libspeed_token.so
ifeq ($(OS),Windows_NT)
    TARGET = libspeed_token.dll
endif

SRCS = src/bpe.c src/loader.c src/pair_hashmap.c src/pretokenize.c src/pretokenize_avx2.c src/pretokenize_neon.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS)

src/pretokenize_avx2.o: src/pretokenize_avx2.c
	$(CC) $(CFLAGS) -mavx2 -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f src/*.o $(TARGET)
