CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -std=c99
INCLUDES = -I.

TARGET = aptivra-cli
SRCS = multitier.c gguf_cli.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
