# Agentic tool-use harness.
#
# The engine objects must be built first in the repo root:
#   make bitnet                          (CPU-only)
#   make BN_ENABLE_METAL=1 bitnet        (Apple GPU)
#
# Then build here:
#   make
#   make BN_ENABLE_METAL=1

CC      ?= cc
CFLAGS  := -O3 -std=c11 -Wall -Wextra -I../include -I.
LDFLAGS := -lm

ifeq ($(BN_ENABLE_METAL),1)
  CFLAGS  += -DBN_ENABLE_METAL
  LDFLAGS += -framework Metal -framework Foundation -framework QuartzCore
endif

ENGINE_OBJS := $(filter-out ../src/main.o,$(wildcard ../src/*.o ../src/quant/*.o ../src/transformer/*.o))

test_agent: test_agent.c tools.c $(ENGINE_OBJS)
	$(CC) $(CFLAGS) -o $@ test_agent.c tools.c $(ENGINE_OBJS) $(LDFLAGS)

clean:
	rm -f test_agent

.PHONY: clean
