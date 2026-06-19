# Agentic tool-use harness.
#
# Build the engine objects in the repo root first, then build here:
#   make bitnet && (cd agentic && make)
#
# On macOS/arm64 Metal is enabled by default (opt out with BN_DISABLE_METAL=1);
# the precompiled shader library is embedded into the binary, so test_agent
# runs from any directory (no shaders/metal/ dependency).

CC      ?= cc
CFLAGS  := -O3 -std=c11 -Wall -Wextra -I../include -I.
LDFLAGS := -lm

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Darwin)
ifneq ($(filter arm% aarch%,$(UNAME_M)),)
ifndef BN_DISABLE_METAL
BN_ENABLE_METAL ?= 1
endif
endif
endif

ifeq ($(BN_ENABLE_METAL),1)
  CFLAGS  += -DBN_ENABLE_METAL
  LDFLAGS += -framework Metal -framework Foundation -framework QuartzCore
  METAL_TOOL := $(shell xcrun -sdk macosx -f metal 2>/dev/null)
  ifneq ($(METAL_TOOL),)
    METAL_LIB := ../bitnet.metallib
    METAL_EMBED_LDFLAGS := -Wl,-sectcreate,__TEXT,__metallib,$(METAL_LIB)
  endif
endif

ENGINE_OBJS := $(filter-out ../src/main.o,$(wildcard ../src/*.o ../src/quant/*.o ../src/transformer/*.o))

# The metallib is produced by the root Makefile; build it there on demand.
ifneq ($(METAL_LIB),)
$(METAL_LIB):
	$(MAKE) -C .. bitnet.metallib
endif

test_agent: test_agent.c tools.c $(ENGINE_OBJS) $(METAL_LIB)
	$(CC) $(CFLAGS) -o $@ test_agent.c tools.c $(ENGINE_OBJS) $(LDFLAGS) $(METAL_EMBED_LDFLAGS)

clean:
	rm -f test_agent

.PHONY: clean
