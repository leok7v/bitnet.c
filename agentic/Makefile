# Agentic tool-use harness.
#
# One step: `cd agentic && make`. Engine compilation is delegated to the root
# Makefile but directed into ./agentic/build (BUILD=...), so this build has its
# own object set that never shares or invalidates the root's ./build. The
# sub-make is incremental, and `engine` is phony, so test_agent can never link
# a stale engine -- no "build the engine in the root first" step.
#
# On macOS/arm64 Metal is enabled by default (opt out with BN_DISABLE_METAL=1);
# the precompiled shader library is embedded into the binary, so test_agent
# runs from any directory (no shaders/metal/ dependency).

CC      ?= cc
CFLAGS  := -O3 -std=c11 -Wall -Wextra -I../include -I.
LDFLAGS := -lm
BUILD   := $(CURDIR)/build

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
    METAL_LIB := $(BUILD)/bitnet.metallib
    METAL_EMBED_LDFLAGS := -Wl,-sectcreate,__TEXT,__metallib,$(METAL_LIB)
  endif
endif

# Default target. `engine` (phony) delegates engine compilation to the root
# Makefile into our own $(BUILD) -- incremental, and always run, so test_agent
# can never link a stale engine.
test_agent: test_agent.c tools.c engine
	$(CC) $(CFLAGS) -o $@ test_agent.c tools.c \
	    $$(ls $(BUILD)/src/*.o $(BUILD)/src/quant/*.o $(BUILD)/src/transformer/*.o \
	        | grep -v '/main\.o$$') \
	    $(LDFLAGS) $(METAL_EMBED_LDFLAGS)

engine:
	$(MAKE) -C .. BUILD=$(BUILD) objs

clean:
	rm -rf test_agent build

.PHONY: clean engine
