# Build the project-owned BoomerAMG interfaces against the shared HYPRE fork.

ROOT := $(abspath $(CURDIR)/../..)
HYPRE_PREFIX ?= $(ROOT)/hypre/install
HYPRE_INCLUDE := $(HYPRE_PREFIX)/include
HYPRE_LIB := $(HYPRE_PREFIX)/lib
BUILD_DIR := $(CURDIR)/build
MPICC ?= mpicc

CPPFLAGS += -I$(HYPRE_INCLUDE) -I$(CURDIR) -DHYPRE_TIMING -DHYPRE_FORTRAN
CFLAGS ?= -O2
CFLAGS += -fPIC
LDLIBS := -L$(HYPRE_LIB) -lHYPRE -lm

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
SHARED_EXT := dylib
SHARED_FLAG := -dynamiclib
SHARED_ID_FLAG := -Wl,-install_name,@rpath/libamg_runtime.dylib
HYPRE_RPATH := @loader_path/../install/lib
else
SHARED_EXT := so
SHARED_FLAG := -shared
SHARED_ID_FLAG :=
HYPRE_RPATH := $$ORIGIN/../install/lib
endif

RUNTIME_TARGET := libamg_runtime.$(SHARED_EXT)
COMMON_OBJECTS := $(BUILD_DIR)/amg_cycle.o $(BUILD_DIR)/amg_rl_shared.o

.PHONY: all clean print-vars

all: $(RUNTIME_TARGET)

$(RUNTIME_TARGET): $(BUILD_DIR)/amg_runtime.o $(COMMON_OBJECTS)
	$(MPICC) $(SHARED_FLAG) $(SHARED_ID_FLAG) -o $@ $^ $(LDLIBS) -Wl,-rpath,$(HYPRE_RPATH)

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(BUILD_DIR)
	$(MPICC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

clean:
	$(RM) -r $(BUILD_DIR) $(RUNTIME_TARGET)

print-vars:
	@echo "ROOT=$(ROOT)"
	@echo "HYPRE_PREFIX=$(HYPRE_PREFIX)"
	@echo "HYPRE_INCLUDE=$(HYPRE_INCLUDE)"
	@echo "HYPRE_LIB=$(HYPRE_LIB)"
	@echo "MPICC=$(MPICC)"
	@echo "HYPRE_RPATH=$(HYPRE_RPATH)"
