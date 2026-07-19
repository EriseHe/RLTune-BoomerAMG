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
HYPRE_RPATH := @loader_path/../install/lib
else
SHARED_EXT := so
SHARED_FLAG := -shared
HYPRE_RPATH := $$ORIGIN/../install/lib
endif

ENV_TARGET := libamg_env.$(SHARED_EXT)
SETUP_TARGET := libamg_setup_solver.$(SHARED_EXT)
COMMON_OBJECTS := $(BUILD_DIR)/amg_cycle.o $(BUILD_DIR)/amg_rl_shared.o

.PHONY: all clean clean-env clean-setup print-vars

all: $(ENV_TARGET) $(SETUP_TARGET)

$(ENV_TARGET): $(BUILD_DIR)/amg_env.o $(COMMON_OBJECTS)
	$(MPICC) $(SHARED_FLAG) -o $@ $^ $(LDLIBS) -Wl,-rpath,$(HYPRE_RPATH)

$(SETUP_TARGET): $(BUILD_DIR)/amg_setup_solver.o $(COMMON_OBJECTS)
	$(MPICC) $(SHARED_FLAG) -o $@ $^ $(LDLIBS) -Wl,-rpath,$(HYPRE_RPATH)

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(BUILD_DIR)
	$(MPICC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

clean-env:
	$(RM) $(ENV_TARGET) $(BUILD_DIR)/amg_env.o

clean-setup:
	$(RM) $(SETUP_TARGET) $(BUILD_DIR)/amg_setup_solver.o

clean:
	$(RM) -r $(BUILD_DIR) $(ENV_TARGET) $(SETUP_TARGET)

print-vars:
	@echo "ROOT=$(ROOT)"
	@echo "HYPRE_PREFIX=$(HYPRE_PREFIX)"
	@echo "HYPRE_INCLUDE=$(HYPRE_INCLUDE)"
	@echo "HYPRE_LIB=$(HYPRE_LIB)"
	@echo "MPICC=$(MPICC)"
	@echo "HYPRE_RPATH=$(HYPRE_RPATH)"
