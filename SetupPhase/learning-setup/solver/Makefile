# Portable build for the setup-phase BoomerAMG solver wrapper.
#
# This compiles libamg_setup_solver.dylib using repository-relative paths so
# the resulting library does not depend on any machine-specific absolute path.

ROOT := $(abspath $(CURDIR)/../../..)
HYPRE_BUILD_DIR ?= $(ROOT)/SolvePhase/hypre/src/hypre
HYPRE_INCLUDE := $(HYPRE_BUILD_DIR)/include
HYPRE_LIB := $(HYPRE_BUILD_DIR)/lib

TARGET := libamg_setup_solver.dylib
MPICC ?= mpicc

COMMON_SRC := $(ROOT)/common/amg_rl_shared.c
CYCLE_SRC := $(ROOT)/SolvePhase/hypre/src/test/amg_cycle.c

# The setup solver lives at SetupPhase/learning-setup/solver, so the HYPRE
# library is three levels up under SolvePhase/hypre/src/hypre/lib.
RPATH := @loader_path/../../../SolvePhase/hypre/src/hypre/lib

.PHONY: all clean print-vars

all: $(TARGET)

$(TARGET): amg_setup_solver.c $(COMMON_SRC) $(CYCLE_SRC)
	$(MPICC) -shared -fPIC -O2 -o $@ \
		amg_setup_solver.c $(COMMON_SRC) $(CYCLE_SRC) \
		-I$(HYPRE_INCLUDE) \
		-L$(HYPRE_LIB) \
		-Wl,-rpath,$(RPATH) \
		-lHYPRE -lm

clean:
	rm -f $(TARGET)

print-vars:
	@echo "ROOT=$(ROOT)"
	@echo "HYPRE_BUILD_DIR=$(HYPRE_BUILD_DIR)"
	@echo "HYPRE_INCLUDE=$(HYPRE_INCLUDE)"
	@echo "HYPRE_LIB=$(HYPRE_LIB)"
	@echo "MPICC=$(MPICC)"
	@echo "RPATH=$(RPATH)"
