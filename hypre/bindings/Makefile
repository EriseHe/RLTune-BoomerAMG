ROOT := $(abspath $(CURDIR)/../..)
INTERFACE_DIR := $(ROOT)/hypre/interfaces
TARGET := $(INTERFACE_DIR)/libamg_setup_solver.dylib
MPICC ?= mpicc

.PHONY: all clean print-vars

all:
	$(MAKE) -C $(INTERFACE_DIR) libamg_setup_solver.dylib MPICC="$(MPICC)"

clean:
	$(MAKE) -C $(INTERFACE_DIR) clean-setup

print-vars:
	@echo "ROOT=$(ROOT)"
	@echo "INTERFACE_DIR=$(INTERFACE_DIR)"
	@echo "TARGET=$(TARGET)"
	@echo "MPICC=$(MPICC)"
