ROOT := $(abspath $(CURDIR)/../..)
INTERFACE_DIR := $(ROOT)/hypre/interfaces
TARGET := $(INTERFACE_DIR)/libamg_runtime.dylib
MPICC ?= mpicc

.PHONY: all clean print-vars

all:
	$(MAKE) -C $(INTERFACE_DIR) libamg_runtime.dylib MPICC="$(MPICC)"

clean:
	$(MAKE) -C $(INTERFACE_DIR) clean

print-vars:
	@echo "ROOT=$(ROOT)"
	@echo "INTERFACE_DIR=$(INTERFACE_DIR)"
	@echo "TARGET=$(TARGET)"
	@echo "MPICC=$(MPICC)"
