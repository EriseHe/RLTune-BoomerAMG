# Out-of-source build for the unmodified HYPRE fork and project interfaces.

ROOT := $(abspath $(CURDIR)/..)
SOURCE_DIR := $(CURDIR)/source/src
BUILD_DIR := $(CURDIR)/build
INSTALL_DIR := $(CURDIR)/install
INTERFACE_DIR := $(CURDIR)/interfaces
MPICC ?= mpicc
MPICXX ?= mpicxx
JOBS ?= 4

CMAKE_CACHE := $(BUILD_DIR)/CMakeCache.txt

.PHONY: all hypre interfaces clean clean-interfaces distclean print-vars

all: hypre interfaces

$(CMAKE_CACHE): $(SOURCE_DIR)/CMakeLists.txt
	@mkdir -p $(BUILD_DIR) $(INSTALL_DIR)
	cmake -S $(SOURCE_DIR) -B $(BUILD_DIR) \
		-DCMAKE_INSTALL_PREFIX=$(INSTALL_DIR) \
		-DCMAKE_C_COMPILER="$(MPICC)" \
		-DCMAKE_CXX_COMPILER="$(MPICXX)" \
		-DBUILD_SHARED_LIBS=ON \
		-DHYPRE_ENABLE_MPI=ON \
		-DHYPRE_ENABLE_TIMING=ON \
		-DHYPRE_BUILD_EXAMPLES=OFF \
		-DHYPRE_BUILD_TESTS=OFF

hypre: $(CMAKE_CACHE)
	cmake --build $(BUILD_DIR) --parallel $(JOBS)
	cmake --install $(BUILD_DIR)

interfaces: hypre
	$(MAKE) -C $(INTERFACE_DIR) MPICC="$(MPICC)"

clean:
	@if [ -f $(CMAKE_CACHE) ]; then cmake --build $(BUILD_DIR) --target clean; fi

clean-interfaces:
	$(MAKE) -C $(INTERFACE_DIR) clean

distclean: clean-interfaces
	$(RM) -r $(BUILD_DIR) $(INSTALL_DIR)

print-vars:
	@echo "SOURCE_DIR=$(SOURCE_DIR)"
	@echo "BUILD_DIR=$(BUILD_DIR)"
	@echo "INSTALL_DIR=$(INSTALL_DIR)"
	@echo "INTERFACE_DIR=$(INTERFACE_DIR)"
	@echo "MPICC=$(MPICC)"
	@echo "MPICXX=$(MPICXX)"
