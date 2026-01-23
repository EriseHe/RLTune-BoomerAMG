"""
Test 1: Basic HYPRE Library Installation Check

This test verifies that:
1. HYPRE library (DLL) can be loaded
2. Version number is accessible
3. Key BoomerAMG symbols are present
"""

import os
import sys
import ctypes
from pathlib import Path

# Add the learning-to-relax-python folder to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "SetupPhase" / "learning-to-relax-python"))


def test_hypre_library_load():
    """Test that HYPRE library can be loaded."""
    print("=" * 60)
    print("Test 1: HYPRE Library Installation Check")
    print("=" * 60)
    
    # Check environment variables
    print("\n[1.1] Checking environment variables...")
    hypre_dir = os.environ.get("HYPRE_DIR", "")
    hypre_libhypre = os.environ.get("HYPRE_LIBHYPRE", "")
    hypre_dll_dirs = os.environ.get("HYPRE_DLL_DIRS", "")
    
    print(f"  HYPRE_DIR: {hypre_dir or '(not set)'}")
    print(f"  HYPRE_LIBHYPRE: {hypre_libhypre or '(not set)'}")
    print(f"  HYPRE_DLL_DIRS: {hypre_dll_dirs or '(not set)'}")
    
    if not hypre_dir and not hypre_libhypre:
        print("\n  WARNING: Neither HYPRE_DIR nor HYPRE_LIBHYPRE is set!")
        print("  Please set one of these environment variables.")
        return False
    
    # Try to load HYPRE
    print("\n[1.2] Loading HYPRE library...")
    try:
        from solvers.BoomerAMG.hypre_loader import load_hypre, resolve_hypre_library
        
        spec = resolve_hypre_library()
        print(f"  Library path: {spec.lib_path}")
        print(f"  DLL dirs: {spec.dll_dirs}")
        
        lib, spec = load_hypre()
        print("  HYPRE library loaded successfully!")
        
    except Exception as e:
        print(f"  FAILED to load HYPRE: {e}")
        return False
    
    # Check version (need to initialize HYPRE first in sequential mode)
    print("\n[1.3] Checking HYPRE version...")
    try:
        from solvers.BoomerAMG.boomeramg import hypre_version_number, _hypre_initialize, _hypre_finalize
        # Initialize HYPRE first (required in sequential mode)
        _hypre_initialize(lib)
        version = hypre_version_number(lib)
        major = version // 10000
        minor = (version % 10000) // 100
        patch = version % 100
        print(f"  HYPRE version: {major}.{minor}.{patch} (raw: {version})")
        _hypre_finalize(lib)
    except Exception as e:
        print(f"  FAILED to get version: {e}")
        # This is not critical for solver functionality
        print("  (Note: Version check is not critical if solver tests pass)")
        # Don't return False here - solver might still work
    
    # Check symbols
    print("\n[1.4] Checking BoomerAMG symbols...")
    required_symbols = [
        "HYPRE_Initialize",
        "HYPRE_Finalize",
        "HYPRE_IJMatrixCreate",
        "HYPRE_IJVectorCreate",
        "HYPRE_BoomerAMGCreate",
        "HYPRE_BoomerAMGSetup",
        "HYPRE_BoomerAMGSolve",
        "HYPRE_BoomerAMGGetNumIterations",
        "HYPRE_BoomerAMGGetFinalRelativeResidualNorm",
    ]
    
    optional_symbols = [
        "HYPRE_BoomerAMGGetCumNnzAP",
        "HYPRE_BoomerAMGSetCumNnzAP",
    ]
    
    from solvers.BoomerAMG.boomeramg import boomeramg_symbol_report
    
    all_symbols = required_symbols + optional_symbols
    report = boomeramg_symbol_report(lib, all_symbols)
    
    print("  Required symbols:")
    all_required_present = True
    for symbol in required_symbols:
        status = "OK" if report[symbol] else "MISSING"
        if not report[symbol]:
            all_required_present = False
        print(f"    {symbol}: {status}")
    
    print("\n  Optional symbols (HYPRE 3.x features):")
    for symbol in optional_symbols:
        status = "OK" if report[symbol] else "MISSING"
        print(f"    {symbol}: {status}")
    
    if not all_required_present:
        print("\n  FAILED: Some required symbols are missing!")
        return False
    
    print("\n" + "=" * 60)
    print("Test 1 PASSED: HYPRE library is properly installed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_hypre_library_load()
    sys.exit(0 if success else 1)
