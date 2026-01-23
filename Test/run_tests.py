"""
HYPRE/BoomerAMG Test Runner

This script runs all tests to verify that:
1. HYPRE library is properly installed
2. BoomerAMG solver is working correctly

Usage:
    python run_tests.py

Environment Variables:
    HYPRE_DIR or HYPRE_LIBHYPRE: Path to HYPRE installation
    HYPRE_DLL_DIRS: Additional DLL search directories (Windows)
"""

import os
import sys
from pathlib import Path

# Ensure the project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "SetupPhase" / "learning-to-relax-python"))


def print_header():
    """Print test suite header."""
    print()
    print("=" * 60)
    print("HYPRE/BoomerAMG Installation and Functionality Tests")
    print("=" * 60)
    print()
    print("Project Root:", PROJECT_ROOT)
    print("Python Version:", sys.version.split()[0])
    print()


def check_dependencies():
    """Check that required Python packages are installed."""
    print("Checking Python dependencies...")
    
    required = ["numpy", "scipy"]
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  {package}: OK")
        except ImportError:
            print(f"  {package}: MISSING")
            missing.append(package)
    
    if missing:
        print(f"\nERROR: Missing required packages: {', '.join(missing)}")
        print("Please install them with: pip install " + " ".join(missing))
        return False
    
    print()
    return True


def run_test_module(module_name: str, description: str) -> bool:
    """Run a test module and return True if passed."""
    print(f"\nRunning: {description}")
    print("-" * 60)
    
    try:
        # Import and run the test
        if module_name == "test_hypre_install":
            from test_hypre_install import test_hypre_library_load
            return test_hypre_library_load()
        elif module_name == "test_boomeramg_solver":
            from test_boomeramg_solver import test_boomeramg_solver
            return test_boomeramg_solver()
        else:
            print(f"Unknown test module: {module_name}")
            return False
            
    except Exception as e:
        print(f"\nFATAL ERROR in {module_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print_header()
    
    # Change to Test directory
    os.chdir(Path(__file__).parent)
    
    # Check dependencies first
    if not check_dependencies():
        return 1
    
    # Define tests to run
    tests = [
        ("test_hypre_install", "Test 1: HYPRE Library Installation"),
        ("test_boomeramg_solver", "Test 2: BoomerAMG Solver Functionality"),
    ]
    
    # Run all tests
    results = {}
    for module_name, description in tests:
        results[module_name] = run_test_module(module_name, description)
    
    # Print summary
    print("\n")
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for module_name, description in tests:
        status = "PASSED" if results[module_name] else "FAILED"
        if not results[module_name]:
            all_passed = False
        print(f"  {description}: {status}")
    
    print()
    if all_passed:
        print("ALL TESTS PASSED!")
        print()
        print("HYPRE BoomerAMG is properly installed and working.")
        print("You can now use it for your research.")
    else:
        print("SOME TESTS FAILED!")
        print()
        print("Please check the error messages above and ensure:")
        print("  1. HYPRE is built with BUILD_SHARED_LIBS=ON")
        print("  2. HYPRE_DIR or HYPRE_LIBHYPRE environment variable is set")
        print("  3. MS-MPI is installed and HYPRE_DLL_DIRS points to its bin folder")
    
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
