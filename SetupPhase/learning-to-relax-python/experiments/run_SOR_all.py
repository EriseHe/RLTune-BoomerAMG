"""
Helper script to run all experiments sequentially.
"""

import sys
import os
import time
import subprocess

def main():
    # List of scripts to run in a logical order
    # (Simpler/faster scripts first, then longer simulations)
    scripts = [
        'degenerate.py',    # Fast, validates basic solver behavior
        'asymptotic.py',    # Theoretical bounds
        'cg.py',            # CG bounds
        'comparators.py',   # Averaged performance
        'learning.py',      # Basic bandit learning (medium duration)
        'contextual.py',    # Contextual bandit learning (medium duration)
        'h2d.py'            # Full heat equation sim (longest, ~few mins)
    ]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Scripts are located in ../scripts
    scripts_dir = os.path.join(os.path.dirname(base_dir), 'scripts')
    plots_dir = os.path.join(scripts_dir, 'plots')
    
    # Ensure we can import from the parent directory
    sys.path.insert(0, os.path.dirname(base_dir))
    
    print(f"Starting execution of {len(scripts)} experiment scripts...")
    print(f"Scripts directory: {scripts_dir}")
    print(f"Plots will be saved to: {plots_dir}\n")
    
    total_start = time.time()
    
    for script in scripts:
        print(f"--------------------------------------------------")
        print(f"Running {script}...")
        print(f"--------------------------------------------------")
        
        start_time = time.time()
        script_path = os.path.join(scripts_dir, script)
        
        # Run script as a separate process
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=scripts_dir,  # Run in scripts dir so imports work
                check=True  # Raise exception on non-zero exit code
            )
            duration = time.time() - start_time
            print(f"✅ {script} completed in {duration:.2f} seconds.\n")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running {script}: Exit code {e.returncode}")
            print("Stopping sequence due to error.\n")
            sys.exit(1)
            
        except Exception as e:
            print(f"❌ Unexpected error running {script}: {e}")
            sys.exit(1)
            
    total_duration = time.time() - total_start
    print(f"==================================================")
    print(f"All experiments completed successfully!")
    print(f"Total time: {total_duration:.2f} seconds")
    print(f"==================================================")

if __name__ == '__main__':
    main()
