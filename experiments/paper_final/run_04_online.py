"""Stage 4 entry point; reuse the existing formal-suite implementation."""

from experiments.runtime import configure_single_thread

if __name__ == "__main__":
    configure_single_thread()

from experiments.joint.solve_control.run_paper_final import main

if __name__ == "__main__":
    main()
