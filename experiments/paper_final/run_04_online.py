"""Run or validate the official Module 04 online suite."""

from experiments.runtime import configure_single_thread

if __name__ == "__main__":
    configure_single_thread()

from experiments.paper_final.online.suite import main

if __name__ == "__main__":
    main()
