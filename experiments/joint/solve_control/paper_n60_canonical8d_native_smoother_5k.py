"""Paper preset; execution and plotting belong to the current JSON entrypoint."""

from pathlib import Path
import sys

from run_joint_experiment import main


if __name__ == "__main__":
    config = Path(__file__).with_name("configs") / (
        "paper_n60_canonical8d_native_smoother_staged1000_5k.json"
    )
    main(["--config", str(config), *sys.argv[1:]])
