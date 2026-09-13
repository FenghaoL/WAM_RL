"""Same as ``prepare_clips.py --preset nohist``; kept for the commands in the docs."""
from scripts.so101_pipeline.ovcr_adapter.cylinder_experiments.prepare_clips import main

if __name__ == "__main__":
    main(["--preset", "nohist"])
