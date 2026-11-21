"""Model resource and output paths for classic metric backends."""
from pathlib import Path
import os
FINAL_ENGINE=Path(__file__).resolve().parent
FINAL_ROOT=FINAL_ENGINE.parents[1]
FINAL_DATA=Path(os.environ.get('HANDWORLDBENCH_CLASSIC_DATA_ROOT',str(FINAL_ROOT/'runtime_data'/'classic'))).expanduser().resolve()
FINAL_RESULTS=Path(os.environ.get('HANDWORLDBENCH_CLASSIC_OUTPUT_ROOT',str(FINAL_ROOT/'results'/'classic'))).expanduser().resolve()
FINAL_SUITE=FINAL_ENGINE/'metric_suite'
