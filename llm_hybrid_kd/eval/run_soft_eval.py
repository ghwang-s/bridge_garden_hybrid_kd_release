"""Convenience wrapper for evaluating the soft-KD AlpacaEval output."""

import os
import sys

from alpacaeval_llm_judge import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--model_outputs",
                os.environ.get("MODEL_OUTPUTS", "output/soft/alpaca_eval_outputs.json"),
                "--batch_delay",
                os.environ.get("BATCH_DELAY", "0.3"),
            ]
        )
    main()
