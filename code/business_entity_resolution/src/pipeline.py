"""
End-to-end pipeline runner for the Amazon ML Challenge 2026.

Usage (from the business_entity_resolution/ directory):
    python src/pipeline.py train      # train the matching model
    python src/pipeline.py predict    # generate test predictions
    python src/pipeline.py all        # train then predict

The pipeline produces two output files in student_resource/output/:
    matching_results.tsv   ← upload this to the leaderboard
    candidate_pairs.tsv    ← included in the submission zip
"""

import sys
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, SRC_DIR)


def run_train():
    logger.info("=" * 60)
    logger.info("PHASE 1: Training the matching model")
    logger.info("=" * 60)
    import train
    train.main()


def run_predict():
    logger.info("=" * 60)
    logger.info("PHASE 2: Generating test predictions")
    logger.info("=" * 60)
    import predict
    predict.main()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == 'train':
        run_train()
    elif cmd == 'predict':
        run_predict()
    elif cmd == 'all':
        run_train()
        run_predict()
    else:
        print(f"Unknown command '{cmd}'. Use: train | predict | all")
        sys.exit(1)

    logger.info("\n✓ Pipeline complete.")


if __name__ == '__main__':
    main()
