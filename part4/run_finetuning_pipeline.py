"""
Main script to run the complete fine-tuning pipeline.
"""

import subprocess
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_step(step_name, script_name):
    """Run a pipeline step"""
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP: {step_name}")
    logger.info(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        logger.info(f"✓ {step_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {step_name} failed with error: {e}")
        return False


def main():
    logger.info("Starting E5 Fine-tuning Pipeline for NBA Q&A")
    
    steps = [
        ("Create Training Data", "create_training_data.py"),
        ("Fine-tune E5 Model", "finetune_e5.py"),
        ("Evaluate Models", "evaluate_embeddings.py"),
    ]
    
    for step_name, script_name in steps:
        success = run_step(step_name, script_name)
        if not success:
            logger.error(f"Pipeline failed at step: {step_name}")
            sys.exit(1)
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info("\nGenerated files:")
    logger.info("  - training_data.json: Training dataset")
    logger.info("  - validation_data.json: Validation dataset")
    logger.info("  - finetuned_e5_st/: Fine-tuned model directory")
    logger.info("  - finetuned_e5_st/hyperparameters.json: Training configuration")
    logger.info("  - evaluation_results.json: Comparison results")


if __name__ == "__main__":
    main()