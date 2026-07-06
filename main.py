import os
import argparse
import data
import analysis
import logging
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("experiment.log"),
            logging.StreamHandler()
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    parser = argparse.ArgumentParser(description="ReasoningTrust Full Pipeline")
    parser.add_argument("--qwen", action="store_true", help="Run using Qwen on Groq")
    parser.add_argument("--glm", action="store_true", help="Run using GLM on Cerebras")
    args = parser.parse_args()
    
    if args.glm:
        os.environ["MODEL_NAME"] = os.environ.get("CEREBRAS_MODEL_NAME", "glm-4")
        logging.info("Configuration set to run CEREBRAS (GLM)")
    elif args.qwen:
        os.environ["MODEL_NAME"] = os.environ.get("GROQ_MODEL_NAME", "qwen-2.5")
        logging.info("Configuration set to run GROQ (Qwen)")
    else:
        logging.info("No specific model flag provided. Using default environment variables.")

    logging.info("========================================")
    logging.info("  PHASE 1: DATA AND AI GENERATION       ")
    logging.info("========================================")
    data.run_generation()

    logging.info("\n========================================")
    logging.info("  PHASE 2: AI EVALUATION                ")
    logging.info("========================================")
    data.run_evaluation()

    logging.info("\n========================================")
    logging.info("  PHASE 3: ANALYSIS                     ")
    logging.info("========================================")
    # Analysis pulls from results/evaluation_results.json 
    # and outputs to analysis/ directory
    analysis.analyze()

if __name__ == "__main__":
    main()
