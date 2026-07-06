import os
import json
import random
from datasets import load_dataset
import agents
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Please install python-dotenv: pip install python-dotenv")

def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("analysis", exist_ok=True)

def get_gsm8k_samples(num_samples=10, seed=43):
    logging.info("Loading GSM8K dataset from Hugging Face...")
    dataset = load_dataset("openai/gsm8k", "main")
    test_data = dataset['test']
    random.seed(seed)
    indices = random.sample(range(len(test_data)), num_samples)
    return [test_data[i] for i in indices]

def get_model_prefix():
    model_name = os.environ.get("MODEL_NAME") or os.environ.get("GROQ_MODEL_NAME", "")
    return "glm" if "glm" in model_name.lower() else "qwen"

def run_generation():
    ensure_dirs()
    logging.info("Starting Phase 1: Data Generation")
    client = agents.get_client()
    
    # 1. Load Data
    raw_file = os.path.join("data", "raw_problems.json")
    if os.path.exists(raw_file):
        with open(raw_file, "r", encoding="utf-8") as f:
            samples = json.load(f)
        logging.info(f"Loaded {len(samples)} existing problems from {raw_file}")
    else:
        raw_samples = get_gsm8k_samples(10)
        samples = []
        for i, sample in enumerate(raw_samples):
            problem = sample['question']
            correct_answer_full = sample['answer']
            # Extract ONLY the exact numerical answer
            correct_answer = correct_answer_full.split("####")[-1].strip() if "####" in correct_answer_full else correct_answer_full
            samples.append({
                "problem_id": i + 1,
                "problem": problem,
                "answer": correct_answer
            })
            
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=4)
        logging.info(f"Saved {len(samples)} raw problems to {raw_file}")
        
    # 2. Generation Step
    model_prefix = get_model_prefix()
    processed_file = os.path.join("data", f"data_{model_prefix}.json")
    processed_data = []
    if os.path.exists(processed_file):
        with open(processed_file, "r", encoding="utf-8") as f:
            processed_data = json.load(f)
    processed_ids = {item.get('problem_id') for item in processed_data}
    
    new_responses_generated = False
    for sample in samples:
        problem_id = sample['problem_id']
        if problem_id in processed_ids:
            continue
            
        logging.info(f"Generating variants for problem {problem_id}/{len(samples)}...")
        problem = sample['problem']
        correct_answer = sample['answer']
        
        variants = agents.generate_responses(client, problem, correct_answer)
        
        sample_data = {
            "problem_id": problem_id,
            "problem": problem,
            "answer": correct_answer,
            "variants": variants
        }
        processed_data.append(sample_data)
        new_responses_generated = True
        
        with open(processed_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=4)
            
    if not new_responses_generated:
        logging.info("All 10 problems already have generated variants.")
    else:
        logging.info(f"Saved generated variants to {processed_file}")

def run_evaluation():
    ensure_dirs()
    logging.info("Starting Phase 2: AI Evaluation")
    client = agents.get_client()
    
    model_prefix = get_model_prefix()
    processed_file = os.path.join("data", f"data_{model_prefix}.json")
    if not os.path.exists(processed_file):
        logging.error(f"{processed_file} not found. Run generation first.")
        return
        
    with open(processed_file, "r", encoding="utf-8") as f:
        processed_data = json.load(f)
        
    results_file = os.path.join("results", f"data_{model_prefix}.json")
    results_data = []
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            results_data = json.load(f)
            
    # Create a set of already evaluated (problem_id, variant_key, persona_name)
    evaluated_keys = set()
    for item in results_data:
        key = f"{item['problem_id']}_{item['variant_type']}_{item['judge_persona']}"
        evaluated_keys.add(key)
    
    personas = ["Math Teacher", "Data Scientist", "Policy Maker", "General User"]
    
    total_evals = len(processed_data) * 4 * len(personas)
    eval_count = len(evaluated_keys)
    
    for sample in processed_data:
        problem_id = sample['problem_id']
        problem = sample['problem']
        variants = sample['variants']
        
        for variant_key, response_text in variants.items():
            for persona_name in personas:
                eval_key = f"{problem_id}_{variant_key}_{persona_name}"
                if eval_key in evaluated_keys:
                    continue
                
                eval_count += 1
                logging.info(f"Evaluating Problem {problem_id} | {variant_key} | {persona_name} ({eval_count}/{total_evals} done)...")
                
                result = agents.evaluate_single_response(client, problem, response_text, persona_name)
                
                # Flatten the structure as requested
                eval_entry = {
                    "problem_id": problem_id,
                    "judge_persona": persona_name,
                    "variant_type": variant_key,
                    "trust_rating": result.get("trust", 0),
                    "explanation_quality": result.get("quality", 0),
                    "note": result.get("note", "")
                }
                
                results_data.append(eval_entry)
                
                # Save continually in case of crash
                with open(results_file, "w", encoding="utf-8") as f:
                    json.dump(results_data, f, indent=4)
                    
    logging.info(f"Evaluation complete! Check {results_file} for all 160 evaluations.")
