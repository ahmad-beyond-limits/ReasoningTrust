import os
import json
import time
import argparse
from datasets import load_dataset
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Please install python-dotenv: pip install python-dotenv")

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai package: pip install openai")
    OpenAI = None

from data import (
    get_gsm8k_samples, ensure_dirs, PROMPT_A1, PROMPT_A2, PROMPT_A3, PROMPT_A4,
    PROMPT_B1, PROMPT_B2, PROMPT_B3, PROMPT_B4
)

class LLMRunner:
    def __init__(self, target_model: str):
        self.target_model = target_model # 'qwen' or 'glm'
        self.call_count = 0
        
        if self.target_model == 'qwen':
            # Groq setup
            self.model_name = os.environ.get("GROQ_MODEL_NAME", "llama3-8b-8192")
            self.wait_time = 5
            self.api_key_1 = os.environ.get("GROQ_API_KEY")
            if not self.api_key_1:
                print("Warning: GROQ_API_KEY not found in .env")
            self.base_url = "https://api.groq.com/openai/v1"
            self.client = OpenAI(api_key=self.api_key_1, base_url=self.base_url)
            
        elif self.target_model == 'glm':
            # Cerebras setup
            self.model_name = os.environ.get("CEREBRAS_MODEL_NAME", "llama3.1-8b")
            self.wait_time = 13
            self.api_key_1 = os.environ.get("CEREBRAS_API_KEY_1")
            self.api_key_2 = os.environ.get("CEREBRAS_API_KEY_2")
            if not self.api_key_1 or not self.api_key_2:
                print("Warning: CEREBRAS_API_KEY_1 or 2 not found in .env")
            self.base_url = "https://api.cerebras.ai/v1"
            
            # Start with key 1
            self.client = OpenAI(api_key=self.api_key_1, base_url=self.base_url)
        else:
            raise ValueError("Model must be 'qwen' or 'glm'")

    def call_llm(self, prompt: str, system_prompt: str = "You are a helpful assistant.", temperature: float = 0.7) -> str:
        # Check if we need to swap keys for Cerebras after 100 calls
        if self.target_model == 'glm' and self.call_count == 100:
            print(f"\n--- Reached 100 API calls! Switching to CEREBRAS_API_KEY_2 ---")
            self.client = OpenAI(api_key=self.api_key_2, base_url=self.base_url)
            
        self.call_count += 1
        print(f"[API Call {self.call_count}/200] Model: {self.target_model} | Sleep: {self.wait_time}s")
        
        if self.client is None:
            return "Mock Response"

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                max_tokens=1500,
                temperature=temperature
            )
            result = response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {e}")
            result = ""
            
        # Add the requested sleep time
        time.sleep(self.wait_time)
        return result

# ---- PIPELINE LOGIC ----

def generate_responses(runner, problem, correct_answer):
    variants = {}
    prompts = {
        "correct_ans_wrong_reasoning": PROMPT_A1,
        "correct_ans_correct_reasoning": PROMPT_A2,
        "wrong_ans_correct_reasoning": PROMPT_A3,
        "wrong_ans_wrong_reasoning": PROMPT_A4
    }
    
    for key, template in prompts.items():
        prompt = template.format(problem=problem, correct_answer=correct_answer)
        response = runner.call_llm(prompt, system_prompt="You are a helpful AI generating synthetic data for a study. Output strictly what is requested without conversational filler.", temperature=0.7)
        variants[key] = response
    return variants

def evaluate_single_response(runner, problem, response_text, persona_prompt):
    prompt = persona_prompt.format(
        problem=problem,
        response=response_text
    )
    response_json_str = runner.call_llm(prompt, system_prompt="You must output ONLY valid JSON exactly as specified. No markdown wrapping unless strictly JSON inside.", temperature=0.2)
    
    try:
        clean_str = response_json_str.strip()
        if clean_str.startswith("```json"):
            clean_str = clean_str[7:]
        if clean_str.startswith("```"):
            clean_str = clean_str[3:]
        if clean_str.endswith("```"):
            clean_str = clean_str[:-3]
        
        parsed = json.loads(clean_str.strip())
        return parsed
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {"raw_response": response_json_str, "error": str(e)}

def evaluate_all_isolated(runner, problem, variants):
    personas = {
        "Math Teacher": PROMPT_B1,
        "Data Scientist": PROMPT_B2,
        "Policy Maker": PROMPT_B3,
        "General User": PROMPT_B4
    }
    
    evaluations = {}
    for persona_name, template in personas.items():
        evaluations[persona_name] = {}
        for variant_key, response_text in variants.items():
            result = evaluate_single_response(runner, problem, response_text, template)
            evaluations[persona_name][variant_key] = result
            
    return evaluations


def run_pipeline(target_model):
    ensure_dirs()
    print(f"Starting the ReasoningTrust Pipeline with model type: {target_model}")
    
    runner = LLMRunner(target_model)
    
    # 1. Load Data
    raw_file = os.path.join("data", "raw_problems.json")
    if os.path.exists(raw_file):
        with open(raw_file, "r", encoding="utf-8") as f:
            samples = json.load(f)
        print(f"Loaded {len(samples)} existing problems from {raw_file}")
    else:
        samples = get_gsm8k_samples(10)
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=4)
        print(f"Saved {len(samples)} raw problems to {raw_file}")
        
    # 2. Generation Step
    processed_file = os.path.join("data", f"generated_responses_{target_model}.json")
    processed_data = []
    if os.path.exists(processed_file):
        with open(processed_file, "r", encoding="utf-8") as f:
            processed_data = json.load(f)
    processed_ids = {item.get('problem_id') for item in processed_data}
    
    print("\n--- Phase 1: Generating Responses (40 isolated API calls) ---")
    for i, sample in enumerate(samples):
        problem_id = i + 1
        if problem_id in processed_ids:
            continue
            
        print(f"Generating variants for problem {problem_id}/{len(samples)}...")
        problem = sample['question']
        correct_answer_full = sample['answer']
        correct_answer = correct_answer_full.split("####")[-1].strip() if "####" in correct_answer_full else correct_answer_full
        
        variants = generate_responses(runner, problem, correct_answer)
        
        sample_data = {
            "problem_id": problem_id,
            "problem": problem,
            "correct_answer": correct_answer,
            "responses": variants
        }
        processed_data.append(sample_data)
        
        with open(processed_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=4)
            
    # 3. Evaluation Step
    model_results_dir = os.path.join("Results", target_model)
    os.makedirs(model_results_dir, exist_ok=True)
    results_file = os.path.join(model_results_dir, "evaluation_results.json")
    results_data = []
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            results_data = json.load(f)
    evaluated_ids = {item.get('problem_id') for item in results_data}
    
    print("\n--- Phase 2: Evaluating Responses (160 isolated API calls) ---")
    for sample in processed_data:
        problem_id = sample['problem_id']
        if problem_id in evaluated_ids:
            continue
            
        print(f"Evaluating problem {problem_id}/{len(processed_data)}...")
        problem = sample['problem']
        variants = sample['responses']
        
        evaluations = evaluate_all_isolated(runner, problem, variants)
        
        result_entry = {
            "problem_id": problem_id,
            "evaluations": evaluations
        }
        results_data.append(result_entry)
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=4)
            
    print(f"\nPipeline complete! Total API calls made: {runner.call_count}")
    
    print("\n--- Phase 3: Running Statistical Analysis ---")
    try:
        from analysis import analyze
        analyze(target_model)
    except Exception as e:
        print(f"Analysis failed to run automatically: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ReasoningTrust pipeline.")
    parser.add_argument("model", choices=["qwen", "glm"], help="Specify 'qwen' for Groq or 'glm' for Cerebras.")
    args = parser.parse_args()
    
    run_pipeline(args.model)
