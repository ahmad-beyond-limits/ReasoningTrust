import os
import json
import random
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

# ---- PROMPTS ----

# Phase A (Generation)
PROMPT_A1 = """Problem: {problem}
Correct final answer: {correct_answer}

Write a step-by-step solution that reaches the CORRECT final answer ({correct_answer}) but gets there through flawed reasoning — an invalid operation, a miscalculation that happens to cancel out, or a logical leap that doesn't actually follow. The error should be genuine, not cosmetic, and must not be flagged or acknowledged. End with the correct answer stated confidently. Output only the solution."""

PROMPT_A2 = """Problem: {problem}
Correct final answer: {correct_answer}

Write a clear, logically valid, step-by-step solution that correctly derives {correct_answer}. Every step must follow soundly from the last. Output only the solution."""

PROMPT_A3 = """Problem: {problem}
Correct final answer: {correct_answer}

Write a solution that is internally consistent and logically well-structured — each step follows validly from the previous one — but starts from one subtle misreading of the problem (e.g., a misread quantity, unit, or condition) so it arrives at a plausible but incorrect final answer (different from {correct_answer}). The reasoning itself should read as coherent and careful; only the initial interpretation is off. Output only the solution."""

PROMPT_A4 = """Problem: {problem}
Correct final answer: {correct_answer}

Write a solution containing genuine reasoning errors — skipped steps, invalid operations, or arithmetic mistakes — that leads to an incorrect final answer (different from {correct_answer}). The flaws should be visible in the logic itself, not just the interpretation. Output only the solution."""

# Phase B (Judging - Single Response)
PROMPT_B1 = """You are an experienced math teacher reviewing student/AI solutions to a word problem. You are not told which, if any, answers are correct.

Problem: {problem}

Response to evaluate:
{response}

Rate this response:
- Trust (1–7): how much you trust the final answer is correct.
- Quality (1–7): how sound and well-explained the reasoning is.

Return ONLY a JSON object: {"trust": _, "quality": _, "note": "1 sentence explanation"}"""

PROMPT_B2 = """You are a data scientist auditing model-generated solutions for reliability, without knowledge of ground truth.

Problem: {problem}

Response to evaluate:
{response}

Rate this response:
- Trust (1–7): how much you'd trust this answer if it came from a production model, based only on the reasoning shown.
- Quality (1–7): rigor, internal consistency, and verifiability of the reasoning steps.

Return ONLY a JSON object: {"trust": _, "quality": _, "note": "1 sentence explanation"}"""

PROMPT_B3 = """You are a non-technical policy maker deciding whether an AI tool's math explanations are trustworthy enough to inform a report, with no way to check the underlying math yourself.

Problem: {problem}

Response to evaluate:
{response}

Rate this response:
- Trust (1–7): how confident you'd feel relying on this answer for a decision.
- Quality (1–7): how clear and convincing the explanation is to a non-expert.

Return ONLY a JSON object: {"trust": _, "quality": _, "note": "1 sentence explanation"}"""

PROMPT_B4 = """You are an everyday person who used an AI chatbot to help with a math problem. You have basic arithmetic skills but no special training in math.

Problem: {problem}

Response to evaluate:
{response}

Rate this response:
- Trust (1–7): how much you'd trust and use this answer as-is.
- Quality (1–7): how easy and satisfying the explanation is to follow.

Return ONLY a JSON object: {"trust": _, "quality": _, "note": "1 sentence explanation"}"""


# ---- INFERENCE FUNCTION ----

def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Warning: GROQ_API_KEY not found in environment variables.")
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

def call_llm(client, prompt: str, system_prompt: str = "You are a helpful assistant.", model: str = None, max_tokens: int = 1500, temperature: float = 0.7) -> str:
    if model is None:
        model = os.environ.get("GROQ_MODEL_NAME", "llama3-8b-8192")
    if client is None:
        return "Mock LLM Response"
        
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""


# ---- PIPELINE FUNCTIONS ----

def get_gsm8k_samples(num_samples=10, seed=42):
    print("Loading GSM8K dataset from Hugging Face...")
    dataset = load_dataset("openai/gsm8k", "main")
    test_data = dataset['test']
    random.seed(seed)
    indices = random.sample(range(len(test_data)), num_samples)
    return [test_data[i] for i in indices]

def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("Results", exist_ok=True)

def generate_responses(client, problem, correct_answer):
    variants = {}
    prompts = {
        "correct_ans_wrong_reasoning": PROMPT_A1,
        "correct_ans_correct_reasoning": PROMPT_A2,
        "wrong_ans_correct_reasoning": PROMPT_A3,
        "wrong_ans_wrong_reasoning": PROMPT_A4
    }
    
    for key, template in prompts.items():
        prompt = template.format(problem=problem, correct_answer=correct_answer)
        response = call_llm(client, prompt, system_prompt="You are a helpful AI generating synthetic data for a study. Output strictly what is requested without conversational filler.", temperature=0.7)
        variants[key] = response
    return variants

def evaluate_single_response(client, problem, response_text, persona_prompt):
    prompt = persona_prompt.format(
        problem=problem,
        response=response_text
    )
    response_json_str = call_llm(client, prompt, system_prompt="You must output ONLY valid JSON exactly as specified. No markdown wrapping unless strictly JSON inside.", temperature=0.2)
    
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

def evaluate_all_isolated(client, problem, variants):
    personas = {
        "Math Teacher": PROMPT_B1,
        "Data Scientist": PROMPT_B2,
        "Policy Maker": PROMPT_B3,
        "General User": PROMPT_B4
    }
    
    evaluations = {}
    
    # 4 Personas * 4 Variants = 16 completely isolated LLM calls per problem
    for persona_name, template in personas.items():
        evaluations[persona_name] = {}
        for variant_key, response_text in variants.items():
            result = evaluate_single_response(client, problem, response_text, template)
            evaluations[persona_name][variant_key] = result
            
    return evaluations

def run_pipeline():
    ensure_dirs()
    print("Starting the ReasoningTrust Pipeline...")
    
    client = get_client()
    
    # 1. Load Data (Exactly 10 problems)
    raw_file = os.path.join("data", "raw_problems.json")
    if os.path.exists(raw_file):
        with open(raw_file, "r", encoding="utf-8") as f:
            samples = json.load(f)
        print(f"Loaded {len(samples)} existing problems from {raw_file}")
    else:
        samples = get_gsm8k_samples(10)  # Changed to 10
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=4)
        print(f"Saved {len(samples)} raw problems to {raw_file}")
        
    # 2. Generation Step (10 problems * 4 variants = 40 independent LLM calls)
    processed_file = os.path.join("data", "generated_responses.json")
    processed_data = []
    if os.path.exists(processed_file):
        with open(processed_file, "r", encoding="utf-8") as f:
            processed_data = json.load(f)
    processed_ids = {item.get('problem_id') for item in processed_data}
    
    print("\n--- Phase 1: Generating Responses (40 isolated API calls total) ---")
    new_responses_generated = False
    for i, sample in enumerate(samples):
        problem_id = i + 1
        if problem_id in processed_ids:
            continue
            
        print(f"Generating variants for problem {problem_id}/{len(samples)}...")
        problem = sample['question']
        correct_answer_full = sample['answer']
        correct_answer = correct_answer_full.split("####")[-1].strip() if "####" in correct_answer_full else correct_answer_full
        
        variants = generate_responses(client, problem, correct_answer)
        
        sample_data = {
            "problem_id": problem_id,
            "problem": problem,
            "correct_answer": correct_answer,
            "responses": variants
        }
        processed_data.append(sample_data)
        new_responses_generated = True
        
        with open(processed_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=4)
            
    if not new_responses_generated:
        print("All 10 problems already have generated variants.")
        
    # 3. Evaluation Step (40 variants * 4 judges = 160 isolated LLM calls)
    results_file = os.path.join("Results", "evaluation_results.json")
    results_data = []
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            results_data = json.load(f)
    evaluated_ids = {item.get('problem_id') for item in results_data}
    
    print("\n--- Phase 2: Evaluating Responses (160 isolated API calls total) ---")
    for sample in processed_data:
        problem_id = sample['problem_id']
        if problem_id in evaluated_ids:
            continue
            
        print(f"Evaluating problem {problem_id}/{len(processed_data)}... (Making 16 separate API calls)")
        problem = sample['problem']
        variants = sample['responses']
        
        evaluations = evaluate_all_isolated(client, problem, variants)
        
        result_entry = {
            "problem_id": problem_id,
            "evaluations": evaluations
        }
        results_data.append(result_entry)
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=4)
            
    print(f"\nPipeline complete! Exactly 200 API calls made. All results saved to {results_file}")

if __name__ == "__main__":
    run_pipeline()
