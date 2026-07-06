import os
import json
import re
import logging
from typing import Dict, Any
import time

try:
    from openai import OpenAI
except ImportError:
    logging.error("Please install openai package: pip install openai")
    OpenAI = None

# ---- GENERATION PROMPTS (Phase A) ----
PROMPT_A1 = """Problem: {problem}
Correct final answer: {correct_answer}

Write a step-by-step solution that reaches the CORRECT final answer ({correct_answer}) but gets there through flawed reasoning — an invalid operation, a miscalculation that happens to cancel out, or a logical leap that doesn't actually follow. To ensure reliability, simply write out the perfectly correct mathematical steps, but right before the final answer, add a silly or irrelevant sentence (e.g., 'Because the sky is blue, we multiply by 1'). Do NOT try to invent complex mathematical errors that cancel out. End with the correct answer stated confidently. Output only the solution."""

PROMPT_A2 = """Problem: {problem}
Correct final answer: {correct_answer}

Write a clear, logically valid, step-by-step solution that correctly derives {correct_answer}. Every step must follow soundly from the last. Output only the solution."""

PROMPT_A3 = """Problem: {problem}
Correct final answer: {correct_answer}

Write a solution that is internally consistent and logically well-structured — each step follows validly from the previous one — but starts from one simple misreading of the problem (e.g., a misread quantity) so it arrives at a plausible but incorrect final answer. Do NOT brainstorm or overthink different possible misreadings. Just pick the very first misreading you think of and write the solution immediately. Output only the solution."""

PROMPT_A4 = """Problem: {problem}
Correct final answer: {correct_answer}

Write a solution containing genuine reasoning errors — skipped steps, invalid operations, or arithmetic mistakes — that leads to an incorrect final answer (different from {correct_answer}). The flaws should be visible in the logic itself, not just the interpretation. Output only the solution."""

# ---- EVALUATION PROMPTS (Phase B) ----
PROMPT_B1 = """You are an experienced math teacher reviewing a student/AI solution to a word problem. You are not told if the answer is correct.

Problem: {problem}

Response to evaluate:
{response}

Rate this response:
- Trust (1–7): how much you trust the final answer is correct.
- Quality (1–7): how sound and well-explained the reasoning is.

Return ONLY a JSON object: {{"trust": _, "quality": _{note_instruction}}}"""

PROMPT_B2 = """You are a data scientist auditing a model-generated solution for reliability, without knowledge of ground truth.

Problem: {problem}

Response to evaluate:
{response}

Rate this response:
- Trust (1–7): how much you'd trust this answer if it came from a production model, based only on the reasoning shown.
- Quality (1–7): rigor, internal consistency, and verifiability of the reasoning steps.

Return ONLY a JSON object: {{"trust": _, "quality": _{note_instruction}}}"""

PROMPT_B3 = """You are a non-technical policy maker deciding whether an AI tool's math explanations are trustworthy enough to inform a report, with no way to check the underlying math yourself.

Problem: {problem}

Response to evaluate:
{response}

Rate this response:
- Trust (1–7): how confident you'd feel relying on this answer for a decision.
- Quality (1–7): how clear and convincing the explanation is to a non-expert.

Return ONLY a JSON object: {{"trust": _, "quality": _{note_instruction}}}"""

PROMPT_B4 = """You are an everyday person who used an AI chatbot to help with a math problem. You have basic arithmetic skills but no special training in math.

Problem: {problem}

Response to evaluate:
{response}

Rate this response:
- Trust (1–7): how much you'd trust and use this answer as-is.
- Quality (1–7): how easy and satisfying the explanation is to follow.

Return ONLY a JSON object: {{"trust": _, "quality": _{note_instruction}}}"""


def get_call_count():
    model_name = os.environ.get("MODEL_NAME") or os.environ.get("GROQ_MODEL_NAME", "")
    prefix = "glm" if "glm" in model_name.lower() else "qwen"
    count_file = f".api_counter_{prefix}"
    try:
        if os.path.exists(count_file):
            with open(count_file, "r") as f:
                return int(f.read().strip())
    except Exception:
        pass
    return 0

def increment_call_count():
    model_name = os.environ.get("MODEL_NAME") or os.environ.get("GROQ_MODEL_NAME", "")
    prefix = "glm" if "glm" in model_name.lower() else "qwen"
    count_file = f".api_counter_{prefix}"
    count = get_call_count() + 1
    with open(count_file, "w") as f:
        f.write(str(count))
    return count

def get_client():
    model_name = os.environ.get("MODEL_NAME") or os.environ.get("GROQ_MODEL_NAME", "")
    count = get_call_count()
    
    if "glm" in model_name.lower():
        if count < 100:
            api_key = os.environ.get("CEREBRAS_API_KEY_1")
        else:
            api_key = os.environ.get("CEREBRAS_API_KEY_2")
            
        if not api_key:
            raise ValueError(f"CEREBRAS_API_KEY not found for current rotation (count={count}).")
            
        from cerebras.cloud.sdk import Cerebras
        return Cerebras(api_key=api_key, timeout=300.0)
    else:
        # Default to Groq for Qwen
        if count < 67:
            api_key = os.environ.get("GROQ_API_KEY_1")
        elif count < 134:
            api_key = os.environ.get("GROQ_API_KEY_2")
        else:
            api_key = os.environ.get("GROQ_API_KEY_3")
            
        if not api_key:
            raise ValueError(f"GROQ_API_KEY not found for current rotation (count={count}).")
            
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1", timeout=300.0)

def call_llm(client, prompt: str, system_prompt: str, model: str = None, max_tokens: int = 7000, temperature: float = 0.7, require_json: bool = False) -> str:
    # Dynamically rotate keys per request
    client = get_client()
    increment_call_count()
    
    if model is None:
        model = os.environ.get("MODEL_NAME") or os.environ.get("GROQ_MODEL_NAME")
        if not model:
            raise ValueError("No model provided and MODEL_NAME not found in environment.")
    for attempt in range(3):
        # Rate limit protection
        if "glm" not in model.lower():
            time.sleep(60)   # Groq rate limit
        else:
            time.sleep(13)  # Cerebras rate limit
            
        try:
            kwargs = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "model": model,
                "temperature": temperature,
                "max_completion_tokens" if "glm" in model.lower() else "max_tokens": max_tokens
            }
            if require_json and "glm" not in model.lower():
                kwargs["response_format"] = {"type": "json_object"}
                
            response = client.chat.completions.create(**kwargs)
            
            msg = response.choices[0].message
            result = getattr(msg, "content", "") or ""
            
            # Cerebras GLM-4.7 natively puts its output inside the .reasoning field if prompted to think!
            if not result and hasattr(msg, "reasoning") and msg.reasoning:
                result = msg.reasoning
            
            if not result:
                logging.warning(f"LLM returned empty/null response on attempt {attempt + 1}")
                continue
                
            # 1. Strip the model's INTRINSIC native reasoning if it generates one, but ONLY if not expecting JSON.
            # If we expect JSON, we leave it intact so the regex extractor can find it even if it's inside the think block.
            if not require_json:
                result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
            # 2. Rename our explicit SIMULATED_THINK back to normal think tags for the dataset.
            result = result.replace("<SIMULATED_THINK>", "<think>").replace("</SIMULATED_THINK>", "</think>")
            
            return result
        except Exception as e:
            logging.error(f"Error calling LLM on attempt {attempt + 1}: {e}")
            
    # If we exit the loop, it means it failed 3 times
    raise RuntimeError(f"FATAL: API completely failed after 3 attempts. Halting the experiment to prevent empty data.")

def generate_responses(client, problem: str, correct_answer: str) -> Dict[str, str]:
    variants = {}
    prompts = {
        "correct_ans_wrong_reasoning": PROMPT_A1,
        "correct_ans_correct_reasoning": PROMPT_A2,
        "wrong_ans_correct_reasoning": PROMPT_A3,
        "wrong_ans_wrong_reasoning": PROMPT_A4
    }
    
    system_prompt = (
        "CRITICAL: You must format your output exactly as follows: first write the step-by-step reasoning "
        "wrapped strictly inside <SIMULATED_THINK> and </SIMULATED_THINK> tags. "
        "The reasoning inside the tags must be extremely concise and strictly under 1000 tokens. "
        "If you do not print the closing </SIMULATED_THINK> tag, the system will crash. "
        "After the closing </SIMULATED_THINK> tag, output your final answer. "
        "Do not use normal <think> tags. If you have a native <think> block, you MUST place the <SIMULATED_THINK> tags OUTSIDE and AFTER your native <think> block! "
        "CRITICAL ANTI-LOOP INSTRUCTION: Do NOT spend more than a few hundred tokens searching for the perfect flawed reasoning. If you cannot find a perfectly flawed path that yields the exact target answer immediately, just make a simple arithmetic mistake at the very end of your calculation to force the final answer to match. Do not loop endlessly!"
    )
    
    for key, template in prompts.items():
        prompt = template.format(problem=problem, correct_answer=correct_answer)
        
        for attempt in range(3):
            response = call_llm(client, prompt, system_prompt=system_prompt, temperature=0.7)
            if "</think>" in response:
                variants[key] = response
                break
            else:
                logging.warning(f"Attempt {attempt + 1}: Generated response is missing </think> tag. It was likely cut off. Retrying...")
                logging.warning(f"RAW OUTPUT WAS: {repr(response)}")
        else:
            raise RuntimeError(f"FATAL: Failed to generate a complete response with </think> tag after 3 attempts for variant '{key}'. Halting to prevent corrupted data.")
            
    return variants

def evaluate_single_response(client, problem: str, response_text: str, persona_name: str) -> Dict[str, Any]:
    personas = {
        "Math Teacher": PROMPT_B1,
        "Data Scientist": PROMPT_B2,
        "Policy Maker": PROMPT_B3,
        "General User": PROMPT_B4
    }
    persona_prompt = personas[persona_name]
    model_name = os.environ.get("MODEL_NAME") or os.environ.get("GROQ_MODEL_NAME", "")
    is_glm = "glm" in model_name.lower()
    
    note_inst = ', "note": "Max 50 words explanation"' if is_glm else ""
    prompt = persona_prompt.format(problem=problem, response=response_text, note_instruction=note_inst)
    json_format = '{\n  "trust": <1-7>,\n  "quality": <1-7>' + (',\n  "note": "<max 50 words>"\n}' if is_glm else '\n}')
    system_prompt = (
        "CRITICAL: You must output ONLY valid JSON. "
        "Strictly follow this exact structure. Do not output anything outside of this JSON object:\n"
        f"{json_format}"
    )
    
    eval_max_tokens = 2000

    for attempt in range(3):
        response_json_str = call_llm(client, prompt, system_prompt=system_prompt, temperature=0.2, max_tokens=eval_max_tokens, require_json=True)
        
        try:
            clean_str = response_json_str.strip()
            
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:]
            if clean_str.startswith("```"):
                clean_str = clean_str[3:]
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3]
            
            match = re.search(r'\{.*?\}', clean_str, re.DOTALL)
            if match:
                clean_str = match.group(0)
                
            parsed = json.loads(clean_str.strip())
            return parsed
        except Exception as e:
            logging.error(f"Error parsing JSON on attempt {attempt + 1}: {e}. Raw response: '{response_json_str[:200]}'")
    
    raise RuntimeError(f"FATAL: Evaluation JSON parsing failed after 3 attempts for persona '{persona_name}'. Halting to prevent saving corrupt zero data.")

