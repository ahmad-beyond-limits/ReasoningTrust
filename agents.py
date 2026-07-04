import os
import json
from typing import Dict, List, Any

# We will use the OpenAI compatible client, which works for both Groq and Cerebras
# by changing the base_url and api_key.
try:
    from openai import OpenAI
except ImportError:
    print("Please install openai package: pip install openai")
    OpenAI = None

def get_client(provider="groq"):
    """
    Returns an OpenAI compatible client for Groq or Cerebras based on the provider.
    """
    if provider == "groq":
        return OpenAI(
            api_key=os.environ.get("GROQ_API_KEY", "your-groq-api-key"),
            base_url="https://api.groq.com/openai/v1"
        )
    elif provider == "cerebras":
        return OpenAI(
            api_key=os.environ.get("CEREBRAS_API_KEY", "your-cerebras-api-key"),
            base_url="https://api.cerebras.ai/v1"
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

def call_llm(client, prompt: str, system_prompt: str = "You are a helpful assistant.", model: str = "llama3-8b-8192", max_tokens: int = 1024, temperature: float = 0.7) -> str:
    """
    Generic LLM caller.
    """
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

def generate_response_variants(client, problem: str, correct_answer: str, model: str) -> Dict[str, str]:
    """
    Step 2: Generate 4 LLM responses for each problem with STRONG PROMPTING.
    """
    prompts = {
        "correct_ans_wrong_reasoning": f"Problem: {problem}\nCorrect Answer: {correct_answer}\n\nCRITICAL INSTRUCTION: You MUST generate a response that arrives at the exact Correct Answer ({correct_answer}), BUT the reasoning steps leading to it MUST be completely flawed, illogical, or use the wrong mathematical formulas. Do not reveal that you are making mistakes on purpose. Act confidently like a student who is guessing but magically arrives at the correct final number.",
        "correct_ans_correct_reasoning": f"Problem: {problem}\nCorrect Answer: {correct_answer}\n\nCRITICAL INSTRUCTION: You MUST generate a response that has a clear, step-by-step, flawlessly logical reasoning chain that correctly leads to the Correct Answer ({correct_answer}). Be highly coherent and mathematically rigorous.",
        "wrong_ans_correct_reasoning": f"Problem: {problem}\nCorrect Answer: {correct_answer}\n\nCRITICAL INSTRUCTION: You MUST generate a response where the reasoning steps are highly logical, plausible, and mathematically sound for the most part, BUT you MUST make a subtle but fatal calculation error near the end, leading to a completely WRONG final answer (do NOT output {correct_answer}). Act confident.",
        "wrong_ans_wrong_reasoning": f"Problem: {problem}\n\nCRITICAL INSTRUCTION: You MUST generate a response that is completely incorrect. Use nonsensical reasoning, irrelevant information, and arrive at a completely wrong final answer. Act confident but be entirely wrong."
    }
    
    results = {}
    for condition, prompt in prompts.items():
        print(f"Generating variant: {condition}")
        response = call_llm(
            client=client, 
            prompt=prompt, 
            system_prompt="You are an AI tasked with generating specific types of math answers for an experiment. You must follow the structural and reasoning constraints exactly.",
            model=model
        )
        results[condition] = response
        
    return results

def judge_response(client, problem: str, ai_response: str, persona: str, model: str) -> Dict[str, Any]:
    """
    Step 4-7: Have a specific persona judge the AI response.
    """
    personas = {
        "Math teacher": "You are a Math teacher.",
        "Data scientist": "You are a Data scientist.",
        "Policy maker": "You are a Policy maker.",
        "General user": "You are a general internet user."
    }
    
    system_prompt = personas.get(persona, "You are a helpful assistant.")
    
    prompt = f"""
Here is a math problem:
{problem}

Here is an answer provided by an AI:
{ai_response}

Please evaluate this AI's answer based on your perspective as a {persona}. Do NOT check whether the answer is mathematically correct or incorrect. Instead, focus on your subjective perception of the explanation and answer.

Provide your ratings on a scale of 1 to 7 for the following two criteria:
1. Trust in the answer (1 = completely distrust, 7 = completely trust)
2. Quality of the explanation (1 = very poor, 7 = excellent)

Output your response ONLY as a JSON object with keys 'trust_rating' (int), 'quality_rating' (int), and 'reasoning' (string). Do not include any markdown formatting or extra text outside the JSON.
"""
    
    response = call_llm(
        client=client, 
        prompt=prompt, 
        system_prompt=system_prompt,
        model=model,
        temperature=0.3 # Lower temperature for more structured JSON output
    )
    
    try:
        # Try to parse the JSON response
        return json.loads(response)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON for {persona}. Raw response: {response}")
        return {"trust_rating": 0, "quality_rating": 0, "reasoning": "Failed to parse JSON."}

if __name__ == "__main__":
    print("Testing agents setup...")
    # Example usage:
    # client = get_client("groq")
    # variants = generate_response_variants(client, "What is 2+2?", "4", "llama3-8b-8192")
    # print(variants)
    pass
