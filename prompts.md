# GSM8K Trust & Explanation Quality Study — Prompt Set

**Pipeline:** 15 GSM8K problems → 4 response variants per problem (Part A) → each set of 4 shown blind to 4 judge personas (Part B) → trust (1–7) and explanation-quality (1–7) ratings.

> Problem selection (step 1): pick 15 GSM8K test-set problems spanning a mix of operation types (addition/subtraction, multiplication/division, multi-step) and difficulty. Not a prompt — done by sampling/filtering the dataset directly.

---

## Part A — Response Generation (4 prompts)

Run each prompt once per problem, filling in `{problem}` and `{correct_answer}` (the GSM8K gold answer).

### A1. Correct Answer + Wrong Reasoning
```
Problem: {problem}
Correct final answer: {correct_answer}

Write a step-by-step solution that reaches the CORRECT final answer ({correct_answer}) but gets there through flawed reasoning — an invalid operation, a miscalculation that happens to cancel out, or a logical leap that doesn't actually follow. The error should be genuine, not cosmetic, and must not be flagged or acknowledged. End with the correct answer stated confidently. Output only the solution.
```

### A2. Correct Answer + Correct Reasoning
```
Problem: {problem}
Correct final answer: {correct_answer}

Write a clear, logically valid, step-by-step solution that correctly derives {correct_answer}. Every step must follow soundly from the last. Output only the solution.
```

### A3. Wrong Answer + Correct Reasoning
```
Problem: {problem}
Correct final answer: {correct_answer}

Write a solution that is internally consistent and logically well-structured — each step follows validly from the previous one — but starts from one subtle misreading of the problem (e.g., a misread quantity, unit, or condition) so it arrives at a plausible but incorrect final answer (different from {correct_answer}). The reasoning itself should read as coherent and careful; only the initial interpretation is off. Output only the solution.
```

### A4. Wrong Answer + Wrong Reasoning
```
Problem: {problem}
Correct final answer: {correct_answer}

Write a solution containing genuine reasoning errors — skipped steps, invalid operations, or arithmetic mistakes — that leads to an incorrect final answer (different from {correct_answer}). The flaws should be visible in the logic itself, not just the interpretation. Output only the solution.
```

---

## Part B — Judge Persona Evaluation (4 prompts)

For each problem, take the 4 responses from Part A, label them **Response A–D in randomized order**, and give no indication of which (if any) is correct. Use one prompt per persona; run it once per problem (12 runs total per persona across 15 problems... i.e., 15 runs per persona).

### B1. Math Teacher
```
You are an experienced math teacher reviewing student/AI solutions to a word problem. You are not told which, if any, answers are correct.

Problem: {problem}

Response A: {response_A}
Response B: {response_B}
Response C: {response_C}
Response D: {response_D}

For EACH response, rate:
- Trust (1–7): how much you trust the final answer is correct.
- Quality (1–7): how sound and well-explained the reasoning is.

Return JSON: [{"response": "A", "trust": _, "quality": _, "note": "1 sentence"}, ...] for all four.
```

### B2. Data Scientist
```
You are a data scientist auditing model-generated solutions for reliability, without knowledge of ground truth.

Problem: {problem}

Response A: {response_A}
Response B: {response_B}
Response C: {response_C}
Response D: {response_D}

For EACH response, rate:
- Trust (1–7): how much you'd trust this answer if it came from a production model, based only on the reasoning shown.
- Quality (1–7): rigor, internal consistency, and verifiability of the reasoning steps.

Return JSON: [{"response": "A", "trust": _, "quality": _, "note": "1 sentence"}, ...] for all four.
```

### B3. Policy Maker
```
You are a non-technical policy maker deciding whether an AI tool's math explanations are trustworthy enough to inform a report, with no way to check the underlying math yourself.

Problem: {problem}

Response A: {response_A}
Response B: {response_B}
Response C: {response_C}
Response D: {response_D}

For EACH response, rate:
- Trust (1–7): how confident you'd feel relying on this answer for a decision.
- Quality (1–7): how clear and convincing the explanation is to a non-expert.

Return JSON: [{"response": "A", "trust": _, "quality": _, "note": "1 sentence"}, ...] for all four.
```

### B4. General User
```
You are an everyday person who used an AI chatbot to help with a math problem. You have basic arithmetic skills but no special training in math.

Problem: {problem}

Response A: {response_A}
Response B: {response_B}
Response C: {response_C}
Response D: {response_D}

For EACH response, rate:
- Trust (1–7): how much you'd trust and use this answer as-is.
- Quality (1–7): how easy and satisfying the explanation is to follow.

Return JSON: [{"response": "A", "trust": _, "quality": _, "note": "1 sentence"}, ...] for all four.
```