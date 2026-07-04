1. Prepare the dataset
Collect all trust ratings (1–7) from judge outputs.
Structure data in long format with columns:
problem_id
judge_persona
answer_correctness (correct / wrong)
reasoning_quality (correct / wrong)
trust_rating
explanation_quality
2. Run 2×2 within-subjects ANOVA (Trust as outcome)
Factors:
Answer correctness: correct vs wrong
Reasoning quality: correct vs wrong
Dependent variable:
Trust rating
Within-subject structure:
Same problems evaluated under all conditions
Output:
Main effect of answer correctness
Main effect of reasoning quality
Interaction effect (correctness × reasoning)
3. Compute “Explainability Paradox Effect Size”
Define:
Condition A: wrong answer + correct reasoning
Condition B: wrong answer + wrong reasoning
Compute:
Effect size = mean(trust_A) − mean(trust_B)
Report:
Mean difference
Optional: Cohen’s d for paired samples
4. Multiple regression for comparative influence
Build regression model:
Dependent variable: trust rating
Predictors:
answer correctness (binary encoded)
reasoning quality (binary encoded)
Compute:
Standardized beta coefficients (β)
Compare:
β_reasoning vs β_answer_correctness
Interpretation:
Whether reasoning has stronger effect on trust than correctness
5. Mediation analysis (reasoning → explanation quality → trust)
Define variables:
X = reasoning quality
M = explanation quality rating
Y = trust rating
Steps:
Test X → M (does reasoning affect explanation quality?)
Test M → Y (does explanation quality affect trust?)
Test X → Y (direct effect)
Test indirect effect (X → M → Y)
Output:
Mediation effect size
Significance (bootstrap confidence intervals preferred)
6. Reporting structure
ANOVA results (main + interaction effects)
Explainability paradox effect size
Regression comparison (standardized betas)
Mediation results (direct + indirect effects)