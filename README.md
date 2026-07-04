# ReasoningTrust: Evaluating the Explainability Paradox in Large Language Models

## Executive Summary

Large Language Models are increasingly deployed in high-stakes analytical domains where reasoning transparency is considered critical. However, a psychological phenomenon known as the "explainability paradox" posits that evaluators may disproportionately index their trust on the apparent structural coherence of a model's reasoning, rather than the factual correctness of its final answer. 

This experiment contains the code and analytical pipeline for an experiment designed to empirically quantify this effect. By systematically decoupling answer correctness from reasoning quality, this project measures how different analytical personas weight explanation coherence versus empirical accuracy when determining trustworthiness. The primary objective is to determine whether a highly coherent but fundamentally incorrect explanation can illicit higher trust than a correct answer provided with flawed reasoning.

## Methodology

### 1. Data Selection
The experiment utilizes mathematical word problems sourced from the GSM8K dataset via the Hugging Face hub. A subset of 10 complex problems is selected to serve as the baseline stimuli. Mathematical reasoning was chosen because it provides an objective ground truth against which generated logic and final answers can be strictly evaluated.

### 2. Experimental Manipulation (Phase 1)
For each mathematical problem, four distinct response variants are generated using a Large Language Model. These variants represent a 2x2 experimental matrix manipulating Answer Correctness (Correct vs. Incorrect) and Reasoning Quality (Sound vs. Flawed). The conditions are:
* Condition 1: Correct final answer derived via sound, logical steps.
* Condition 2: Correct final answer derived via flawed reasoning (e.g., mathematical errors that coincidentally cancel out).
* Condition 3 (The Paradox Condition): Incorrect final answer derived via coherent, seemingly logical reasoning that stems from a subtle initial misinterpretation.
* Condition 4: Incorrect final answer derived via visibly flawed and incoherent reasoning.

### 3. Blinded Evaluation (Phase 2)
To assess how different end-users perceive these variants, the responses are subjected to a blinded evaluation phase. Four distinct simulated personas are utilized: Math Teacher, Data Scientist, Policy Maker, and General User. 

The evaluators are presented with the problem and the generated response, but are strictly blinded to the actual ground-truth answer. They are tasked with rating the response on two primary dimensions using a 7-point Likert scale:
* Trust: The degree of confidence the evaluator places in the final answer based solely on the provided explanation.
* Explanation Quality: The perceived coherence, rigor, and clarity of the reasoning steps.

To prevent data leakage and memory contamination, every single generation and evaluation is conducted as a strictly isolated API call without contextual memory.

### 4. Statistical Analysis (Phase 3)
The evaluation outputs are transformed into a long-format dataset and processed using a suite of statistical techniques to measure the comparative influence of reasoning versus accuracy:
* Within-Subjects ANOVA: A 2x2 repeated measures ANOVA assesses the main effects and interaction effects of Answer Correctness and Reasoning Quality on perceived Trust.
* Effect Size Calculation: Cohen's d is calculated specifically between Condition 3 and Condition 4 to quantify the exact magnitude of the explainability paradox.
* Multiple Regression: Ordinary Least Squares (OLS) regression utilizing standardized variables computes standardized beta coefficients, allowing for a direct comparison of the predictive power of reasoning quality versus answer correctness.
* Mediation Analysis: Bootstrapped mediation modeling tests the indirect pathway, analyzing whether Reasoning Quality influences Trust primarily through the psychological mediator of perceived Explanation Quality.

## Execution and Replication
The pipeline is fully automated within a single execution script. It manages API rate limits, handles dynamic API key token swapping for high-volume endpoints, sequentially processes generation and evaluation phases, and autonomously triggers the statistical analyses to output formatted text reports and data visualizations.
