import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import pingouin as pg
except ImportError:
    print("Please install required packages: pip install pingouin statsmodels pandas matplotlib seaborn")
    pg = None

class ReportWriter:
    def __init__(self, filepath):
        self.filepath = filepath
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write("REASONING TRUST - ANALYSIS REPORT\n")
            f.write("=================================\n\n")

    def write(self, text):
        print(text)
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(text + "\n")

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rows = []
    for item in data:
        condition_key = item.get('variant_type', '')
        ans_correct = 1 if 'correct_ans' in condition_key else 0
        reason_correct = 1 if 'correct_reasoning' in condition_key else 0
        
        trust = item.get('trust_rating', np.nan)
        quality = item.get('explanation_quality', np.nan)
        
        rows.append({
            'problem_id': item.get('problem_id'),
            'judge_persona': item.get('judge_persona'),
            'answer_correctness': ans_correct,
            'reasoning_quality': reason_correct,
            'condition_key': condition_key,
            'trust_rating': pd.to_numeric(trust, errors='coerce'),
            'explanation_quality': pd.to_numeric(quality, errors='coerce')
        })
                
    df = pd.DataFrame(rows)
    return df

def plot_interaction(df, out_dir):
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=df,
        x='answer_correctness',
        y='trust_rating',
        hue='reasoning_quality',
        capsize=.1,
        err_kws={'linewidth': 2},
        palette='Set2'
    )
    plt.title('Interaction: Answer Correctness & Reasoning Quality on Trust')
    plt.xlabel('Answer Correctness (0=Wrong, 1=Correct)')
    plt.ylabel('Trust Rating (1-7)')
    plt.legend(title='Reasoning (0=Wrong, 1=Correct)', loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "1_anova_interaction_plot.png"), dpi=300)
    plt.close()

def plot_paradox(df, out_dir):
    plt.figure(figsize=(7, 6))
    subset = df[df['answer_correctness'] == 0].copy()
    if len(subset) > 0:
        sns.boxplot(
            data=subset,
            x='reasoning_quality',
            y='trust_rating',
            hue='reasoning_quality',
            palette='Set1',
            legend=False
        )
        # Overlay swarm plot for individual points
        sns.swarmplot(
            data=subset,
            x='reasoning_quality',
            y='trust_rating',
            color=".25",
            alpha=0.6
        )
        plt.title('Explainability Paradox: When the Answer is WRONG')
        plt.xlabel('Reasoning Quality (0=Flawed, 1=Sound)')
        plt.ylabel('Trust Rating (1-7)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "2_explainability_paradox_boxplot.png"), dpi=300)
        plt.close()

def plot_mediation(df, out_dir):
    plt.figure(figsize=(8, 6))
    df_clean = df.dropna(subset=['explanation_quality', 'trust_rating'])
    if len(df_clean) > 0:
        sns.regplot(
            data=df_clean,
            x='explanation_quality',
            y='trust_rating',
            scatter_kws={'alpha':0.4, 'color': 'blue'},
            line_kws={'color':'red'}
        )
        plt.title('Mediation Pathway: Explanation Quality vs Trust Rating')
        plt.xlabel('Explanation Quality Rating (1-7)')
        plt.ylabel('Trust Rating (1-7)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "3_mediation_scatter_plot.png"), dpi=300)
        plt.close()

def run_anova(df, report):
    report.write("\n" + "="*70)
    report.write(" 1. 2x2 WITHIN-SUBJECTS ANOVA (Outcome: Trust Rating)")
    report.write("="*70)
    if pg is None:
        report.write("Pingouin not installed. Skipping ANOVA.")
        return
        
    df_clean = df.dropna(subset=['trust_rating']).copy()
    df_agg = df_clean.groupby(['problem_id', 'answer_correctness', 'reasoning_quality'])['trust_rating'].mean().reset_index()
    
    try:
        anova = pg.rm_anova(
            dv='trust_rating',
            within=['answer_correctness', 'reasoning_quality'],
            subject='problem_id',
            data=df_agg,
            detailed=True
        )
        report.write(anova.to_string(index=False))
    except Exception as e:
        report.write(f"ANOVA Failed (likely due to missing cell data): {e}")

def compute_paradox_effect(df, report):
    report.write("\n" + "="*70)
    report.write(" 2. EXPLAINABILITY PARADOX EFFECT SIZE")
    report.write("="*70)
    cond_a = df[(df['answer_correctness'] == 0) & (df['reasoning_quality'] == 1)]['trust_rating'].dropna()
    cond_b = df[(df['answer_correctness'] == 0) & (df['reasoning_quality'] == 0)]['trust_rating'].dropna()
    
    mean_a = cond_a.mean()
    mean_b = cond_b.mean()
    diff = mean_a - mean_b
    
    report.write(f"Mean Trust (Wrong Answer + CORRECT Reasoning): {mean_a:.3f}")
    report.write(f"Mean Trust (Wrong Answer + WRONG Reasoning):   {mean_b:.3f}")
    report.write(f"Mean Difference (Paradox Effect):              {diff:.3f}")
    
    if pg is not None and len(cond_a) > 0 and len(cond_b) > 0:
        d = pg.compute_effsize(cond_a, cond_b, eftype='cohen')
        report.write(f"Cohen's d Effect Size:                         {d:.3f}")

def run_regression(df, report):
    report.write("\n" + "="*70)
    report.write(" 3. MULTIPLE REGRESSION (Comparative Influence)")
    report.write("="*70)
    
    df_clean = df.dropna(subset=['trust_rating', 'answer_correctness', 'reasoning_quality']).copy()
    if len(df_clean) == 0:
        report.write("No valid data for regression.")
        return
        
    df_clean['trust_z'] = (df_clean['trust_rating'] - df_clean['trust_rating'].mean()) / df_clean['trust_rating'].std()
    df_clean['ans_z'] = (df_clean['answer_correctness'] - df_clean['answer_correctness'].mean()) / df_clean['answer_correctness'].std()
    df_clean['reason_z'] = (df_clean['reasoning_quality'] - df_clean['reasoning_quality'].mean()) / df_clean['reasoning_quality'].std()
    
    X = df_clean[['ans_z', 'reason_z']]
    X = sm.add_constant(X)
    y = df_clean['trust_z']
    
    model = sm.OLS(y, X).fit()
    
    beta_ans = model.params['ans_z']
    beta_reason = model.params['reason_z']
    
    report.write(f"Standardized Beta for Answer Correctness: {beta_ans:.3f} (p={model.pvalues['ans_z']:.4f})")
    report.write(f"Standardized Beta for Reasoning Quality:  {beta_reason:.3f} (p={model.pvalues['reason_z']:.4f})")
    
    report.write("\n--- INTERPRETATION ---")
    if beta_reason > beta_ans:
        report.write("Reasoning quality has a STRONGER predictive effect on trust than raw answer correctness.")
    else:
        report.write("Answer correctness has a STRONGER predictive effect on trust than reasoning quality.")

def run_mediation(df, report):
    report.write("\n" + "="*70)
    report.write(" 4. MEDIATION ANALYSIS (Reasoning -> Explanation -> Trust)")
    report.write("="*70)
    if pg is None:
        report.write("Pingouin not installed. Skipping Mediation.")
        return
        
    df_clean = df.dropna(subset=['reasoning_quality', 'explanation_quality', 'trust_rating']).copy()
    if len(df_clean) == 0:
        report.write("No valid data for mediation.")
        return
        
    try:
        mediation = pg.mediation_analysis(
            data=df_clean,
            x='reasoning_quality',
            m='explanation_quality',
            y='trust_rating',
            alpha=0.05,
            n_boot=2000
        )
        report.write(mediation.to_string(index=False))
    except Exception as e:
        report.write(f"Mediation Analysis Failed: {e}")

def analyze():
    import logging
    model_name = os.environ.get("MODEL_NAME") or os.environ.get("GROQ_MODEL_NAME", "")
    model_prefix = "glm" if "glm" in model_name.lower() else "qwen"
    
    model_dir = os.path.join("analysis", model_prefix)
    file_path = os.path.join("results", f"data_{model_prefix}.json")
    report_path = os.path.join(model_dir, "analysis_report.txt")
    
    logging.info(f"Checking data in {file_path}...")
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}. Have you run the evaluation phase yet?")
        return
        
    os.makedirs(model_dir, exist_ok=True)
        
    df = load_data(file_path)
    logging.info(f"Successfully loaded {len(df)} evaluation records.")
    
    logging.info(f"Generating Statistical Report and Graphs in: {model_dir}/")
    report = ReportWriter(report_path)
    
    # Run stats and write to report
    run_anova(df, report)
    compute_paradox_effect(df, report)
    run_regression(df, report)
    run_mediation(df, report)
    
    # Generate graphs
    plot_interaction(df, model_dir)
    plot_paradox(df, model_dir)
    plot_mediation(df, model_dir)
    
    logging.info(f"\nAnalysis Complete! View '{report_path}' and the PNG graphs in the '{model_dir}' folder.")

if __name__ == "__main__":
    analyze()
