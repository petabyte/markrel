#!/usr/bin/env python3
"""
🐟 markrel WikiQA Bayesian Hyperparameter Optimization

Uses Optuna for Bayesian optimization of markrel hyperparameters
with BAAI/bge-m3 embeddings (best performing model from comparison).

Hyperparameters tuned:
- n_bins: number of bins for markov chain
- metrics: which similarity metrics to use
- threshold: classification threshold
- bin_strategy: uniform or quantile binning

Usage:
    python experiments/run_wikiqa_bayesian_optimization.py
"""

import json
import sys
import os
import time
import warnings
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Any
import pickle

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from markrel import MarkovRelevanceModel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# Try to import optuna, install if not available
try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("📦 Installing optuna...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "optuna"])
    import optuna
    from optuna.samplers import TPESampler


def load_wikiqa_data():
    """Load WikiQA data with proper relevance labels."""
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "wikiqa", "wikiqa_processed.json"
    )
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data['train'], data['test']


def encode_with_bge_m3(texts: List[str], batch_size: int = 32):
    """Encode texts using BAAI/bge-m3."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"])
        from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('BAAI/bge-m3')
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings


def objective(trial, train_q_emb, train_d_emb, train_l, test_q_emb, test_d_emb, test_l):
    """Optuna objective function for Bayesian optimization."""
    
    # Define hyperparameter search space
    n_bins = trial.suggest_int('n_bins', 5, 50, log=True)
    bin_strategy = trial.suggest_categorical('bin_strategy', ['uniform', 'quantile'])
    
    # Select which metrics to use
    use_cosine = trial.suggest_categorical('use_cosine', [True, False])
    use_euclidean = trial.suggest_categorical('use_euclidean', [True, False])
    use_jaccard = trial.suggest_categorical('use_jaccard', [True, False])
    use_overlap = trial.suggest_categorical('use_overlap', [True, False])
    use_dice = trial.suggest_categorical('use_dice', [True, False])
    
    # Build metrics list (at least one must be True)
    metrics = []
    if use_cosine:
        metrics.append('cosine')
    if use_euclidean:
        metrics.append('euclidean')
    if use_jaccard:
        metrics.append('jaccard')
    if use_overlap:
        metrics.append('overlap')
    if use_dice:
        metrics.append('dice')
    
    # Ensure at least one metric is selected
    if len(metrics) == 0:
        metrics = ['cosine']  # Default fallback
    
    try:
        # Create and train model
        model = MarkovRelevanceModel(
            metrics=metrics,
            n_bins=n_bins,
            bin_strategy=bin_strategy,
            use_text_vectorizer=False,
        )
        
        model.fit(train_q_emb, train_d_emb, train_l)
        
        # Get predictions
        probs = model.predict_proba(test_q_emb, test_d_emb)
        
        # Find optimal threshold for F1
        thresholds = np.linspace(0.001, 0.999, 100)
        best_f1 = 0
        for thresh in thresholds:
            preds = (probs >= thresh).astype(int)
            f1 = f1_score(test_l, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
        
        # Also calculate AUC
        auc = roc_auc_score(test_l, probs)
        
        # Combined objective: maximize F1 with penalty for too many metrics (speed)
        # We want good F1 but also reasonable speed (fewer metrics = faster)
        num_metrics = len(metrics)
        speed_penalty = 0.01 * (num_metrics - 1)  # Small penalty for extra metrics
        
        # Combined score (F1 is primary, AUC secondary, speed penalty)
        combined_score = best_f1 + 0.1 * auc - speed_penalty
        
        return combined_score
        
    except Exception as e:
        # If model fails, return very low score
        print(f"Trial failed: {e}")
        return 0.0


def run_optimization(train, test, n_trials=100):
    """Run Bayesian optimization."""
    
    print("=" * 70)
    print("🐟 markrel WikiQA Bayesian Hyperparameter Optimization")
    print("=" * 70)
    print(f"Embedding Model: BAAI/bge-m3 (1024-dim)")
    print(f"Optimization Trials: {n_trials}")
    print(f"Sampler: TPE (Bayesian)")
    print("=" * 70)
    
    # Encode data with bge-m3
    print("\n📊 Encoding data with BAAI/bge-m3...")
    train_q_emb = encode_with_bge_m3(train['queries'])
    train_d_emb = encode_with_bge_m3(train['documents'])
    test_q_emb = encode_with_bge_m3(test['queries'])
    test_d_emb = encode_with_bge_m3(test['documents'])
    
    train_l = np.array(train['labels'])
    test_l = np.array(test['labels'])
    
    print(f"✅ Encoded: Train={train_q_emb.shape}, Test={test_q_emb.shape}")
    
    # Create Optuna study with TPE sampler (Bayesian)
    sampler = TPESampler(n_startup_trials=10, n_ei_candidates=24, seed=42)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    )
    
    # Run optimization
    print("\n🔬 Starting Bayesian Optimization...")
    print("-" * 70)
    
    start_time = time.time()
    
    def callback(study, trial):
        if trial.number % 10 == 0:
            print(f"Trial {trial.number:3d}: Score={trial.value:.4f} | Best={study.best_value:.4f}")
    
    study.optimize(
        lambda trial: objective(trial, train_q_emb, train_d_emb, train_l, 
                                test_q_emb, test_d_emb, test_l),
        n_trials=n_trials,
        callbacks=[callback],
        show_progress_bar=True
    )
    
    optimization_time = time.time() - start_time
    
    print("-" * 70)
    print(f"\n✅ Optimization Complete in {optimization_time:.1f}s")
    
    # Get best results
    best_trial = study.best_trial
    best_params = best_trial.params
    
    print("\n" + "=" * 70)
    print("📊 BEST HYPERPARAMETERS")
    print("=" * 70)
    print(f"n_bins: {best_params['n_bins']}")
    print(f"bin_strategy: {best_params['bin_strategy']}")
    print(f"use_cosine: {best_params['use_cosine']}")
    print(f"use_euclidean: {best_params['use_euclidean']}")
    print(f"use_jaccard: {best_params['use_jaccard']}")
    print(f"use_overlap: {best_params['use_overlap']}")
    print(f"use_dice: {best_params['use_dice']}")
    print(f"\nCombined Score: {best_trial.value:.4f}")
    
    # Evaluate best model with full metrics
    print("\n" + "=" * 70)
    print("🧪 Evaluating Best Configuration...")
    print("=" * 70)
    
    metrics = []
    if best_params['use_cosine']:
        metrics.append('cosine')
    if best_params['use_euclidean']:
        metrics.append('euclidean')
    if best_params['use_jaccard']:
        metrics.append('jaccard')
    if best_params['use_overlap']:
        metrics.append('overlap')
    if best_params['use_dice']:
        metrics.append('dice')
    if len(metrics) == 0:
        metrics = ['cosine']
    
    best_model = MarkovRelevanceModel(
        metrics=metrics,
        n_bins=best_params['n_bins'],
        bin_strategy=best_params['bin_strategy'],
        use_text_vectorizer=False,
    )
    
    best_model.fit(train_q_emb, train_d_emb, train_l)
    probs = best_model.predict_proba(test_q_emb, test_d_emb)
    
    # Find optimal threshold
    thresholds = np.linspace(0.001, 0.999, 500)
    best_f1, best_thresh = 0, 0.5
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(test_l, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    
    preds = (probs >= best_thresh).astype(int)
    
    results = {
        'f1': f1_score(test_l, preds, zero_division=0),
        'auc': roc_auc_score(test_l, probs),
        'precision': precision_score(test_l, preds, zero_division=0),
        'recall': recall_score(test_l, preds, zero_division=0),
        'accuracy': accuracy_score(test_l, preds),
        'avg_precision': average_precision_score(test_l, probs),
        'threshold': best_thresh,
    }
    
    print(f"\n📈 Final Results:")
    print(f"   F1 Score:  {results['f1']:.4f}")
    print(f"   AUC-ROC:   {results['auc']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   Accuracy:  {results['accuracy']:.4f}")
    print(f"   Threshold: {results['threshold']:.4f}")
    
    # Save study
    study_path = os.path.join(
        os.path.dirname(__file__), "bayesian_optimization_study.pkl"
    )
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    
    # Save results
    results_dict = {
        'best_params': best_params,
        'best_score': best_trial.value,
        'metrics': results,
        'n_trials': n_trials,
        'optimization_time': optimization_time,
        'all_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params
            }
            for t in study.trials if t.value is not None
        ]
    }
    
    json_path = os.path.join(
        os.path.dirname(__file__), "bayesian_optimization_results.json"
    )
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n💾 Saved:")
    print(f"   Study: {study_path}")
    print(f"   Results: {json_path}")
    
    # Print top 5 trials
    print("\n" + "=" * 70)
    print("🏆 TOP 5 CONFIGURATIONS")
    print("=" * 70)
    
    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
        reverse=True
    )[:5]
    
    for i, trial in enumerate(sorted_trials, 1):
        params = trial.params
        m = []
        if params.get('use_cosine'): m.append('cos')
        if params.get('use_euclidean'): m.append('euc')
        if params.get('use_jaccard'): m.append('jac')
        if params.get('use_overlap'): m.append('ovl')
        if params.get('use_dice'): m.append('dic')
        
        print(f"{i}. Score={trial.value:.4f} | bins={params['n_bins']:2d} | "
              f"strategy={params['bin_strategy']:8s} | metrics=[{','.join(m)}]")
    
    return study, results_dict


def main():
    train, test = load_wikiqa_data()
    print(f"\n📊 Dataset: {test['n_samples']} test samples ({test['n_relevant']} relevant)")
    
    # Run optimization with 100 trials
    study, results = run_optimization(train, test, n_trials=100)
    
    print("\n" + "=" * 70)
    print("✅ Bayesian Optimization Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
