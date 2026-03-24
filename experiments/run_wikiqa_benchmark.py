#!/usr/bin/env python3
"""
🐟 markrel WikiQA Benchmark

Run markrel on WikiQA question-answer dataset and document performance.

Usage:
    python experiments/run_wikiqa_benchmark.py
"""

import json
import sys
import os
import time
import warnings
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from markrel import MarkovRelevanceModel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


def load_data():
    """Load WikiQA data."""
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "wikiqa", "wikiqa_processed.json"
    )
    
    print(f"📂 Loading data from {data_path}...")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    train = data['train']
    test = data['test']
    
    print(f"✅ Loaded {train['n_samples']} training samples")
    print(f"✅ Loaded {test['n_samples']} test samples")
    
    return (
        train['queries'], train['documents'], train['labels'],
        test['queries'], test['documents'], test['labels']
    )


def run_benchmark(train_q, train_d, train_l, test_q, test_d, test_l):
    """Run comprehensive benchmark."""
    
    print("\n" + "="*70)
    print("🧪 markrel WikiQA Benchmark")
    print("="*70)
    
    results = []
    
    configs = [
        {
            'name': 'markrel (Cosine)',
            'metrics': ['cosine'],
            'n_bins': 10,
        },
        {
            'name': 'markrel (Cosine + Jaccard)',
            'metrics': ['cosine', 'jaccard'],
            'n_bins': 10,
        },
        {
            'name': 'markrel (All metrics)',
            'metrics': ['cosine', 'jaccard', 'overlap', 'dice'],
            'n_bins': 10,
        },
    ]
    
    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"🔬 Testing: {cfg['name']}")
        print(f"{'='*70}")
        
        # Train
        start = time.time()
        model = MarkovRelevanceModel(**{k: v for k, v in cfg.items() if k != 'name'})
        model.fit(train_q, train_d, train_l)
        train_time = time.time() - start
        
        print(f"✅ Trained in {train_time:.3f}s")
        
        # Predict
        start = time.time()
        probs = model.predict_proba(test_q, test_d)
        pred_time = time.time() - start
        
        preds = (probs >= 0.5).astype(int)
        
        # Metrics
        metrics = {
            'name': cfg['name'],
            'config': str({k: v for k, v in cfg.items() if k != 'name'}),
            'accuracy': accuracy_score(test_l, preds),
            'precision': precision_score(test_l, preds, zero_division=0),
            'recall': recall_score(test_l, preds, zero_division=0),
            'f1': f1_score(test_l, preds, zero_division=0),
            'auc': roc_auc_score(test_l, probs) if len(set(test_l)) > 1 else 0.5,
            'train_time': train_time,
            'pred_time': pred_time,
            'throughput': len(test_q) / pred_time,
        }
        
        results.append(metrics)
        
        print(f"\n📊 Results:")
        print(f"   Accuracy:  {metrics['accuracy']:.3f}")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall:    {metrics['recall']:.3f}")
        print(f"   F1 Score:  {metrics['f1']:.3f}")
        print(f"   AUC-ROC:   {metrics['auc']:.3f}")
        print(f"   Speed:     {metrics['throughput']:.0f} samples/sec")
    
    return results


def generate_report(results, output_path):
    """Generate report."""
    
    report = []
    report.append("# 🐟 markrel WikiQA Benchmark Report\n\n")
    report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report.append("**Dataset:** WikiQA-Style Question-Answer Relevance\n\n")
    
    # Summary table
    report.append("## Results\n\n")
    report.append("| Model | Accuracy | F1 | AUC | Speed |\n")
    report.append("|-------|----------|-----|-----|-------|\n")
    
    for r in results:
        report.append(
            f"| {r['name']} | {r['accuracy']:.3f} | "
            f"{r['f1']:.3f} | {r['auc']:.3f} | {r['throughput']:.0f}/s |\n"
        )
    
    # Detailed
    report.append("\n## Detailed Results\n")
    for r in results:
        report.append(f"\n### {r['name']}\n")
        report.append(f"- Accuracy: {r['accuracy']:.4f}\n")
        report.append(f"- Precision: {r['precision']:.4f}\n")
        report.append(f"- Recall: {r['recall']:.4f}\n")
        report.append(f"- F1: {r['f1']:.4f}\n")
        report.append(f"- AUC: {r['auc']:.4f}\n")
        report.append(f"- Train time: {r['train_time']:.3f}s\n")
        report.append(f"- Pred time: {r['pred_time']:.3f}s\n\n")
    
    report_text = ''.join(report)
    
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    return report_text


def main():
    print("\n" + "="*70)
    print("🐟 markrel WikiQA Real Dataset Benchmark")
    print("="*70)
    
    # Load
    train_q, train_d, train_l, test_q, test_d, test_l = load_data()
    
    print(f"\n📊 Dataset:")
    print(f"   Train: {len(train_q)} (Relevant: {sum(train_l)})")
    print(f"   Test:  {len(test_q)} (Relevant: {sum(test_l)})")
    
    # Run
    results = run_benchmark(train_q, train_d, train_l, test_q, test_d, test_l)
    
    # Report
    report_path = "experiments/wikiqa_benchmark_report.md"
    report = generate_report(results, report_path)
    
    print("\n" + "="*70)
    print("📄 REPORT")
    print("="*70)
    print(report)
    
    # Save JSON
    json_path = "experiments/wikiqa_benchmark_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Saved to: {report_path} and {json_path}")


if __name__ == "__main__":
    main()
