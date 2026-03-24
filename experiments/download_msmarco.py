#!/usr/bin/env python3
"""
Download and process real MS MARCO dataset for markrel benchmarking.

MS MARCO (Microsoft Machine Reading Comprehension) is a large-scale dataset
for passage retrieval, document ranking, and question answering.

Dataset size:
- ~8.8M passages
- ~500K training queries
- ~6,900 dev queries

We'll use a sample for benchmarking.
"""

import os
import sys
import json
import gzip
import urllib.request
import urllib.error
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# MS MARCO URLs
MSMARCO_URLS = {
    "train_queries": "https://msmarco.blob.core.windows.net/msmarcoranking/train_queries.tsv.gz",
    "train_qrels": "https://msmarco.blob.core.windows.net/msmarcoranking/train_qrels.tsv.gz",
    "dev_queries": "https://msmarco.blob.core.windows.net/msmarcoranking/dev_queries.tsv.gz",
    "dev_qrels": "https://msmarco.blob.core.windows.net/msmarcoranking/dev_qrels.tsv.gz",
    "passages": "https://msmarco.blob.core.windows.net/msmarcoranking/collection.tsv.gz",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "msmarco")


def download_file(url: str, output_path: str) -> bool:
    """Download a file from URL."""
    if os.path.exists(output_path):
        print(f"  ✓ Already exists: {os.path.basename(output_path)}")
        return True
    
    print(f"  ⬇️  Downloading {os.path.basename(url)}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"  ✓ Downloaded: {os.path.basename(output_path)}")
        return True
    except Exception as e:
        print(f"  ✗ Error downloading: {e}")
        return False


def download_msmarco_files() -> bool:
    """Download all MS MARCO files."""
    print("\n" + "="*60)
    print("📥 Downloading MS MARCO Dataset")
    print("="*60)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for name, url in MSMARCO_URLS.items():
        output_path = os.path.join(DATA_DIR, os.path.basename(url))
        if not download_file(url, output_path):
            return False
    
    print("\n✅ All files downloaded successfully!")
    return True


def read_gzipped_tsv(filepath: str) -> Dict[str, str]:
    """Read a gzipped TSV file into a dictionary."""
    data = {}
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    data[parts[0]] = parts[1]
    return data


def read_qrels(filepath: str) -> Dict[str, List[str]]:
    """Read qrels file (query_id -> [passage_ids])."""
    qrels = defaultdict(list)
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 4:
                    query_id = parts[0]
                    passage_id = parts[2]
                    relevance = int(parts[3])
                    if relevance > 0:  # Only positive relevance
                        qrels[query_id].append(passage_id)
    return dict(qrels)


def create_sample_dataset(
    n_train: int = 5000,
    n_test: int = 1000,
    n_negatives_per_positive: int = 5
) -> Tuple[List[str], List[str], List[int], List[str], List[str], List[int]]:
    """
    Create a sampled dataset from MS MARCO.
    
    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        n_negatives_per_positive: Number of negative samples per positive
    
    Returns:
        train_queries, train_docs, train_labels, test_queries, test_docs, test_labels
    """
    print("\n" + "="*60)
    print("📊 Processing MS MARCO Data")
    print("="*60)
    
    # Load data
    print("\n⏳ Loading passages...")
    passages_file = os.path.join(DATA_DIR, "collection.tsv.gz")
    passages = read_gzipped_tsv(passages_file)
    print(f"  ✓ Loaded {len(passages):,} passages")
    
    print("\n⏳ Loading train queries...")
    train_queries_file = os.path.join(DATA_DIR, "train_queries.tsv.gz")
    train_queries_dict = read_gzipped_tsv(train_queries_file)
    print(f"  ✓ Loaded {len(train_queries_dict):,} train queries")
    
    print("\n⏳ Loading train qrels...")
    train_qrels_file = os.path.join(DATA_DIR, "train_qrels.tsv.gz")
    train_qrels = read_qrels(train_qrels_file)
    print(f"  ✓ Loaded {len(train_qrels):,} queries with relevance judgments")
    
    print("\n⏳ Loading dev queries...")
    dev_queries_file = os.path.join(DATA_DIR, "dev_queries.tsv.gz")
    dev_queries_dict = read_gzipped_tsv(dev_queries_file)
    print(f"  ✓ Loaded {len(dev_queries_dict):,} dev queries")
    
    print("\n⏳ Loading dev qrels...")
    dev_qrels_file = os.path.join(DATA_DIR, "dev_qrels.tsv.gz")
    dev_qrels = read_qrels(dev_qrels_file)
    print(f"  ✓ Loaded {len(dev_qrels):,} dev queries with relevance judgments")
    
    # Create training set
    print(f"\n🎲 Creating training set ({n_train} samples)...")
    train_q_list = []
    train_d_list = []
    train_l_list = []
    
    query_ids_with_labels = [qid for qid in train_qrels if qid in train_queries_dict]
    np.random.shuffle(query_ids_with_labels)
    
    for qid in query_ids_with_labels:
        if len(train_q_list) >= n_train:
            break
            
        query_text = train_queries_dict[qid]
        relevant_passages = train_qrels[qid]
        
        # Add positive examples
        for pid in relevant_passages[:2]:  # Max 2 positives per query
            if pid in passages:
                train_q_list.append(query_text)
                train_d_list.append(passages[pid])
                train_l_list.append(1)
                
                if len(train_q_list) >= n_train:
                    break
        
        # Add negative examples
        n_negatives = min(n_negatives_per_positive, len(relevant_passages))
        neg_count = 0
        while neg_count < n_negatives and len(train_q_list) < n_train:
            random_pid = np.random.choice(list(passages.keys()))
            if random_pid not in relevant_passages:
                train_q_list.append(query_text)
                train_d_list.append(passages[random_pid])
                train_l_list.append(0)
                neg_count += 1
    
    print(f"  ✓ Created {len(train_q_list)} training samples")
    print(f"     Relevant: {sum(train_l_list)} | Not relevant: {len(train_l_list) - sum(train_l_list)}")
    
    # Create test set
    print(f"\n🎲 Creating test set ({n_test} samples)...")
    test_q_list = []
    test_d_list = []
    test_l_list = []
    
    dev_query_ids = [qid for qid in dev_qrels if qid in dev_queries_dict]
    np.random.shuffle(dev_query_ids)
    
    for qid in dev_query_ids:
        if len(test_q_list) >= n_test:
            break
            
        query_text = dev_queries_dict[qid]
        relevant_passages = dev_qrels[qid]
        
        # Add positive examples
        for pid in relevant_passages[:3]:
            if pid in passages:
                test_q_list.append(query_text)
                test_d_list.append(passages[pid])
                test_l_list.append(1)
                
                if len(test_q_list) >= n_test:
                    break
        
        # Add negative examples
        n_negatives = min(n_negatives_per_positive, len(relevant_passages))
        neg_count = 0
        while neg_count < n_negatives and len(test_q_list) < n_test:
            random_pid = np.random.choice(list(passages.keys()))
            if random_pid not in relevant_passages:
                test_q_list.append(query_text)
                test_d_list.append(passages[random_pid])
                test_l_list.append(0)
                neg_count += 1
    
    print(f"  ✓ Created {len(test_q_list)} test samples")
    print(f"     Relevant: {sum(test_l_list)} | Not relevant: {len(test_l_list) - sum(test_l_list)}")
    
    return train_q_list, train_d_list, train_l_list, test_q_list, test_d_list, test_l_list


def save_processed_data(
    train_q, train_d, train_l,
    test_q, test_d, test_l,
    output_dir: str = None
):
    """Save processed data to files."""
    if output_dir is None:
        output_dir = os.path.join(DATA_DIR, "processed")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n💾 Saving processed data to {output_dir}...")
    
    # Save as JSON
    data = {
        "train": {
            "queries": train_q,
            "documents": train_d,
            "labels": train_l,
        },
        "test": {
            "queries": test_q,
            "documents": test_d,
            "labels": test_l,
        }
    }
    
    output_file = os.path.join(output_dir, "msmarco_sample.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"  ✓ Saved to {output_file}")
    
    # Also save as TSV for easy viewing
    train_file = os.path.join(output_dir, "train.tsv")
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write("query\tdocument\tlabel\n")
        for q, d, l in zip(train_q, train_d, train_l):
            f.write(f"{q}\t{d}\t{l}\n")
    
    test_file = os.path.join(output_dir, "test.tsv")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("query\tdocument\tlabel\n")
        for q, d, l in zip(test_q, test_d, test_l):
            f.write(f"{q}\t{d}\t{l}\n")
    
    print(f"  ✓ Saved TSV files")
    
    return output_file


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("🐟 MS MARCO Dataset Downloader for markrel")
    print("="*60)
    
    # Download files
    if not download_msmarco_files():
        print("\n✗ Failed to download files. Exiting.")
        sys.exit(1)
    
    # Create sample dataset
    train_q, train_d, train_l, test_q, test_d, test_l = create_sample_dataset(
        n_train=5000,
        n_test=1000,
        n_negatives_per_positive=5
    )
    
    # Save processed data
    output_file = save_processed_data(
        train_q, train_d, train_l,
        test_q, test_d, test_l
    )
    
    print("\n" + "="*60)
    print("✅ MS MARCO data ready for benchmarking!")
    print("="*60)
    print(f"\n📁 Processed data: {output_file}")
    print(f"\n🚀 Next step: Run experiments/run_msmarco_benchmark.py")


if __name__ == "__main__":
    main()
