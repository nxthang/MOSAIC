"""
Data Loading Utilities
=======================
Dataset loaders for MOSAIC training and evaluation.

Supports:
- CulturePark dataset
- NORMAD dataset
- Custom preference datasets

Author: Thang Nguyen Xuan
Institution: Hanoi University, Vietnam
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional, Any
import json
import os
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CulturalDataConfig:
    """Configuration for data loading."""
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf"
    max_seq_length: int = 512
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    cultural_context_column: str = "cultural_context"


class CulturalPreferenceDataset(Dataset):
    """
    Dataset for cultural preference learning.
    
    Loads preference pairs (chosen, rejected) with cultural context annotations.
    """
    
    def __init__(
        self,
        data_path: str,
        config: CulturalDataConfig,
        split: str = "train"
    ):
        self.config = config
        self.data_path = data_path
        self.split = split
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load data
        self.data = self._load_data(data_path, split)
        
        logger.info(f"Loaded {len(self.data)} examples from {data_path} ({split})")
    
    def _load_data(self, data_path: str, split: str) -> List[Dict]:
        """Load data from various formats."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        # Support multiple formats
        if data_path.endswith('.jsonl'):
            return self._load_jsonl(data_path)
        elif data_path.endswith('.json'):
            return self._load_json(data_path, split)
        elif data_path.endswith('.csv'):
            return self._load_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    
    def _load_jsonl(self, data_path: str) -> List[Dict]:
        """Load JSONL format."""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def _load_json(self, data_path: str, split: str) -> List[Dict]:
        """Load JSON format with train/val/test splits."""
        with open(data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        if isinstance(all_data, dict):
            return all_data.get(split, [])
        else:
            return all_data
    
    def _load_csv(self, data_path: str) -> List[Dict]:
        """Load CSV format."""
        import csv
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        example = self.data[idx]
        
        # Extract fields
        prompt = example.get(self.config.prompt_column, "")
        chosen = example.get(self.config.chosen_column, "")
        rejected = example.get(self.config.rejected_column, "")
        cultural_context = example.get(self.config.cultural_context_column, 0)
        
        # Tokenize chosen response
        chosen_encoding = self.tokenizer(
            chosen,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding='max_length'
        )
        
        # Tokenize rejected response
        rejected_encoding = self.tokenizer(
            rejected,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding='max_length'
        )
        
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "input_ids_chosen": torch.tensor(chosen_encoding["input_ids"]),
            "attention_mask_chosen": torch.tensor(chosen_encoding["attention_mask"]),
            "input_ids_rejected": torch.tensor(rejected_encoding["input_ids"]),
            "attention_mask_rejected": torch.tensor(rejected_encoding["attention_mask"]),
            "cultural_context": int(cultural_context)
        }


class CultureParkDataset(CulturalPreferenceDataset):
    """
    CulturePark dataset for cultural alignment evaluation.
    
    Reference: Wang et al. (2024). CulturePark: A benchmark for cross-cultural 
    reasoning in large language models. EMNLP.
    """
    
    def __init__(
        self,
        data_path: str,
        config: CulturalDataConfig,
        split: str = "validation",
        subset: Optional[str] = None
    ):
        self.subset = subset
        super().__init__(data_path, config, split)
    
    def _load_data(self, data_path: str, split: str) -> List[Dict]:
        """Load CulturePark format."""
        data = super()._load_data(data_path, split)
        
        # Filter by subset if specified
        if self.subset is not None:
            data = [d for d in data if d.get("subset") == self.subset]
        
        logger.info(f"CulturePark {split} subset: {len(data)} examples")
        return data


class NORMADDataset(CulturalPreferenceDataset):
    """
    NORMAD dataset for sociocultural alignment evaluation.
    
    Reference: Arora et al. (2024). NORMAD: Norms and morals across dialects 
    for sociocultural alignment evaluation. arXiv:2404.12464.
    """
    
    def __init__(
        self,
        data_path: str,
        config: CulturalDataConfig,
        split: str = "test",
        countries: Optional[List[str]] = None,
        max_samples: Optional[int] = None
    ):
        self.countries = countries
        self.max_samples = max_samples
        super().__init__(data_path, config, split)
    
    def _load_data(self, data_path: str, split: str) -> List[Dict]:
        """Load NORMAD format."""
        data = super()._load_data(data_path, split)
        
        # Filter by countries if specified
        if self.countries is not None:
            data = [d for d in data if d.get("country") in self.countries]
        
        # Limit samples if specified
        if self.max_samples is not None:
            data = data[:self.max_samples]
        
        logger.info(f"NORMAD {split}: {len(data)} examples")
        return data


class CombinedCulturalDataset(Dataset):
    """
    Combined dataset from multiple sources.
    """
    
    def __init__(
        self,
        datasets: List[CulturalPreferenceDataset],
        weights: Optional[List[float]] = None
    ):
        self.datasets = datasets
        self.weights = weights or [1.0] * len(datasets)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        # Calculate cumulative lengths
        self.cumulative_lengths = []
        total = 0
        for dataset in datasets:
            total += len(dataset)
            self.cumulative_lengths.append(total)
        
        logger.info(f"Combined dataset: {total} examples from {len(datasets)} sources")
    
    def __len__(self) -> int:
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find which dataset this index belongs to
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                if i == 0:
                    actual_idx = idx
                else:
                    actual_idx = idx - self.cumulative_lengths[i - 1]
                return self.datasets[i][actual_idx]
        
        # Fallback to last dataset
        return self.datasets[-1][idx]


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn: Optional[callable] = None
) -> DataLoader:
    """
    Create a DataLoader for cultural preference datasets.
    
    Args:
        dataset: The dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        collate_fn: Custom collate function
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


def default_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Default collate function for cultural preference datasets.
    
    Args:
        batch: List of examples
        
    Returns:
        Collated batch as dictionary of tensors
    """
    return {
        "input_ids_chosen": torch.stack([item["input_ids_chosen"] for item in batch]),
        "attention_mask_chosen": torch.stack([item["attention_mask_chosen"] for item in batch]),
        "input_ids_rejected": torch.stack([item["input_ids_rejected"] for item in batch]),
        "attention_mask_rejected": torch.stack([item["attention_mask_rejected"] for item in batch]),
        "cultural_context": torch.tensor([item["cultural_context"] for item in batch])
    }


def load_culturepark_and_normad(
    culturepark_path: str,
    normad_path: str,
    config: CulturalDataConfig,
    normad_max_samples: int = 5000
) -> Tuple[CombinedCulturalDataset, CombinedCulturalDataset]:
    """
    Load combined CulturePark and NORMAD datasets.
    
    Args:
        culturepark_path: Path to CulturePark data
        normad_path: Path to NORMAD data
        config: Data configuration
        normad_max_samples: Maximum samples from NORMAD
        
    Returns:
        train_dataset, eval_dataset
    """
    # Load training datasets
    cp_train = CultureParkDataset(culturepark_path, config, split="train")
    normad_train = NORMADDataset(
        normad_path, 
        config, 
        split="train",
        max_samples=normad_max_samples
    )
    
    # Load evaluation datasets
    cp_eval = CultureParkDataset(culturepark_path, config, split="validation")
    normad_eval = NORMADDataset(
        normad_path, 
        config, 
        split="test",
        max_samples=normad_max_samples // 2
    )
    
    # Combine datasets
    train_dataset = CombinedCulturalDataset(
        [cp_train, normad_train],
        weights=[0.4, 0.6]  # Weight NORMAD more heavily
    )
    
    eval_dataset = CombinedCulturalDataset(
        [cp_eval, normad_eval]
    )
    
    return train_dataset, eval_dataset


if __name__ == "__main__":
    # Example usage
    config = CulturalDataConfig()
    
    print("Data loader utilities loaded successfully!")
    print("Supported datasets: CulturePark, NORMAD, Custom JSON/JSONL/CSV")
