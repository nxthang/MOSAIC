#!/usr/bin/env python3
"""
MOSAIC - Main Training Script
=============================================
Train and evaluate MOSAIC and baseline models for cultural alignment.

Usage:
    python main.py --model mosaic --config configs/mosaic_config.yaml
    python main.py --model dpo --config configs/dpo_config.yaml
    python main.py --model rlhf --config configs/rlhf_config.yaml
    python main.py --model cultural_ft --config configs/cultural_ft_config.yaml
    python main.py --model soft_prompt --config configs/soft_prompt_config.yaml

Author: Thang Nguyen Xuan
Institution: Hanoi University, Vietnam

"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mosaic.mosaic_trainer import MOSAICTrainer, MOSAICConfig
from baselines.dpo_trainer import DPOTrainer, DPOConfig
from baselines.rlhf_trainer import RLHFTrainer, RLHFConfig
from baselines.cultural_finetuning import CulturalFineTuningTrainer, CulturalFTConfig
from baselines.soft_prompt_tuning import SoftPromptTrainer, SoftPromptConfig
from utils.data_loader import (
    CulturalDataConfig,
    CulturalPreferenceDataset,
    CombinedCulturalDataset,
    load_culturepark_and_normad
)
from evaluation.metrics import CulturalEquilibriumEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config or {}


def update_dataclass_from_dict(dataclass_obj, config_dict: Dict[str, Any]):
    """Update dataclass object from dictionary."""
    for key, value in config_dict.items():
        if hasattr(dataclass_obj, key):
            setattr(dataclass_obj, key, value)
    return dataclass_obj


def train_mosaic(args, config: Dict[str, Any]) -> Dict[str, Any]:
    """Train MOSAIC model."""
    logger.info("=" * 60)
    logger.info("Training MOSAIC (Multicultural Optimization Alignment Framework)")
    logger.info("=" * 60)
    
    # Initialize config
    mosaic_config = MOSAICConfig()
    mosaic_config = update_dataclass_from_dict(mosaic_config, config.get('mosaic', {}))
    
    # Initialize trainer
    trainer = MOSAICTrainer(mosaic_config)
    
    # Load datasets
    data_config = CulturalDataConfig()
    train_dataset, eval_dataset = load_culturepark_and_normad(
        culturepark_path=args.culturepark_data,
        normad_path=args.normad_data,
        config=data_config
    )
    
    # Train
    results = trainer.train(train_dataset, eval_dataset)
    
    # Save model
    if args.output_dir:
        trainer.save_model(os.path.join(args.output_dir, "mosaic"))
    
    return results


def train_dpo(args, config: Dict[str, Any]) -> Dict[str, Any]:
    """Train DPO baseline."""
    logger.info("=" * 60)
    logger.info("Training DPO (Direct Preference Optimization)")
    logger.info("=" * 60)
    
    # Initialize config
    dpo_config = DPOConfig()
    dpo_config = update_dataclass_from_dict(dpo_config, config.get('dpo', {}))
    
    # Initialize trainer
    trainer = DPOTrainer(dpo_config)
    
    # Load datasets
    data_config = CulturalDataConfig()
    train_dataset = CulturalPreferenceDataset(
        args.culturepark_data,
        data_config,
        split="train"
    )
    
    eval_dataset = CulturalPreferenceDataset(
        args.culturepark_data,
        data_config,
        split="validation"
    ) if args.eval_data else None
    
    # Train
    results = trainer.train(train_dataset, eval_dataset)
    
    # Save model
    if args.output_dir:
        trainer.save_model(os.path.join(args.output_dir, "dpo"))
    
    return results


def train_rlhf(args, config: Dict[str, Any]) -> Dict[str, Any]:
    """Train RLHF baseline."""
    logger.info("=" * 60)
    logger.info("Training RLHF (Reinforcement Learning from Human Feedback)")
    logger.info("=" * 60)
    
    # Initialize config
    rlhf_config = RLHFConfig()
    rlhf_config = update_dataclass_from_dict(rlhf_config, config.get('rlhf', {}))
    
    # Initialize trainer
    trainer = RLHFTrainer(rlhf_config)
    
    # Load datasets
    data_config = CulturalDataConfig()
    train_dataset = CulturalPreferenceDataset(
        args.culturepark_data,
        data_config,
        split="train"
    )
    
    # Train reward model
    reward_results = trainer.train_reward_model(train_dataset)
    logger.info(f"Reward model training complete: {reward_results}")
    
    # Train policy with PPO
    policy_results = trainer.train_policy(train_dataset)
    
    # Save model
    if args.output_dir:
        trainer.save_model(os.path.join(args.output_dir, "rlhf"))
    
    return {
        "reward_results": reward_results,
        "policy_results": policy_results
    }


def train_cultural_ft(args, config: Dict[str, Any]) -> Dict[str, Any]:
    """Train Cultural Fine-tuning baseline."""
    logger.info("=" * 60)
    logger.info("Training Cultural Fine-tuning")
    logger.info("=" * 60)
    
    # Initialize config
    cultural_ft_config = CulturalFTConfig()
    cultural_ft_config = update_dataclass_from_dict(cultural_ft_config, config.get('cultural_ft', {}))
    
    # Initialize trainer
    trainer = CulturalFineTuningTrainer(cultural_ft_config)
    
    # Load datasets
    data_config = CulturalDataConfig()
    train_dataset = CulturalPreferenceDataset(
        args.culturepark_data,
        data_config,
        split="train"
    )
    
    eval_dataset = CulturalPreferenceDataset(
        args.eval_data or args.culturepark_data,
        data_config,
        split="validation"
    ) if args.eval_data else None
    
    # Train
    results = trainer.train(train_dataset, eval_dataset)
    
    # Save model
    if args.output_dir:
        trainer.save_model(os.path.join(args.output_dir, "cultural_ft"))
    
    return results


def train_soft_prompt(args, config: Dict[str, Any]) -> Dict[str, Any]:
    """Train Soft Prompt Tuning baseline."""
    logger.info("=" * 60)
    logger.info("Training Soft Prompt Tuning")
    logger.info("=" * 60)
    
    # Initialize config
    soft_prompt_config = SoftPromptConfig()
    soft_prompt_config = update_dataclass_from_dict(soft_prompt_config, config.get('soft_prompt', {}))
    
    # Initialize trainer
    trainer = SoftPromptTrainer(soft_prompt_config)
    
    # Load datasets
    data_config = CulturalDataConfig()
    train_dataset = CulturalPreferenceDataset(
        args.culturepark_data,
        data_config,
        split="train"
    )
    
    eval_dataset = CulturalPreferenceDataset(
        args.eval_data or args.culturepark_data,
        data_config,
        split="validation"
    ) if args.eval_data else None
    
    # Train
    results = trainer.train(train_dataset, eval_dataset)
    
    # Save model
    if args.output_dir:
        trainer.save_model(os.path.join(args.output_dir, "soft_prompt"))
    
    return results


def evaluate(args, config: Dict[str, Any]):
    """Evaluate trained models."""
    logger.info("=" * 60)
    logger.info("Evaluating Models")
    logger.info("=" * 60)
    
    evaluator = CulturalEquilibriumEvaluator()
    
    # Load evaluation data
    # (Implementation depends on data format)
    
    # Generate responses from models
    # (Implementation depends on model format)
    
    # Compute metrics
    # results = evaluator.evaluate(...)
    
    logger.info("Evaluation complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Cultural Equilibrium - Train and evaluate cultural alignment models"
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mosaic", "dpo", "rlhf", "cultural_ft", "soft_prompt"],
        help="Model to train"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    # Data paths
    parser.add_argument(
        "--culturepark-data",
        type=str,
        required=True,
        help="Path to CulturePark dataset"
    )
    parser.add_argument(
        "--normad-data",
        type=str,
        default=None,
        help="Path to NORMAD dataset"
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to evaluation dataset"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for models and results"
    )
    
    # Mode
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only evaluate, don't train"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Save config to output
    with open(os.path.join(args.output_dir, "config_used.yaml"), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Select training function
    train_functions = {
        "mosaic": train_mosaic,
        "dpo": train_dpo,
        "rlhf": train_rlhf,
        "cultural_ft": train_cultural_ft,
        "soft_prompt": train_soft_prompt
    }
    
    if args.evaluate_only:
        evaluate(args, config)
    else:
        train_fn = train_functions[args.model]
        results = train_fn(args, config)
        
        # Save results
        results_path = os.path.join(args.output_dir, f"{args.model}_results.json")
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert(obj):
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(v) for v in obj]
                elif isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            json.dump(convert(results), f, indent=2)
        
        logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    import numpy as np
    main()
