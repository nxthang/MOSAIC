"""
RLHF Baseline Implementation
=============================
Standard Reinforcement Learning from Human Feedback for cultural alignment.

This implements the classic RLHF pipeline:
1. Supervised Fine-Tuning (SFT)
2. Reward Model Training
3. PPO Optimization

Author: Thang Nguyen Xuan
Institution: Hanoi University, Vietnam
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    Trainer,
    TrainingArguments
)
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RLHFConfig:
    """Configuration for RLHF training."""
    
    # Model
    model_name: str = "meta-llama/Llama-2-13b-hf"
    
    # SFT parameters
    sft_learning_rate: float = 2e-5
    sft_epochs: int = 3
    sft_batch_size: int = 4
    
    # Reward model parameters
    reward_learning_rate: float = 1e-5
    reward_epochs: int = 2
    reward_batch_size: int = 8
    
    # PPO parameters
    ppo_learning_rate: float = 1e-6
    ppo_epochs: int = 2
    ppo_batch_size: int = 4
    ppo_clip_epsilon: float = 0.2
    ppo_entropy_coef: float = 0.01
    
    # General
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class RewardModel(nn.Module):
    """
    Reward model for scoring responses based on cultural alignment.
    """
    
    def __init__(self, config: RLHFConfig, num_labels: int = 1):
        super().__init__()
        self.config = config
        
        # Load base model for reward scoring
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=num_labels,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reward score for a response.
        
        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] attention mask
            
        Returns:
            rewards: [batch, 1] reward scores
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        rewards = outputs.logits  # [batch, 1]
        return rewards
    
    def compute_reward_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute pairwise ranking loss for reward model training.
        
        Args:
            chosen_input_ids: Preferred responses
            chosen_attention_mask: Attention masks for chosen
            rejected_input_ids: Non-preferred responses
            rejected_attention_mask: Attention masks for rejected
            
        Returns:
            loss: Ranking loss
            metrics: Dictionary of metrics
        """
        # Get rewards
        rewards_chosen = self.forward(chosen_input_ids, chosen_attention_mask).squeeze(-1)
        rewards_rejected = self.forward(rejected_input_ids, rejected_attention_mask).squeeze(-1)
        
        # Pairwise ranking loss
        loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()
        
        # Accuracy: how often chosen is ranked higher
        accuracy = (rewards_chosen > rewards_rejected).float().mean()
        
        metrics = {
            "reward_loss": loss.item(),
            "ranking_accuracy": accuracy.item(),
            "mean_reward_chosen": rewards_chosen.mean().item(),
            "mean_reward_rejected": rewards_rejected.mean().item()
        }
        
        return loss, metrics


class PPOTrainer:
    """
    PPO trainer for RLHF policy optimization.
    """
    
    def __init__(
        self,
        policy_model: PreTrainedModel,
        reward_model: RewardModel,
        ref_model: PreTrainedModel,
        config: RLHFConfig
    ):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.ref_model = ref_model  # Reference model (frozen)
        self.config = config
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            policy_model.parameters(),
            lr=config.ppo_learning_rate
        )
        
    def compute_kl_divergence(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between policy and reference."""
        policy_probs = F.log_softmax(policy_logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)
        kl_div = (ref_probs * (ref_probs.log() - policy_probs)).sum(dim=-1)
        return kl_div.mean()
    
    def ppo_step(
        self,
        batch: Dict[str, torch.Tensor],
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform one PPO update step.
        
        Args:
            batch: Input batch
            old_log_probs: Log probabilities from old policy
            advantages: Advantage estimates
            
        Returns:
            metrics: Training metrics
        """
        # Get current policy outputs
        outputs = self.policy_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        logits = outputs.logits
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Compute importance weights
        ratio = torch.exp(log_probs - old_log_probs)
        
        # PPO clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip_epsilon, 
                           1 + self.config.ppo_clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy bonus
        entropy = -(F.softmax(logits, dim=-1) * log_probs).sum(dim=-1).mean()
        
        # KL penalty
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
        kl_div = self.compute_kl_divergence(logits, ref_outputs.logits)
        
        # Total loss
        loss = (
            policy_loss - 
            self.config.ppo_entropy_coef * entropy +
            kl_div  # KL penalty
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "kl_divergence": kl_div.item(),
            "total_loss": loss.item()
        }
        
        return metrics
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        reward_fn
    ) -> Dict[str, float]:
        """
        Train for one epoch using PPO.
        
        Args:
            dataloader: Data loader
            reward_fn: Function to compute rewards
            
        Returns:
            metrics: Average metrics for the epoch
        """
        self.policy_model.train()
        all_metrics = []
        
        for batch in dataloader:
            batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Generate responses (simplified - in practice use sampling)
            with torch.no_grad():
                outputs = self.policy_model.generate(
                    batch["input_ids"],
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7
                )
            
            # Compute rewards
            with torch.no_grad():
                rewards = reward_fn(outputs)
            
            # Compute advantages (simplified - just use rewards)
            advantages = rewards - rewards.mean()
            
            # Get old log probabilities
            with torch.no_grad():
                old_outputs = self.policy_model(
                    input_ids=outputs,
                    attention_mask=(outputs != self.policy_model.config.pad_token_id).long()
                )
                old_log_probs = F.log_softmax(old_outputs.logits, dim=-1)
            
            # PPO update
            metrics = self.ppo_step(batch, old_log_probs, advantages)
            metrics["mean_reward"] = rewards.mean().item()
            all_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics


class RLHFTrainer:
    """
    Main trainer for RLHF pipeline.
    """
    
    def __init__(self, config: RLHFConfig):
        self.config = config
        
        # Load tokenizer and models
        logger.info(f"Loading models: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Policy model
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32
        )
        self.policy_model.to(config.device)
        
        # Reference model (frozen)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32
        )
        self.ref_model.to(config.device)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Reward model
        self.reward_model = RewardModel(config)
        self.reward_model.to(config.device)
        
        logger.info("RLHF Trainer initialized")
    
    def train_reward_model(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ) -> Dict[str, float]:
        """Train the reward model."""
        logger.info("Training reward model...")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.reward_batch_size,
            shuffle=True,
            collate_fn=self._reward_collate_fn
        )
        
        optimizer = torch.optim.AdamW(
            self.reward_model.parameters(),
            lr=self.config.reward_learning_rate
        )
        
        self.reward_model.train()
        for epoch in range(self.config.reward_epochs):
            epoch_losses = []
            for batch in train_loader:
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                
                loss, metrics = self.reward_model.compute_reward_loss(
                    chosen_input_ids=batch["chosen_input_ids"],
                    chosen_attention_mask=batch["chosen_attention_mask"],
                    rejected_input_ids=batch["rejected_input_ids"],
                    rejected_attention_mask=batch["rejected_attention_mask"]
                )
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_losses.append(metrics["reward_loss"])
            
            logger.info(f"Reward Model Epoch {epoch+1}/{self.config.reward_epochs} - Loss: {np.mean(epoch_losses):.4f}")
        
        return {"final_reward_loss": np.mean(epoch_losses)}
    
    def train_policy(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ) -> Dict[str, Any]:
        """Train policy using PPO."""
        logger.info("Training policy with PPO...")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.ppo_batch_size,
            shuffle=True,
            collate_fn=self._policy_collate_fn
        )
        
        ppo_trainer = PPOTrainer(
            self.policy_model,
            self.reward_model,
            self.ref_model,
            self.config
        )
        
        # Define reward function
        def reward_fn(outputs):
            with torch.no_grad():
                rewards = self.reward_model(
                    input_ids=outputs,
                    attention_mask=(outputs != self.tokenizer.pad_token_id).long()
                )
            return rewards.squeeze(-1)
        
        # Train
        all_metrics = []
        for epoch in range(self.config.ppo_epochs):
            metrics = ppo_trainer.train_epoch(train_loader, reward_fn)
            all_metrics.append(metrics)
            logger.info(f"PPO Epoch {epoch+1}/{self.config.ppo_epochs} - {metrics}")
        
        return {
            "ppo_metrics": all_metrics,
            "final_metrics": all_metrics[-1] if all_metrics else None
        }
    
    def _reward_collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function for reward model training."""
        return {
            "chosen_input_ids": torch.stack([item["chosen_input_ids"] for item in batch]),
            "chosen_attention_mask": torch.stack([item["chosen_attention_mask"] for item in batch]),
            "rejected_input_ids": torch.stack([item["rejected_input_ids"] for item in batch]),
            "rejected_attention_mask": torch.stack([item["rejected_attention_mask"] for item in batch])
        }
    
    def _policy_collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function for policy training."""
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch])
        }
    
    def save_model(self, output_dir: str):
        """Save trained models."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        self.policy_model.save_pretrained(os.path.join(output_dir, "policy"))
        self.reward_model.save_pretrained(os.path.join(output_dir, "reward"))
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Models saved to {output_dir}")


if __name__ == "__main__":
    config = RLHFConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        ppo_epochs=1
    )
    trainer = RLHFTrainer(config)
    print("RLHF Trainer ready!")
