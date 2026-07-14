"""
DPO Baseline Implementation
============================
Direct Preference Optimization for cultural alignment.

Reference: Rafailov et al. (2024). Direct Preference Optimization: 
Your Language Model is Secretly a Reward Model. NeurIPS.

Author: Thang Nguyen Xuan
Institution: Hanoi University, Vietnam
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    
    # Model
    model_name: str = "meta-llama/Llama-2-13b-hf"
    
    # DPO parameters
    beta: float = 0.1  # Temperature parameter for DPO
    learning_rate: float = 5e-7
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    
    # Training
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_seq_length: int = 512
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DPOTrainer:
    """
    Trainer for Direct Preference Optimization.
    
    DPO directly optimizes the policy to satisfy preferences without
    explicitly training a reward model.
    """
    
    def __init__(self, config: DPOConfig):
        self.config = config
        
        # Load tokenizer and model
        logger.info(f"Loading model: {config.model_name}")
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
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.learning_rate
        )
        
        logger.info("DPO Trainer initialized")
    
    def concatenated_forward(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for concatenated chosen and rejected inputs.
        
        Args:
            model: The model to forward through
            input_ids: Concatenated input IDs [2*batch, seq_len]
            attention_mask: Concatenated attention mask
            labels: Optional labels
            
        Returns:
            all_logps: Log probabilities for each token
            all_losses: Per-example losses
        """
        # Run forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs.logits
        
        # Shift for next token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_logps = torch.gather(
            log_probs, 
            dim=2, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Average over sequence length
        sequence_logps = (token_logps * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)
        
        # Split into chosen and rejected
        batch_size = input_ids.shape[0] // 2
        all_logps = sequence_logps
        all_losses = -sequence_logps  # Negative log likelihood
        
        return all_logps, all_losses
    
    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss.
        
        The DPO loss is:
        L_DPO = -log σ(β * (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x))))
        
        where:
        - y_w: chosen (winning) response
        - y_l: rejected (losing) response
        - β: temperature parameter
        
        Args:
            policy_chosen_logps: Log probs of chosen under policy
            policy_rejected_logps: Log probs of rejected under policy
            reference_chosen_logps: Log probs of chosen under reference
            reference_rejected_logps: Log probs of rejected under reference
            
        Returns:
            loss: DPO loss
            metrics: Dictionary of metrics
        """
        # Compute log odds ratios
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        
        # DPO loss
        logits = self.config.beta * (chosen_logratios - rejected_logratios)
        losses = -F.logsigmoid(logits)
        
        loss = losses.mean()
        
        # Metrics
        chosen_rewards = self.config.beta * chosen_logratios.detach()
        rejected_rewards = self.config.beta * rejected_logratios.detach()
        reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
        reward_margin = (chosen_rewards - rejected_rewards).mean()
        
        metrics = {
            "dpo_loss": loss.item(),
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item(),
            "reward_accuracy": reward_accuracy.item(),
            "reward_margin": reward_margin.item(),
            "rewards_chosen_std": chosen_rewards.std().item(),
            "rewards_rejected_std": rejected_rewards.std().item()
        }
        
        return loss, metrics
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch: Dictionary containing:
                - chosen_input_ids, chosen_attention_mask
                - rejected_input_ids, rejected_attention_mask
                
        Returns:
            metrics: Training metrics
        """
        self.policy_model.train()
        
        # Concatenate chosen and rejected for efficient processing
        input_ids = torch.cat([
            batch["chosen_input_ids"],
            batch["rejected_input_ids"]
        ], dim=0)
        attention_mask = torch.cat([
            batch["chosen_attention_mask"],
            batch["rejected_attention_mask"]
        ], dim=0)
        
        # Forward pass through policy model
        policy_all_logps, policy_all_losses = self.concatenated_forward(
            self.policy_model, input_ids, attention_mask
        )
        
        # Forward pass through reference model (no grad)
        with torch.no_grad():
            ref_all_logps, ref_all_losses = self.concatenated_forward(
                self.ref_model, input_ids, attention_mask
            )
        
        # Split into chosen and rejected
        batch_size = batch["chosen_input_ids"].shape[0]
        policy_chosen_logps = policy_all_logps[:batch_size]
        policy_rejected_logps = policy_all_logps[batch_size:]
        ref_chosen_logps = ref_all_logps[:batch_size]
        ref_rejected_logps = ref_all_logps[batch_size:]
        
        # Compute DPO loss
        loss, metrics = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation step
        if self._step_count % self.config.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        self._step_count += 1
        
        return metrics
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ) -> Dict[str, Any]:
        """
        Train the model using DPO.
        
        Args:
            train_dataset: Training dataset with preference pairs
            eval_dataset: Optional evaluation dataset
            
        Returns:
            training_results: Dictionary with training results
        """
        logger.info("Starting DPO training...")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        self._step_count = 0
        all_epoch_metrics = []
        
        for epoch in range(self.config.num_epochs):
            self.policy_model.train()
            epoch_metrics = []
            
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Training step
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
                
                # Log progress
                if (batch_idx + 1) % 100 == 0:
                    avg_metrics = {k: np.mean([m[k] for m in epoch_metrics[-100:]]) 
                                  for k in metrics.keys()}
                    logger.info(f"Epoch {epoch+1}, Step {batch_idx+1} - "
                               f"Loss: {avg_metrics['dpo_loss']:.4f}, "
                               f"Accuracy: {avg_metrics['reward_accuracy']:.4f}")
            
            # Average epoch metrics
            avg_epoch_metrics = {k: np.mean([m[k] for m in epoch_metrics]) 
                                for k in epoch_metrics[0].keys()}
            all_epoch_metrics.append(avg_epoch_metrics)
            
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} completed - "
                       f"Loss: {avg_epoch_metrics['dpo_loss']:.4f}")
        
        # Evaluate if eval dataset provided
        eval_results = None
        if eval_dataset is not None:
            eval_results = self.evaluate(eval_dataset)
        
        return {
            "epoch_metrics": all_epoch_metrics,
            "final_metrics": all_epoch_metrics[-1] if all_epoch_metrics else None,
            "eval_results": eval_results
        }
    
    def evaluate(
        self,
        eval_dataset: Dataset
    ) -> Dict[str, float]:
        """Evaluate the model on a dataset."""
        logger.info("Evaluating DPO model...")
        
        self.policy_model.eval()
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self._collate_fn
        )
        
        all_metrics = []
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                input_ids = torch.cat([
                    batch["chosen_input_ids"],
                    batch["rejected_input_ids"]
                ], dim=0)
                attention_mask = torch.cat([
                    batch["chosen_attention_mask"],
                    batch["rejected_attention_mask"]
                ], dim=0)
                
                policy_all_logps, _ = self.concatenated_forward(
                    self.policy_model, input_ids, attention_mask
                )
                
                with torch.no_grad():
                    ref_all_logps, _ = self.concatenated_forward(
                        self.ref_model, input_ids, attention_mask
                    )
                
                batch_size = batch["chosen_input_ids"].shape[0]
                policy_chosen_logps = policy_all_logps[:batch_size]
                policy_rejected_logps = policy_all_logps[batch_size:]
                ref_chosen_logps = ref_all_logps[:batch_size]
                ref_rejected_logps = ref_all_logps[batch_size:]
                
                _, metrics = self.dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps
                )
                all_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) 
                      for k in all_metrics[0].keys()}
        
        return avg_metrics
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function for data loader."""
        return {
            "chosen_input_ids": torch.stack([item["chosen_input_ids"] for item in batch]),
            "chosen_attention_mask": torch.stack([item["chosen_attention_mask"] for item in batch]),
            "rejected_input_ids": torch.stack([item["rejected_input_ids"] for item in batch]),
            "rejected_attention_mask": torch.stack([item["rejected_attention_mask"] for item in batch])
        }
    
    def save_model(self, output_dir: str):
        """Save the trained model."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        self.policy_model.save_pretrained(os.path.join(output_dir, "dpo_policy"))
        self.tokenizer.save_pretrained(output_dir)
        
        # Save config
        import json
        with open(os.path.join(output_dir, "dpo_config.json"), "w") as f:
            json.dump(vars(self.config), f, indent=2)
        
        logger.info(f"DPO model saved to {output_dir}")


if __name__ == "__main__":
    config = DPOConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        num_epochs=1,
        batch_size=2
    )
    trainer = DPOTrainer(config)
    print("DPO Trainer ready for training!")
