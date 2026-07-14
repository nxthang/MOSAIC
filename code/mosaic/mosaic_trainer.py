"""
MOSAIC: Multicultural Optimization through Strategic Agents and Iterative Consensus
===============================================
Game-theoretic approach to cultural alignment of LLMs using Nash Equilibrium.

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
class MOSAICConfig:
    """Configuration for MOSAIC training."""
    
    # Model
    model_name: str = "meta-llama/Llama-2-13b-hf"
    
    # Cultural contexts
    num_cultural_contexts: int = 12  # Number of cultural clusters
    
    # Game-theoretic parameters
    equilibrium_iterations: int = 500  # Iterations for Nash equilibrium computation
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    
    # Loss weights
    cultural_alignment_weight: float = 1.0
    safety_weight: float = 0.5
    bias_penalty_weight: float = 0.3
    
    # Training
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_seq_length: int = 512

    # Ablation: enable/disable Nash Bargaining Solution integration
    use_bargaining: bool = True

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CulturalUtilityFunction(nn.Module):
    """
    Utility function for a specific cultural context.
    
    Models the preferences and values of a cultural group as a learnable function.
    """
    
    def __init__(self, config: MOSAICConfig, context_id: int, hidden_dim: int = 768):
        super().__init__()
        self.context_id = context_id
        self.config = config
        
        # Utility network
        self.utility_network = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Scalar utility score
        )
        
        # Cultural-specific parameters (Hofstede dimensions encoding)
        self.cultural_embedding = nn.Embedding(config.num_cultural_contexts, hidden_dim)
        
    def forward(self, response_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute utility score for a response given cultural context.
        
        Args:
            response_embeddings: [batch_size, hidden_dim] encoded response representations
            
        Returns:
            utility_scores: [batch_size, 1] utility values
        """
        # Get cultural context embedding
        context_emb = self.cultural_embedding(
            torch.tensor([self.context_id], device=response_embeddings.device)
        )
        
        # Combine response and cultural context
        combined = response_embeddings + context_emb
        
        # Compute utility
        utility = self.utility_network(combined)
        return utility


class PairwisePreferenceModel(nn.Module):
    """
    Pairwise preference model for cultural alignment.
    
    Learns P(y ≻ y' | x, c) without explicit reward modeling.
    """
    
    def __init__(self, config: MOSAICConfig, base_model: PreTrainedModel):
        super().__init__()
        self.config = config
        self.base_model = base_model
        
        # Freeze base model parameters (optional, can be fine-tuned)
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Cultural utility functions for each context
        self.cultural_utilities = nn.ModuleList([
            CulturalUtilityFunction(config, i, hidden_dim=base_model.config.hidden_size)
            for i in range(config.num_cultural_contexts)
        ])
        
        # Context weights (learnable importance of each cultural context)
        self.context_weights = nn.Parameter(torch.ones(config.num_cultural_contexts))
        
    def get_response_embeddings(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract response embeddings from the model."""
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # Use last hidden state mean pooling
            last_hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden]
            embeddings = (last_hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        return embeddings
    
    def compute_pairwise_preference(
        self,
        input_ids_y: torch.Tensor,
        attention_mask_y: torch.Tensor,
        input_ids_y_prime: torch.Tensor,
        attention_mask_y_prime: torch.Tensor,
        context_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute pairwise preference P(y ≻ y').
        
        Args:
            input_ids_y: Token IDs for response y
            attention_mask_y: Attention mask for y
            input_ids_y_prime: Token IDs for response y'
            attention_mask_y_prime: Attention mask for y'
            context_weights: Optional custom context weights
            
        Returns:
            preference_prob: Probability that y is preferred over y'
        """
        # Get embeddings
        emb_y = self.get_response_embeddings(input_ids_y, attention_mask_y)
        emb_y_prime = self.get_response_embeddings(input_ids_y_prime, attention_mask_y_prime)
        
        # Compute utilities for each cultural context
        weights = context_weights if context_weights is not None else F.softmax(self.context_weights, dim=0)
        
        utility_diff = 0.0
        for i, utility_fn in enumerate(self.cultural_utilities):
            u_y = utility_fn(emb_y).squeeze(-1)  # [batch]
            u_y_prime = utility_fn(emb_y_prime).squeeze(-1)  # [batch]
            utility_diff += weights[i] * (u_y - u_y_prime)
        
        # Sigmoid to get probability
        preference_prob = torch.sigmoid(utility_diff)
        return preference_prob
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass for MOSAIC training.
        
        Args:
            batch: Dictionary containing:
                - input_ids_chosen: [batch, seq_len] preferred responses
                - attention_mask_chosen: [batch, seq_len]
                - input_ids_rejected: [batch, seq_len] non-preferred responses
                - attention_mask_rejected: [batch, seq_len]
                - cultural_context: [batch] cultural context IDs
                
        Returns:
            loss: Total loss
            metrics: Dictionary of metrics
        """
        # Compute pairwise preferences
        pref_prob = self.compute_pairwise_preference(
            input_ids_y=batch["input_ids_chosen"],
            attention_mask_y=batch["attention_mask_chosen"],
            input_ids_y_prime=batch["input_ids_rejected"],
            attention_mask_y_prime=batch["attention_mask_rejected"]
        )
        
        # MOSAIC loss: negative log likelihood of preferring chosen over rejected
        mosaic_loss = -torch.log(pref_prob + 1e-7).mean()
        
        # Safety constraint loss (optional - can add safety classifier)
        safety_loss = torch.tensor(0.0, device=self.config.device)
        
        # Bias penalty: encourage diversity across cultural contexts
        context_entropy = -(F.softmax(self.context_weights, dim=0) * 
                           F.log_softmax(self.context_weights, dim=0)).sum()
        bias_loss = -context_entropy  # Maximize entropy to avoid bias
        
        # Nash Bargaining Solution: maximize product of gains over disagreement point
        # d (disagreement) = base model output; we approximate by using 0.5 as
        # the neutral preference (no culture prefers one over another).
        bargaining_loss = torch.tensor(0.0, device=self.config.device)
        if self.config.use_bargaining:
            # Each cultural context's gain over disagreement
            weights = F.softmax(self.context_weights, dim=0)
            gains = (pref_prob - 0.5).clamp(min=1e-7)
            # Nash product: log product = sum of log gains (maximize)
            log_nash_product = sum(
                weights[i] * torch.log(gains + 1e-7) for i in range(len(weights))
            )
            bargaining_loss = -log_nash_product.mean()

        # Total loss
        total_loss = (
            self.config.cultural_alignment_weight * mosaic_loss +
            self.config.safety_weight * safety_loss +
            self.config.bias_penalty_weight * bias_loss
        )
        if self.config.use_bargaining:
            total_loss = total_loss + 0.2 * bargaining_loss

        metrics = {
            "mosaic_loss": mosaic_loss.item(),
            "safety_loss": safety_loss.item(),
            "bias_loss": bias_loss.item(),
            "bargaining_loss": bargaining_loss.item(),
            "total_loss": total_loss.item(),
            "preference_accuracy": (pref_prob > 0.5).float().mean().item()
        }
        
        return total_loss, metrics


class MOSAICOptimizer:
    """
    Optimizer for computing Nash equilibrium in cultural alignment game.
    
    Uses fictitious play algorithm to find equilibrium policy.
    """
    
    def __init__(self, model: PairwisePreferenceModel, config: MOSAICConfig):
        self.model = model
        self.config = config
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate
        )
        
        # Track policy distributions for fictitious play
        self.policy_history: List[torch.Tensor] = []
        
    def compute_nash_equilibrium(
        self,
        dataloader: DataLoader,
        num_iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute Nash equilibrium using fictitious play.
        
        Args:
            dataloader: Data loader for training data
            num_iterations: Number of iterations (default: from config)
            
        Returns:
            equilibrium_policy: Learned policy at equilibrium
            convergence_info: Information about convergence
        """
        num_iterations = num_iterations or self.config.equilibrium_iterations
        
        self.model.train()
        convergence_losses = []
        
        logger.info(f"Starting Nash equilibrium computation ({num_iterations} iterations)")
        
        for iteration in range(num_iterations):
            epoch_losses = []
            epoch_metrics = []
            
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                loss, metrics = self.model(batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                epoch_losses.append(metrics["total_loss"])
                epoch_metrics.append(metrics)
            
            # Average metrics for this iteration
            avg_loss = np.mean(epoch_losses)
            convergence_losses.append(avg_loss)
            
            # Log progress
            if (iteration + 1) % 50 == 0:
                logger.info(f"Iteration {iteration + 1}/{num_iterations} - Loss: {avg_loss:.4f}")
            
            # Store policy for fictitious play
            with torch.no_grad():
                policy_dist = F.softmax(self.model.context_weights, dim=0).cpu()
                self.policy_history.append(policy_dist)
        
        # Check convergence
        if len(convergence_losses) > 10:
            recent_std = np.std(convergence_losses[-10:])
            converged = recent_std < 0.01
        else:
            converged = False
        
        return {
            "equilibrium_policy": F.softmax(self.model.context_weights, dim=0).detach().cpu().numpy(),
            "policy_history": self.policy_history,
            "convergence_losses": convergence_losses,
            "converged": converged,
            "final_loss": convergence_losses[-1] if convergence_losses else None
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "context_weights": self.model.context_weights.detach().cpu().numpy(),
            "policy_history": self.policy_history,
            "config": self.config
        }, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model.context_weights.data = torch.tensor(
            checkpoint["context_weights"], device=self.config.device
        )
        logger.info(f"Checkpoint loaded from {path}")


class MOSAICTrainer:
    """
    Main trainer class for MOSAIC framework.
    
    Orchestrates the training process including:
    - Data loading
    - Model initialization
    - Nash equilibrium computation
    - Evaluation
    """
    
    def __init__(self, config: MOSAICConfig):
        self.config = config
        
        # Load tokenizer and model
        logger.info(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
            device_map="auto" if config.device == "cuda" else None
        )
        
        # Initialize MOSAIC model
        self.model = PairwisePreferenceModel(config, base_model)
        self.model.to(config.device)
        
        # Initialize optimizer
        self.optimizer = MOSAICOptimizer(self.model, config)
        
        logger.info("MOSAIC Trainer initialized successfully")
        
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """
        Train the MOSAIC model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            
        Returns:
            training_results: Dictionary with training results
        """
        # Create data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        # Compute Nash equilibrium
        equilibrium_results = self.optimizer.compute_nash_equilibrium(train_loader)
        
        # Evaluate if eval dataset provided
        eval_results = None
        if eval_dataset is not None:
            eval_results = self.evaluate(eval_dataset)
        
        return {
            "equilibrium_results": equilibrium_results,
            "eval_results": eval_results,
            "config": self.config
        }
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """Evaluate the model on a dataset."""
        self.model.eval()
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
                _, metrics = self.model(batch)
                all_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function for data loader."""
        return {
            "input_ids_chosen": torch.stack([item["input_ids_chosen"] for item in batch]),
            "attention_mask_chosen": torch.stack([item["attention_mask_chosen"] for item in batch]),
            "input_ids_rejected": torch.stack([item["input_ids_rejected"] for item in batch]),
            "attention_mask_rejected": torch.stack([item["attention_mask_rejected"] for item in batch]),
            "cultural_context": torch.tensor([item["cultural_context"] for item in batch])
        }
    
    def save_model(self, output_dir: str):
        """Save the trained model."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save config
        self.optimizer.save_checkpoint(os.path.join(output_dir, "mosaic_checkpoint.pt"))
        
        logger.info(f"Model saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    config = MOSAICConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        num_cultural_contexts=12,
        batch_size=2,
        equilibrium_iterations=100
    )
    
    trainer = MOSAICTrainer(config)
    print("MOSAIC Trainer ready for training!")
