"""
Cultural Fine-tuning Baseline
==============================
Fine-tuning on culturally-diverse datasets.

Reference: Kumar et al. (2024). Cultural fine-tuning for cross-cultural 
language model adaptation. NAACL.

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
class CulturalFTConfig:
    """Configuration for Cultural Fine-tuning."""
    
    # Model
    model_name: str = "meta-llama/Llama-2-13b-hf"
    
    # Cultural adaptation
    num_cultural_adapters: int = 12  # Number of cultural adapters
    adapter_hidden_dim: int = 256
    adapter_dropout: float = 0.1
    
    # Training
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    max_seq_length: int = 512
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CulturalAdapter(nn.Module):
    """
    Cultural adapter module for adapting model behavior to specific cultures.
    
    Uses lightweight adapter layers that can be swapped based on cultural context.
    """
    
    def __init__(self, config: CulturalFTConfig, hidden_dim: int):
        super().__init__()
        self.config = config
        
        # Adapter network (bottleneck architecture)
        self.adapter_down = nn.Linear(hidden_dim, config.adapter_hidden_dim)
        self.adapter_up = nn.Linear(config.adapter_hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(config.adapter_dropout)
        self.activation = nn.GELU()
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply cultural adapter to hidden states.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim] model hidden states
            
        Returns:
            adapted_states: [batch, seq_len, hidden_dim] adapted hidden states
        """
        residual = hidden_states
        
        # Adapter forward pass
        hidden = self.adapter_down(hidden_states)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.adapter_up(hidden)
        
        # Residual connection
        output = self.layer_norm(hidden + residual)
        return output


class CulturalAdapterModel(nn.Module):
    """
    LLM with cultural adapters for cross-cultural adaptation.
    """
    
    def __init__(self, config: CulturalFTConfig, base_model: PreTrainedModel):
        super().__init__()
        self.config = config
        self.base_model = base_model
        
        # Create cultural adapters for each layer
        self.cultural_adapters = nn.ModuleDict()
        for i in range(config.num_cultural_adapters):
            self.cultural_adapters[f"adapter_{i}"] = CulturalAdapter(
                config, 
                hidden_dim=base_model.config.hidden_size
            )
        
        # Cultural context embedding
        self.cultural_embedding = nn.Embedding(
            config.num_cultural_adapters, 
            base_model.config.hidden_size
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cultural_context: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with cultural adaptation.
        
        Args:
            input_ids: [batch, seq_len] input token IDs
            attention_mask: [batch, seq_len] attention mask
            cultural_context: [batch] cultural context IDs
            labels: [batch, seq_len] optional labels for loss computation
            
        Returns:
            Dictionary containing:
                - logits: Model outputs
                - loss: Computed loss if labels provided
        """
        # Get base model outputs with hidden states
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = outputs.hidden_states
        
        # Apply cultural adaptation if context provided
        if cultural_context is not None:
            # Get cultural embedding
            cultural_emb = self.cultural_embedding(cultural_context)  # [batch, hidden]
            
            # Apply cultural adapter to last few layers
            adapter_name = f"adapter_{cultural_context[0].item()}"  # Use first context in batch
            if adapter_name in self.cultural_adapters:
                adapter = self.cultural_adapters[adapter_name]
                
                # Apply adapter to last hidden state
                last_hidden = hidden_states[-1]  # [batch, seq, hidden]
                adapted_hidden = adapter(last_hidden)
                
                # Add cultural embedding
                adapted_hidden = adapted_hidden + cultural_emb.unsqueeze(1)
                
                # Replace last hidden state
                hidden_states = list(hidden_states[:-1]) + [adapted_hidden]
        
        # Get logits from final hidden state
        logits = self.base_model.lm_head(hidden_states[-1])
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": hidden_states
        }


class CulturalFineTuningTrainer:
    """
    Trainer for Cultural Fine-tuning.
    """
    
    def __init__(self, config: CulturalFTConfig):
        self.config = config
        
        # Load tokenizer and model
        logger.info(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32
        )
        
        # Wrap with cultural adapters
        self.model = CulturalAdapterModel(config, base_model)
        self.model.to(config.device)
        
        # Optimizer (only train adapter parameters)
        adapter_params = [p for p in self.model.cultural_adapters.parameters() if p.requires_grad]
        adapter_params += list(self.model.cultural_embedding.parameters())
        
        self.optimizer = torch.optim.AdamW(adapter_params, lr=config.learning_rate)
        
        logger.info(f"Cultural Fine-tuning Trainer initialized with {sum(p.numel() for p in adapter_params):,} trainable parameters")
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ) -> Dict[str, Any]:
        """
        Train the model with cultural fine-tuning.
        
        Args:
            train_dataset: Training dataset with cultural annotations
            eval_dataset: Optional evaluation dataset
            
        Returns:
            training_results: Dictionary with training results
        """
        logger.info("Starting Cultural Fine-tuning...")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        self.model.train()
        all_epoch_metrics = []
        
        for epoch in range(self.config.num_epochs):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    cultural_context=batch["cultural_context"],
                    labels=batch["labels"]
                )
                
                loss = outputs["loss"]
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                epoch_losses.append(loss.item())
                
                # Log progress
                if (batch_idx + 1) % 50 == 0:
                    avg_loss = np.mean(epoch_losses[-50:])
                    logger.info(f"Epoch {epoch+1}, Step {batch_idx+1} - Loss: {avg_loss:.4f}")
            
            # Average epoch loss
            avg_epoch_loss = np.mean(epoch_losses)
            all_epoch_metrics.append({"loss": avg_epoch_loss})
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - Loss: {avg_epoch_loss:.4f}")
        
        # Evaluate
        eval_results = None
        if eval_dataset is not None:
            eval_results = self.evaluate(eval_dataset)
        
        return {
            "epoch_metrics": all_epoch_metrics,
            "final_loss": all_epoch_metrics[-1]["loss"] if all_epoch_metrics else None,
            "eval_results": eval_results
        }
    
    def evaluate(
        self,
        eval_dataset: Dataset
    ) -> Dict[str, float]:
        """Evaluate the model."""
        logger.info("Evaluating Cultural Fine-tuning model...")
        
        self.model.eval()
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self._collate_fn
        )
        
        all_losses = []
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    cultural_context=batch["cultural_context"],
                    labels=batch["labels"]
                )
                
                all_losses.append(outputs["loss"].item())
        
        return {
            "eval_loss": np.mean(all_losses),
            "perplexity": np.exp(np.mean(all_losses))
        }
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function for data loader."""
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
            "cultural_context": torch.tensor([item["cultural_context"] for item in batch])
        }
    
    def save_model(self, output_dir: str):
        """Save the model."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save base model
        self.model.base_model.save_pretrained(os.path.join(output_dir, "base_model"))
        
        # Save cultural adapters
        torch.save({
            "cultural_adapters": self.model.cultural_adapters.state_dict(),
            "cultural_embedding": self.model.cultural_embedding.state_dict(),
            "config": self.config
        }, os.path.join(output_dir, "cultural_adapters.pt"))
        
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Cultural Fine-tuning model saved to {output_dir}")


if __name__ == "__main__":
    config = CulturalFTConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        num_epochs=1,
        batch_size=2
    )
    trainer = CulturalFineTuningTrainer(config)
    print("Cultural Fine-tuning Trainer ready!")
