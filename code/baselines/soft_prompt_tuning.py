"""
Soft Prompt Tuning Baseline
============================
Learning culture-specific prompts for steering model behavior.

Reference: Zhang et al. (2024). Soft prompt tuning for cultural 
adaptation of large language models. arXiv:2403.09876.

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
class SoftPromptConfig:
    """Configuration for Soft Prompt Tuning."""
    
    # Model
    model_name: str = "meta-llama/Llama-2-13b-hf"
    
    # Soft prompt parameters
    num_cultural_prompts: int = 12  # Number of cultural prompt sets
    prompt_length: int = 20  # Number of soft prompt tokens per culture
    prompt_hidden_dim: int = 4096  # Should match model hidden size
    
    # Training
    learning_rate: float = 0.01  # Higher LR for prompt tuning
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 5
    max_seq_length: int = 512
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CulturalSoftPrompts(nn.Module):
    """
    Learnable soft prompts for different cultural contexts.
    """
    
    def __init__(self, config: SoftPromptConfig):
        super().__init__()
        self.config = config
        
        # Soft prompt embeddings for each cultural context
        # Shape: [num_cultures, prompt_length, hidden_dim]
        self.cultural_prompts = nn.Parameter(
            torch.randn(config.num_cultural_prompts, config.prompt_length, config.prompt_hidden_dim)
        )
        
        # Initialize with small values
        nn.init.normal_(self.cultural_prompts, mean=0.0, std=0.02)
        
    def forward(self, cultural_context: torch.Tensor) -> torch.Tensor:
        """
        Get soft prompts for given cultural contexts.
        
        Args:
            cultural_context: [batch] cultural context IDs
            
        Returns:
            prompts: [batch, prompt_length, hidden_dim] soft prompts
        """
        # Gather prompts for each example in batch
        batch_size = cultural_context.shape[0]
        prompts = self.cultural_prompts[cultural_context]  # [batch, prompt_len, hidden]
        
        return prompts


class SoftPromptTuningModel(nn.Module):
    """
    LLM with soft prompt tuning for cultural adaptation.
    """
    
    def __init__(self, config: SoftPromptConfig, base_model: PreTrainedModel):
        super().__init__()
        self.config = config
        self.base_model = base_model
        
        # Cultural soft prompts
        self.soft_prompts = CulturalSoftPrompts(config)
        
        # Freeze base model (only train soft prompts)
        for param in self.base_model.parameters():
            param.requires_grad = False
            
    def prepare_inputs_with_prompts(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cultural_context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare inputs by prepending soft prompts.
        
        Args:
            input_ids: [batch, seq_len] input token IDs
            attention_mask: [batch, seq_len] attention mask
            cultural_context: [batch] cultural context IDs
            
        Returns:
            new_input_ids: Extended input IDs
            new_attention_mask: Extended attention mask
            prompt_attention_mask: Mask for prompt tokens
        """
        batch_size = input_ids.shape[0]
        prompt_length = self.config.prompt_length
        
        # Get soft prompts
        prompts = self.soft_prompts(cultural_context)  # [batch, prompt_len, hidden]
        
        # Create dummy input IDs for prompts (will be replaced by embeddings)
        prompt_input_ids = torch.full(
            (batch_size, prompt_length),
            self.base_model.config.pad_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device
        )
        
        # Concatenate prompts with input
        new_input_ids = torch.cat([prompt_input_ids, input_ids], dim=1)
        new_attention_mask = torch.cat([
            torch.ones((batch_size, prompt_length), device=input_ids.device),
            attention_mask
        ], dim=1)
        
        return new_input_ids, new_attention_mask, prompts
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cultural_context: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with soft prompts.
        
        Args:
            input_ids: [batch, seq_len] input token IDs
            attention_mask: [batch, seq_len] attention mask
            cultural_context: [batch] cultural context IDs
            labels: [batch, seq_len] optional labels
            
        Returns:
            Dictionary with logits and loss
        """
        batch_size = input_ids.shape[0]
        prompt_length = self.config.prompt_length
        
        # Prepare inputs with soft prompts
        new_input_ids, new_attention_mask, prompts = self.prepare_inputs_with_prompts(
            input_ids, attention_mask, cultural_context
        )
        
        # Get input embeddings
        input_embeds = self.base_model.model.embed_tokens(new_input_ids)
        
        # Replace prompt embeddings with soft prompts
        input_embeds[:, :prompt_length, :] = prompts
        
        # Forward through model
        outputs = self.base_model(
            inputs_embeds=input_embeds,
            attention_mask=new_attention_mask,
            return_dict=True
        )
        
        logits = outputs.logits[:, prompt_length:, :]  # Remove prompt positions
        
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
            "loss": loss
        }
    
    def generate_with_culture(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cultural_context: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        Generate text with cultural soft prompts.
        
        Args:
            input_ids: [batch, seq_len] input token IDs
            attention_mask: [batch, seq_len] attention mask
            cultural_context: [batch] cultural context IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            generated_ids: [batch, seq_len + new_tokens] generated sequences
        """
        self.eval()
        
        with torch.no_grad():
            # Get soft prompts
            prompts = self.soft_prompts(cultural_context)
            prompt_length = self.config.prompt_length
            
            # Prepare initial inputs
            input_embeds = self.base_model.model.embed_tokens(input_ids)
            batch_size = input_ids.shape[0]
            
            # Prepend prompts
            prompt_embeds = prompts
            current_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
            current_input_ids = input_ids
            
            generated = []
            
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.base_model(
                    inputs_embeds=current_embeds,
                    use_cache=True
                )
                
                # Get next token logits
                next_token_logits = outputs.logits[:, -prompt_length-1, :]
                
                # Sample
                if temperature > 0:
                    probs = F.softmax(next_token_logits / temperature, dim=-1)
                    # Top-p sampling
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum_probs > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    sorted_probs[mask] = 0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = torch.gather(sorted_indices, -1, next_token).squeeze(-1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1)
                
                generated.append(next_token)
                
                # Update inputs
                next_embeds = self.base_model.model.embed_tokens(next_token)
                current_embeds = next_embeds.unsqueeze(1)
                current_input_ids = torch.cat([current_input_ids, next_token.unsqueeze(1)], dim=1)
                
                # Stop if all sequences have EOS
                if (next_token == self.base_model.config.eos_token_id).all():
                    break
            
            generated_ids = torch.stack(generated, dim=1) if generated else torch.tensor([], device=input_ids.device)
            return torch.cat([input_ids, generated_ids], dim=1)


class SoftPromptTrainer:
    """
    Trainer for Soft Prompt Tuning.
    """
    
    def __init__(self, config: SoftPromptConfig):
        self.config = config
        
        # Load tokenizer and model
        logger.info(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32
        )
        
        # Wrap with soft prompt tuning
        self.model = SoftPromptTuningModel(config, base_model)
        self.model.to(config.device)
        
        # Optimizer (only train soft prompts)
        self.optimizer = torch.optim.AdamW(
            self.model.soft_prompts.parameters(),
            lr=config.learning_rate
        )
        
        num_params = sum(p.numel() for p in self.model.soft_prompts.parameters())
        logger.info(f"Soft Prompt Trainer initialized with {num_params:,} trainable parameters")
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ) -> Dict[str, Any]:
        """Train with soft prompt tuning."""
        logger.info("Starting Soft Prompt Tuning...")
        
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
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """Evaluate the model."""
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
        """Collate function."""
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
        
        # Save base model (frozen)
        self.model.base_model.save_pretrained(os.path.join(output_dir, "base_model"))
        
        # Save soft prompts
        torch.save({
            "soft_prompts": self.model.soft_prompts.state_dict(),
            "config": self.config
        }, os.path.join(output_dir, "soft_prompts.pt"))
        
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Soft Prompt model saved to {output_dir}")


if __name__ == "__main__":
    config = SoftPromptConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        num_epochs=1,
        batch_size=2
    )
    trainer = SoftPromptTrainer(config)
    print("Soft Prompt Tuning Trainer ready!")
