"""
Evaluation Metrics
===================
Metrics for evaluating cultural alignment of LLMs.

Metrics:
- Cultural Appropriateness Score (CAS)
- Cultural Bias Index (CBI)
- Pairwise Win Rate
- Diversity Score

Author: Thang Nguyen Xuan
Institution: Hanoi University, Vietnam
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result of a metric computation."""
    name: str
    value: float
    std: Optional[float] = None
    breakdown: Optional[Dict[str, float]] = None


class CulturalAppropriatenessScore:
    """
    Cultural Appropriateness Score (CAS).
    
    Measures how appropriate a response is for a given cultural context.
    Uses trained judge models or human annotations.
    """
    
    def __init__(self, judge_model=None, num_cultural_dimensions: int = 12):
        self.judge_model = judge_model
        self.num_cultural_dimensions = num_cultural_dimensions
        
        # Hofstede's cultural dimensions
        self.dimensions = [
            "Power Distance",
            "Individualism vs Collectivism",
            "Masculinity vs Femininity",
            "Uncertainty Avoidance",
            "Long-term Orientation",
            "Indulgence vs Restraint",
            "High-context vs Low-context",
            "Monochronic vs Polychronic",
            "Universalism vs Particularism",
            "Achievement vs Ascription",
            "Specific vs Diffuse",
            "Neutral vs Emotional"
        ]
    
    def compute(
        self,
        responses: List[str],
        cultural_contexts: List[int],
        reference_responses: Optional[List[str]] = None
    ) -> MetricResult:
        """
        Compute Cultural Appropriateness Score.
        
        Args:
            responses: Generated responses
            cultural_contexts: Cultural context IDs
            reference_responses: Optional reference (gold) responses
            
        Returns:
            MetricResult with CAS score
        """
        if self.judge_model is not None:
            # Use trained judge model
            scores = self._judge_model_score(responses, cultural_contexts)
        elif reference_responses is not None:
            # Use similarity to reference
            scores = self._reference_similarity_score(responses, reference_responses)
        else:
            # Use rule-based heuristics
            scores = self._heuristic_score(responses, cultural_contexts)
        
        # Aggregate by cultural dimension
        breakdown = self._compute_dimension_breakdown(scores, cultural_contexts)
        
        return MetricResult(
            name="Cultural Appropriateness Score (CAS)",
            value=np.mean(scores),
            std=np.std(scores),
            breakdown=breakdown
        )
    
    def _judge_model_score(
        self,
        responses: List[str],
        cultural_contexts: List[int]
    ) -> np.ndarray:
        """Score using trained judge model."""
        # Placeholder - would use actual judge model
        scores = np.random.uniform(3.0, 4.5, len(responses))
        return scores
    
    def _reference_similarity_score(
        self,
        responses: List[str],
        reference_responses: List[str]
    ) -> np.ndarray:
        """Score based on similarity to reference responses."""
        # Use BLEU/ROUGE or embedding similarity
        from difflib import SequenceMatcher
        
        scores = []
        for resp, ref in zip(responses, reference_responses):
            similarity = SequenceMatcher(None, resp.lower(), ref.lower()).ratio()
            # Scale to 1-5 range
            score = 1 + 4 * similarity
            scores.append(score)
        
        return np.array(scores)
    
    def _heuristic_score(
        self,
        responses: List[str],
        cultural_contexts: List[int]
    ) -> np.ndarray:
        """Rule-based heuristic scoring."""
        # Placeholder heuristic
        scores = np.random.uniform(3.0, 4.0, len(responses))
        return scores
    
    def _compute_dimension_breakdown(
        self,
        scores: np.ndarray,
        cultural_contexts: List[int]
    ) -> Dict[str, float]:
        """Compute score breakdown by cultural dimension."""
        breakdown = {}
        
        # Group by cultural context
        context_scores = defaultdict(list)
        for score, context in zip(scores, cultural_contexts):
            context_scores[context].append(score)
        
        # Map contexts to dimensions
        for context, ctx_scores in context_scores.items():
            dim_name = self.dimensions[context % len(self.dimensions)]
            breakdown[dim_name] = np.mean(ctx_scores)
        
        return breakdown


class CulturalBiasIndex:
    """
    Cultural Bias Index (CBI).
    
    Measures the degree of cultural bias in model outputs.
    Lower is better (less biased).
    """
    
    def __init__(self):
        self.cultural_markers = self._load_cultural_markers()
    
    def _load_cultural_markers(self) -> Dict[int, List[str]]:
        """Load cultural markers for bias detection."""
        # Simplified cultural markers
        return {
            0: ["American", "USA", "Western", "individual"],  # North America
            1: ["British", "European", "Western"],  # Western Europe
            2: ["Chinese", "Japan", "Korean", "Asian", "collective"],  # East Asia
            3: ["Indian", "South Asian"],  # South Asia
            4: ["Thai", "Vietnamese", "Indonesian", "Southeast Asian"],  # Southeast Asia
            5: ["Brazilian", "Latin", "Hispanic"],  # Latin America
            6: ["Arab", "Middle Eastern", "Islamic"],  # Middle East
            7: ["African", "Nigerian", "Kenyan"],  # Sub-Saharan Africa
            8: ["Russian", "Eastern European"],  # Eastern Europe
            9: ["Australian", "Oceanian"],  # Oceania
            10: ["Nordic", "Scandinavian"],  # Nordic
            11: ["Global", "Universal", "Neutral"]  # Neutral/Global
        }
    
    def compute(
        self,
        responses: List[str],
        expected_contexts: List[int]
    ) -> MetricResult:
        """
        Compute Cultural Bias Index.
        
        Args:
            responses: Model responses
            expected_contexts: Expected cultural contexts
            
        Returns:
            MetricResult with CBI score (lower is better)
        """
        bias_scores = []
        
        for response, expected_context in zip(responses, expected_contexts):
            bias_score = self._compute_response_bias(response, expected_context)
            bias_scores.append(bias_score)
        
        bias_array = np.array(bias_scores)
        
        # Breakdown by region
        breakdown = self._compute_regional_breakdown(bias_scores, expected_contexts)
        
        return MetricResult(
            name="Cultural Bias Index (CBI)",
            value=np.mean(bias_array),
            std=np.std(bias_array),
            breakdown=breakdown
        )
    
    def _compute_response_bias(
        self,
        response: str,
        expected_context: int
    ) -> float:
        """Compute bias score for a single response."""
        response_lower = response.lower()
        
        # Count markers from non-expected cultures
        non_expected_markers = 0
        for context, markers in self.cultural_markers.items():
            if context != expected_context:
                for marker in markers:
                    if marker.lower() in response_lower:
                        non_expected_markers += 1
        
        # Normalize by response length
        bias_score = non_expected_markers / (len(response.split()) + 1)
        
        return min(bias_score * 10, 1.0)  # Scale to 0-1
    
    def _compute_regional_breakdown(
        self,
        bias_scores: List[float],
        contexts: List[int]
    ) -> Dict[str, float]:
        """Compute bias breakdown by region."""
        region_names = [
            "North America", "Western Europe", "East Asia", "South Asia",
            "Southeast Asia", "Latin America", "Middle East", 
            "Sub-Saharan Africa", "Eastern Europe", "Oceania", "Nordic", "Neutral"
        ]
        
        breakdown = {}
        region_scores = defaultdict(list)
        
        for score, context in zip(bias_scores, contexts):
            region_scores[region_names[context % len(region_names)]].append(score)
        
        for region, scores in region_scores.items():
            breakdown[region] = np.mean(scores)
        
        return breakdown


class PairwiseWinRate:
    """
    Pairwise Win Rate.
    
    Measures how often model outputs are preferred over baselines
    in pairwise comparisons.
    """
    
    def __init__(self, judge_model=None):
        self.judge_model = judge_model
    
    def compute(
        self,
        model_responses: List[str],
        baseline_responses: List[str],
        cultural_contexts: List[int],
        prompts: Optional[List[str]] = None
    ) -> MetricResult:
        """
        Compute pairwise win rate.
        
        Args:
            model_responses: Responses from our model
            baseline_responses: Responses from baseline
            cultural_contexts: Cultural contexts
            prompts: Optional prompts
            
        Returns:
            MetricResult with win rate
        """
        wins = 0
        ties = 0
        total = len(model_responses)
        
        for i in range(total):
            preference = self._judge_preference(
                model_responses[i],
                baseline_responses[i],
                cultural_contexts[i],
                prompts[i] if prompts else None
            )
            
            if preference == 1:
                wins += 1
            elif preference == 0:
                ties += 1
        
        win_rate = (wins + 0.5 * ties) / total
        
        # Breakdown by context
        breakdown = self._compute_context_breakdown(
            model_responses, baseline_responses, cultural_contexts
        )
        
        return MetricResult(
            name="Pairwise Win Rate",
            value=win_rate,
            breakdown=breakdown
        )
    
    def _judge_preference(
        self,
        response_a: str,
        response_b: str,
        context: int,
        prompt: Optional[str] = None
    ) -> int:
        """
        Judge preference between two responses.
        
        Returns:
            1 if A preferred, -1 if B preferred, 0 if tie
        """
        if self.judge_model is not None:
            # Use trained judge
            return self._model_judge(response_a, response_b, context, prompt)
        else:
            # Use heuristic (placeholder)
            # In practice, would use human annotations or LLM judge
            score_a = len(response_a)  # Simplified heuristic
            score_b = len(response_b)
            
            if score_a > score_b * 1.1:
                return 1
            elif score_b > score_a * 1.1:
                return -1
            else:
                return 0
    
    def _model_judge(
        self,
        response_a: str,
        response_b: str,
        context: int,
        prompt: Optional[str]
    ) -> int:
        """Use model to judge preference."""
        # Placeholder for actual judge model
        return 0
    
    def _compute_context_breakdown(
        self,
        model_responses: List[str],
        baseline_responses: List[str],
        contexts: List[int]
    ) -> Dict[str, float]:
        """Compute win rate breakdown by cultural context."""
        context_wins = defaultdict(lambda: {"wins": 0, "total": 0})
        
        for i, context in enumerate(contexts):
            pref = self._judge_preference(
                model_responses[i],
                baseline_responses[i],
                context
            )
            context_wins[context]["total"] += 1
            if pref == 1:
                context_wins[context]["wins"] += 1
            elif pref == 0:
                context_wins[context]["wins"] += 0.5
        
        breakdown = {
            f"Context {c}": data["wins"] / data["total"]
            for c, data in context_wins.items()
        }
        
        return breakdown


class DiversityScore:
    """
    Diversity Score.
    
    Measures the diversity of model outputs across different cultural contexts.
    """
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
    
    def compute(
        self,
        responses: List[str],
        cultural_contexts: List[int]
    ) -> MetricResult:
        """
        Compute diversity score.
        
        Args:
            responses: Model responses
            cultural_contexts: Cultural contexts
            
        Returns:
            MetricResult with diversity score
        """
        # Get embeddings
        if self.embedding_model is not None:
            embeddings = self._get_embeddings(responses)
        else:
            # Use simple bag-of-words
            embeddings = self._get_bow_embeddings(responses)
        
        # Compute within-context diversity
        within_context_diversity = self._compute_within_context_diversity(
            embeddings, cultural_contexts
        )
        
        # Compute across-context diversity
        across_context_diversity = self._compute_across_context_diversity(
            embeddings, cultural_contexts
        )
        
        # Combined score
        diversity_score = 0.5 * within_context_diversity + 0.5 * across_context_diversity
        
        return MetricResult(
            name="Diversity Score",
            value=diversity_score,
            breakdown={
                "within_context": within_context_diversity,
                "across_context": across_context_diversity
            }
        )
    
    def _get_embeddings(self, responses: List[str]) -> np.ndarray:
        """Get embeddings using embedding model."""
        # Placeholder - would use actual embedding model
        embeddings = np.random.randn(len(responses), 768)
        return embeddings
    
    def _get_bow_embeddings(self, responses: List[str]) -> np.ndarray:
        """Get bag-of-words embeddings."""
        from sklearn.feature_extraction.text import CountVectorizer
        
        vectorizer = CountVectorizer(max_features=1000)
        embeddings = vectorizer.fit_transform(responses).toarray()
        
        # Normalize
        embeddings = embeddings / (embeddings.sum(axis=1, keepdims=True) + 1e-7)
        
        return embeddings
    
    def _compute_within_context_diversity(
        self,
        embeddings: np.ndarray,
        contexts: List[int]
    ) -> float:
        """Compute diversity within each cultural context."""
        context_embeddings = defaultdict(list)
        
        for emb, context in zip(embeddings, contexts):
            context_embeddings[context].append(emb)
        
        diversities = []
        for context, embs in context_embeddings.items():
            if len(embs) > 1:
                # Compute pairwise distances
                embs_array = np.array(embs)
                distances = []
                for i in range(len(embs_array)):
                    for j in range(i + 1, len(embs_array)):
                        dist = 1 - np.dot(embs_array[i], embs_array[j])
                        distances.append(dist)
                if distances:
                    diversities.append(np.mean(distances))
        
        return np.mean(diversities) if diversities else 0.0
    
    def _compute_across_context_diversity(
        self,
        embeddings: np.ndarray,
        contexts: List[int]
    ) -> float:
        """Compute diversity across cultural contexts."""
        context_means = {}
        
        for emb, context in zip(embeddings, contexts):
            if context not in context_means:
                context_means[context] = []
            context_means[context].append(emb)
        
        # Compute mean embedding per context
        context_mean_embs = {
            c: np.mean(embs, axis=0)
            for c, embs in context_means.items()
        }
        
        # Compute pairwise distances between context means
        mean_embs = list(context_mean_embs.values())
        distances = []
        for i in range(len(mean_embs)):
            for j in range(i + 1, len(mean_embs)):
                dist = 1 - np.dot(mean_embs[i], mean_embs[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0


class CulturalEquilibriumEvaluator:
    """
    Main evaluator for Cultural Equilibrium models.
    
    Computes all metrics and generates evaluation reports.
    """
    
    def __init__(self):
        self.cas = CulturalAppropriatenessScore()
        self.cbi = CulturalBiasIndex()
        self.win_rate = PairwiseWinRate()
        self.diversity = DiversityScore()
    
    def evaluate(
        self,
        model_responses: List[str],
        cultural_contexts: List[int],
        baseline_responses: Optional[List[str]] = None,
        reference_responses: Optional[List[str]] = None,
        prompts: Optional[List[str]] = None
    ) -> Dict[str, MetricResult]:
        """
        Compute all evaluation metrics.
        
        Args:
            model_responses: Responses from evaluated model
            cultural_contexts: Cultural context IDs
            baseline_responses: Optional baseline responses for comparison
            reference_responses: Optional reference (gold) responses
            prompts: Optional prompts
            
        Returns:
            Dictionary of metric results
        """
        results = {}
        
        # Cultural Appropriateness Score
        results["cas"] = self.cas.compute(
            model_responses, cultural_contexts, reference_responses
        )
        
        # Cultural Bias Index
        results["cbi"] = self.cbi.compute(model_responses, cultural_contexts)
        
        # Pairwise Win Rate (if baseline provided)
        if baseline_responses is not None:
            results["win_rate"] = self.win_rate.compute(
                model_responses, baseline_responses, cultural_contexts, prompts
            )
        
        # Diversity Score
        results["diversity"] = self.diversity.compute(
            model_responses, cultural_contexts
        )
        
        # Log results
        self._log_results(results)
        
        return results
    
    def _log_results(self, results: Dict[str, MetricResult]):
        """Log evaluation results."""
        logger.info("=" * 60)
        logger.info("CULTURAL EQUILIBRIUM EVALUATION RESULTS")
        logger.info("=" * 60)
        
        for name, result in results.items():
            logger.info(f"\n{result.name}:")
            logger.info(f"  Mean: {result.value:.4f}")
            if result.std is not None:
                logger.info(f"  Std:  {result.std:.4f}")
            if result.breakdown:
                logger.info("  Breakdown:")
                for key, value in result.breakdown.items():
                    logger.info(f"    {key}: {value:.4f}")
        
        logger.info("=" * 60)


if __name__ == "__main__":
    # Example usage
    evaluator = CulturalEquilibriumEvaluator()
    
    # Dummy data for testing
    responses = ["Response 1", "Response 2", "Response 3"]
    contexts = [0, 1, 2]
    
    results = evaluator.evaluate(responses, contexts)
    print("Evaluation complete!")
