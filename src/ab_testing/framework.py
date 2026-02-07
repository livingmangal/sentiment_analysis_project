"""
A/B Testing Framework for Model Versioning
Implements traffic splitting, metrics collection, and statistical analysis
"""
import hashlib
import json
import logging
import os
import random
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TrafficSplitStrategy(Enum):
    """Traffic splitting strategies"""
    RANDOM = "random"
    SESSION_HASH = "session_hash"
    WEIGHTED = "weighted"


class ModelVariant:
    """Represents a model variant in A/B testing"""

    def __init__(
        self,
        variant_id: str,
        model_path: str,
        preprocessor_path: str,
        metadata_path: Optional[str] = None,
        weight: float = 1.0,
        description: str = ""
    ):
        from src.predict import SentimentPredictor
        self.variant_id = variant_id
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.metadata_path = metadata_path
        self.weight = weight
        self.description = description
        self.predictor: Optional[SentimentPredictor] = None

    def initialize_predictor(self) -> Any:
        """Lazy load the predictor"""
        if self.predictor is None:
            from src.predict import SentimentPredictor
            self.predictor = SentimentPredictor(
                self.model_path,
                self.preprocessor_path,
                self.metadata_path
            )
        return self.predictor

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "variant_id": self.variant_id,
            "model_path": self.model_path,
            "preprocessor_path": self.preprocessor_path,
            "metadata_path": self.metadata_path,
            "weight": self.weight,
            "description": self.description
        }


class ABTestingFramework:
    """
    Core A/B testing framework for model comparison
    """

    def __init__(
        self,
        experiment_name: str,
        strategy: TrafficSplitStrategy = TrafficSplitStrategy.SESSION_HASH,
        enable_logging: bool = True
    ):
        self.experiment_name = experiment_name
        self.strategy = strategy
        self.variants: Dict[str, ModelVariant] = {}
        self.enable_logging = enable_logging
        self.total_weight = 0.0

        # Set up experiment directory
        self.experiment_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "experiments",
            experiment_name
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

        logger.info(f"Initialized A/B test: {experiment_name}")

    def add_variant(self, variant: ModelVariant) -> None:
        """Add a model variant to the experiment"""
        self.variants[variant.variant_id] = variant
        self.total_weight += variant.weight
        logger.info(f"Added variant: {variant.variant_id} (weight: {variant.weight})")

    def remove_variant(self, variant_id: str) -> None:
        """Remove a variant from the experiment"""
        if variant_id in self.variants:
            self.total_weight -= self.variants[variant_id].weight
            del self.variants[variant_id]
            logger.info(f"Removed variant: {variant_id}")

    def select_variant(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> ModelVariant:
        """
        Select a variant based on the configured strategy
        
        Args:
            session_id: Session identifier for hash-based selection
            user_id: User identifier for hash-based selection
            
        Returns:
            Selected ModelVariant
        """
        if not self.variants:
            raise ValueError("No variants configured for A/B test")

        if self.strategy == TrafficSplitStrategy.RANDOM:
            return self._random_selection()
        elif self.strategy == TrafficSplitStrategy.SESSION_HASH:
            identifier = session_id or user_id
            if not identifier:
                logger.warning("No session/user ID provided, falling back to random")
                return self._random_selection()
            return self._hash_based_selection(identifier)
        elif self.strategy == TrafficSplitStrategy.WEIGHTED:
            return self._weighted_selection()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _random_selection(self) -> ModelVariant:
        """Random variant selection"""
        variant_id = random.choice(list(self.variants.keys()))
        return self.variants[variant_id]

    def _hash_based_selection(self, identifier: str) -> ModelVariant:
        """
        Consistent hash-based selection
        Ensures same user/session always gets same variant
        """
        # Create hash of identifier
        hash_value = int(hashlib.md5(identifier.encode()).hexdigest(), 16)

        # Use hash to select variant consistently
        cumulative_weight = 0.0
        threshold = (hash_value % 10000) / 10000.0 * self.total_weight

        for variant in self.variants.values():
            cumulative_weight += variant.weight
            if threshold <= cumulative_weight:
                return variant

        # Fallback to first variant
        return list(self.variants.values())[0]

    def _weighted_selection(self) -> ModelVariant:
        """Weighted random selection based on variant weights"""
        if self.total_weight == 0:
            return self._random_selection()

        threshold = random.uniform(0, self.total_weight)
        cumulative_weight = 0.0

        for variant in self.variants.values():
            cumulative_weight += variant.weight
            if threshold <= cumulative_weight:
                return variant

        return list(self.variants.values())[0]

    def predict(
        self,
        text: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make prediction using selected variant and log results
        
        Args:
            text: Input text for sentiment analysis
            session_id: Session identifier
            user_id: User identifier
            metadata: Additional metadata to log
            
        Returns:
            Prediction result with variant information
        """
        # Select variant
        variant = self.select_variant(session_id, user_id)

        # Initialize predictor if needed
        predictor = variant.initialize_predictor()

        # Make prediction
        result = predictor.predict(text)

        # Add variant information
        if isinstance(result, dict):
            result["variant_id"] = variant.variant_id
            result["experiment_name"] = self.experiment_name

        # Log prediction if enabled
        if self.enable_logging:
            self._log_prediction(
                variant_id=variant.variant_id,
                text=text,
                result=result,
                session_id=session_id,
                user_id=user_id,
                metadata=metadata
            )

        return result if isinstance(result, dict) else {}

    def predict_batch(
        self,
        texts: List[str],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Make batch predictions using selected variant
        
        Args:
            texts: List of input texts
            session_id: Session identifier
            user_id: User identifier
            metadata: Additional metadata to log
            
        Returns:
            List of prediction results
        """
        variant = self.select_variant(session_id, user_id)
        predictor = variant.initialize_predictor()

        results = predictor.predict_batch(texts)

        # Add variant information to all results
        if isinstance(results, list):
            for result in results:
                if isinstance(result, dict):
                    result["variant_id"] = variant.variant_id
                    result["experiment_name"] = self.experiment_name

        # Log batch prediction
        if self.enable_logging and isinstance(results, list):
            for text, result in zip(texts, results):
                self._log_prediction(
                    variant_id=variant.variant_id,
                    text=text,
                    result=result,
                    session_id=session_id,
                    user_id=user_id,
                    metadata=metadata
                )

        return results if isinstance(results, list) else []

    def _log_prediction(
        self,
        variant_id: str,
        text: str,
        result: Dict[str, Any],
        session_id: Optional[str],
        user_id: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Log prediction for metrics collection"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "experiment_name": self.experiment_name,
            "variant_id": variant_id,
            "session_id": session_id,
            "user_id": user_id,
            "text_length": len(text),
            "sentiment": result.get("sentiment"),
            "confidence": result.get("confidence"),
            "raw_score": result.get("raw_score"),
            "inference_time_ms": result.get("inference_time_ms"),
            "metadata": metadata or {}
        }

        # Write to variant-specific log file
        log_file = os.path.join(
            self.experiment_dir,
            f"{variant_id}_predictions.jsonl"
        )

        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def save_config(self) -> None:
        """Save experiment configuration"""
        config = {
            "experiment_name": self.experiment_name,
            "strategy": self.strategy.value,
            "variants": {
                vid: variant.to_dict()
                for vid, variant in self.variants.items()
            },
            "created_at": datetime.now().isoformat()
        }

        config_file = os.path.join(self.experiment_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved experiment config to {config_file}")

    @classmethod
    def load_config(cls, experiment_name: str) -> 'ABTestingFramework':
        """Load experiment from saved configuration"""
        experiment_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "experiments",
            experiment_name
        )

        config_file = os.path.join(experiment_dir, "config.json")

        with open(config_file, 'r') as f:
            config = json.load(f)

        framework = cls(
            experiment_name=config["experiment_name"],
            strategy=TrafficSplitStrategy(config["strategy"])
        )

        for variant_id, variant_data in config["variants"].items():
            variant = ModelVariant(
                variant_id=variant_data["variant_id"],
                model_path=variant_data["model_path"],
                preprocessor_path=variant_data["preprocessor_path"],
                metadata_path=variant_data.get("metadata_path"),
                weight=variant_data.get("weight", 1.0),
                description=variant_data.get("description", "")
            )
            framework.add_variant(variant)

        logger.info(f"Loaded experiment config from {config_file}")
        return framework
