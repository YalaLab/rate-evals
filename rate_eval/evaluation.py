"""Consolidated evaluation module for RATE Eval.

This module combines all evaluation functionality to eliminate code duplication.
"""

import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import wandb

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

from .common import get_logger, ensure_dir_exists, RATEEvalError
from .config import get_config_value

logger = get_logger(__name__)


def discover_questions_from_json(labels_json_path: str) -> List[str]:
    """
    Discover all unique questions from the JSON labels file.

    Args:
        labels_json_path: Path to JSON file with labels (qa_results format)

    Returns:
        List of unique questions found in the data
    """
    logger.info(f"Discovering questions from {labels_json_path}")

    with open(labels_json_path, "r") as f:
        labels_data = json.load(f)

    questions = {
        question
        for data in labels_data.values()
        for questions_list in data.get("qa_results", {}).values()
        if isinstance(questions_list, list)
        for qa_pair in questions_list
        if isinstance(qa_pair, dict)
        for question in qa_pair.keys()
    }

    questions_list = sorted(questions)
    logger.info(f"Discovered {len(questions_list)} unique questions")
    logger.info(f"Sample questions: {questions_list[:5]}")

    return questions_list


# =============================================================================
# PYTORCH GPU-ACCELERATED CLASSIFIER COMPONENTS
# =============================================================================


class EmbeddingDataset(Dataset):
    """Dataset for loading cached embeddings and multi-label targets."""

    def __init__(
        self, embeddings: np.ndarray, labels_dict: Dict[str, np.ndarray], questions: List[str]
    ):
        """
        Initialize dataset with embeddings and labels.

        Args:
            embeddings: Numpy array of shape (N, D) with embeddings
            labels_dict: Dictionary mapping questions to label arrays
            questions: List of questions (determines label order)
        """
        self.embeddings = torch.FloatTensor(embeddings)  # (N, D)
        self.questions = questions

        # Stack all question labels into (N, num_questions) tensor
        label_arrays = []
        for question in questions:
            if question in labels_dict:
                label_arrays.append(torch.FloatTensor(labels_dict[question]))
            else:
                # If question not found, create zero labels
                label_arrays.append(torch.zeros(len(embeddings), dtype=torch.float32))

        self.labels = torch.stack(label_arrays, dim=1)  # (N, num_questions)

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


class MultiTaskLinearClassifier(nn.Module):
    """Multi-task linear classifier for disease prediction."""

    def __init__(self, input_dim: int, num_diseases: int):
        """
        Initialize multi-task classifier.

        Args:
            input_dim: Dimension of input embeddings
            num_diseases: Number of diseases to predict
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, num_diseases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input embeddings of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, num_diseases)
        """
        return self.linear(x)


class ClassBalancedBCELoss(nn.Module):
    """Class-balanced BCE loss for multi-task classification."""

    def __init__(self, pos_weights: torch.Tensor):
        """
        Initialize loss function.

        Args:
            pos_weights: Positive class weights for each disease (num_diseases,)
        """
        super().__init__()
        self.pos_weights = pos_weights
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction="mean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute class-balanced BCE loss.

        Args:
            logits: Model predictions of shape (batch_size, num_diseases)
            targets: True labels of shape (batch_size, num_diseases)

        Returns:
            Loss scalar
        """
        return self.bce_loss(logits, targets)


class PyTorchClassifierTrainer:
    """GPU-accelerated trainer for multi-task classification."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.check_nan = get_config_value(config, "evaluation.check_nan")

        # Training hyperparameters
        self.batch_size = int(get_config_value(config, "evaluation.pytorch.batch_size"))
        self.learning_rate = float(get_config_value(config, "evaluation.pytorch.learning_rate"))
        self.num_epochs = int(get_config_value(config, "evaluation.pytorch.num_epochs"))
        self.weight_decay = float(get_config_value(config, "evaluation.pytorch.weight_decay"))
        self.num_workers = int(get_config_value(config, "evaluation.pytorch.num_workers"))
        self.use_wandb = get_config_value(config, "evaluation.use_wandb")

        logger.info(f"PyTorch trainer initialized on {self.device}")
        logger.info(
            f"Training config: batch_size={self.batch_size}, lr={self.learning_rate}, "
            f"epochs={self.num_epochs}, weight_decay={self.weight_decay}, num_workers={self.num_workers}"
        )
        logger.info(f"WandB logging: {'enabled' if self.use_wandb else 'disabled'}")

    def _calculate_pos_weights(
        self, labels_dict: Dict[str, np.ndarray], questions: List[str]
    ) -> torch.Tensor:
        """
        Calculate positive class weights for balancing.

        For BCEWithLogitsLoss, pos_weight should be the ratio of negative to positive samples.
        This gives higher weight to positive samples when they are underrepresented.
        """
        pos_weights = []

        for question in questions:
            if question not in labels_dict:
                pos_weights.append(1.0)  # Default weight
                continue

            labels = labels_dict[question]
            # Handle NaN values
            valid_labels = labels[~pd.isna(labels)] if not np.all(pd.isna(labels)) else labels

            if len(valid_labels) == 0:
                pos_weights.append(1.0)
                continue

            pos_count = np.sum(valid_labels == 1)
            neg_count = np.sum(valid_labels == 0)

            if pos_count == 0:
                pos_weights.append(1.0)  # No positive samples, use default weight
            else:
                # Standard class balancing: neg_count / pos_count
                # This gives higher weight to positive samples when they are rare
                weight = neg_count / pos_count
                pos_weights.append(weight)

        pos_weights_tensor = torch.FloatTensor(pos_weights).to(self.device)

        # Log some statistics for debugging
        logger.debug(
            f"Pos weights stats: min={pos_weights_tensor.min().item():.3f}, "
            f"max={pos_weights_tensor.max().item():.3f}, "
            f"mean={pos_weights_tensor.mean().item():.3f}"
        )

        return pos_weights_tensor

    def train(
        self, embeddings: np.ndarray, labels_dict: Dict[str, np.ndarray], questions: List[str]
    ) -> MultiTaskLinearClassifier:
        """
        Train multi-task classifier on GPU.

        Args:
            embeddings: Training embeddings of shape (N, D)
            labels_dict: Dictionary mapping questions to label arrays
            questions: List of questions to train on

        Returns:
            Trained PyTorch model
        """
        logger.info(
            f"Training PyTorch classifier for {len(questions)} diseases on {embeddings.shape[0]} samples"
        )
        # Log training configuration to WandB if available
        if self.use_wandb and wandb.run is not None:
            wandb.config.update(
                {
                    "training/batch_size": self.batch_size,
                    "training/learning_rate": self.learning_rate,
                    "training/num_epochs": self.num_epochs,
                    "training/weight_decay": self.weight_decay,
                    "training/num_diseases": len(questions),
                    "training/num_samples": embeddings.shape[0],
                    "training/embedding_dim": embeddings.shape[1],
                    "training/device": str(self.device),
                }
            )
            logger.info("Training configuration logged to WandB")

        # Calculate class weights for balancing
        pos_weights = self._calculate_pos_weights(labels_dict, questions)
        logger.info(f"Calculated pos_weights for {len(questions)} diseases")
        logger.info(
            f"Pos weights range: min={pos_weights.min().item():.3f}, "
            f"max={pos_weights.max().item():.3f}, mean={pos_weights.mean().item():.3f}"
        )

        # Create model and move to GPU
        model = MultiTaskLinearClassifier(
            input_dim=embeddings.shape[1], num_diseases=len(questions)
        ).to(self.device)

        # Setup loss and optimizer with weight decay
        criterion = ClassBalancedBCELoss(pos_weights)
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,  # L2 regularization via weight decay
        )

        # Create dataset and dataloader
        dataset = EmbeddingDataset(embeddings, labels_dict, questions)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        # Training loop
        model.train()
        total_batches = len(dataloader) * self.num_epochs

        with tqdm(total=total_batches, desc="Training PyTorch classifier") as pbar:
            for epoch in range(self.num_epochs):
                epoch_loss = 0.0
                for batch_embeddings, batch_labels in dataloader:
                    # Move to GPU
                    batch_embeddings = batch_embeddings.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    # Forward pass
                    optimizer.zero_grad()
                    logits = model(batch_embeddings)
                    loss = criterion(logits, batch_labels)

                    # Check for NaN loss and exit early if detected (only if NaN checking enabled)
                    if self.check_nan and torch.isnan(loss):
                        logger.warning(
                            f"NaN loss detected at epoch {epoch + 1}! Skipping this batch."
                        )
                        # Debug: Print input and weights sums
                        input_sum = batch_embeddings.sum().item()
                        weights_sum = sum(param.sum().item() for param in model.parameters())
                        logger.debug(f"Input sum: {input_sum}, Weights sum: {weights_sum}")
                        continue

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    pbar.update(1)
                    pbar.set_postfix({"epoch": epoch + 1, "loss": f"{loss.item():.4f}"})

                # Log epoch statistics
                avg_loss = epoch_loss / len(dataloader)
                # Log to WandB
                if self.use_wandb and wandb.run is not None:
                    wandb.log(
                        {
                            "training/epoch": epoch + 1,
                            "training/train_loss": avg_loss,
                            "training/batch_loss": loss.item(),
                        }
                    )
                if (epoch + 1) % 20 == 0:
                    logger.info(f"Epoch {epoch + 1}/{self.num_epochs}, Avg Loss: {avg_loss:.4f}")

        logger.info(f"PyTorch training completed in {self.num_epochs} epochs")
        return model


class PyTorchEvaluator:
    """Evaluation that maintains sklearn compatibility."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = get_config_value(config, "evaluation.pytorch.batch_size")
        self.num_workers = int(get_config_value(config, "evaluation.pytorch.num_workers"))

    def evaluate_model(
        self,
        model: MultiTaskLinearClassifier,
        test_embeddings: np.ndarray,
        test_labels_dict: Dict[str, np.ndarray],
        questions: List[str],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Evaluate PyTorch model and return sklearn-compatible results.

        Args:
            model: Trained PyTorch model
            test_embeddings: Test embeddings of shape (N, D)
            test_labels_dict: Dictionary mapping questions to test labels
            questions: List of questions

        Returns:
            Dictionary mapping questions to prediction results
        """
        model.eval()
        test_dataset = EmbeddingDataset(test_embeddings, test_labels_dict, questions)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        all_probs = []
        all_preds = []
        all_labels = []

        logger.info(f"Evaluating PyTorch model on {len(test_dataset)} samples")

        with torch.no_grad():
            for batch_embeddings, batch_labels in tqdm(test_loader, desc="Evaluating"):
                batch_embeddings = batch_embeddings.to(self.device)
                logits = model(batch_embeddings)

                # Convert to probabilities and predictions
                probs = torch.sigmoid(logits)  # Convert logits to probabilities
                preds = (probs > 0.5).float()  # Binary predictions

                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())
                all_labels.append(batch_labels)

        # Concatenate all results
        all_probs = torch.cat(all_probs, dim=0)  # (N, num_diseases)
        all_preds = torch.cat(all_preds, dim=0)  # (N, num_diseases)
        all_labels = torch.cat(all_labels, dim=0)  # (N, num_diseases)

        # Generate sklearn-compatible results per question
        results = {}
        for i, question in enumerate(questions):
            results[question] = {
                "probabilities": all_probs[:, i].numpy(),
                "predictions": all_preds[:, i].numpy(),
                "true_labels": all_labels[:, i].numpy(),
            }

        logger.info(f"PyTorch evaluation completed for {len(questions)} diseases")
        return results


class EmbeddingEvaluator:
    """
    Enhanced evaluator for medical image embeddings with disease finding classification.

    This evaluator loads cached embeddings, trains logistic regression models for each
    disease finding, and provides comprehensive evaluation metrics.
    """

    def __init__(self, config: dict):
        self.config = config
        self.questions = None  # Will be set when loading labels
        self.models = {}

        # Configuration from config file
        self.treat_na_as_no = get_config_value(config, "evaluation.treat_na_as_no")
        self.check_nan = get_config_value(config, "evaluation.check_nan")

        # Classifier configuration
        self.use_pytorch = get_config_value(config, "evaluation.use_pytorch")

        # WandB configuration for evaluation (always set)
        self.use_wandb = get_config_value(config, "evaluation.use_wandb")

        # PyTorch configuration
        if self.use_pytorch:
            self.batch_size = int(get_config_value(config, "evaluation.pytorch.batch_size"))
            self.learning_rate = float(get_config_value(config, "evaluation.pytorch.learning_rate"))
            self.num_epochs = int(get_config_value(config, "evaluation.pytorch.num_epochs"))
            self.weight_decay = float(get_config_value(config, "evaluation.pytorch.weight_decay"))
            self.num_workers = int(get_config_value(config, "evaluation.pytorch.num_workers"))

        # Sklearn fallback configuration
        self.solver = get_config_value(config, "evaluation.sklearn.solver")
        self.max_iter = get_config_value(config, "evaluation.sklearn.max_iter")
        self.C = get_config_value(config, "evaluation.sklearn.C")
        self.class_weight = get_config_value(config, "evaluation.sklearn.class_weight")

        if self.use_pytorch:
            logger.info("Initialized EmbeddingEvaluator with PyTorch GPU acceleration")
        else:
            logger.info("Initialized EmbeddingEvaluator with sklearn fallback")
        logger.info("Questions will be discovered from labels")
        logger.info(f"WandB evaluation logging: {'enabled' if self.use_wandb else 'disabled'}")

    def _handle_nan_labels(
        self, labels: np.ndarray, embeddings: np.ndarray, context: str
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Handle NaN labels according to configuration. Returns (labels, embeddings, should_skip)."""
        if np.all(pd.isna(labels)):
            logger.warning(f"Skipping {context}: all labels are NaN")
            return labels, embeddings, True

        if self.treat_na_as_no:
            return np.where(pd.isna(labels), 0, labels), embeddings, False
        else:
            valid_mask = ~pd.isna(labels)
            if not np.any(valid_mask):
                logger.warning(f"Skipping {context}: no valid labels")
                return labels, embeddings, True
            return labels[valid_mask], embeddings[valid_mask], False

    def _find_answer_for_question(self, question: str, qa_results: dict) -> Optional[int]:
        """Find binary answer (0/1) for a question in qa_results."""
        for questions_list in qa_results.values():
            if isinstance(questions_list, list):
                for qa_pair in questions_list:
                    if isinstance(qa_pair, dict) and question in qa_pair:
                        answer = qa_pair[question]
                        # Handle both string and non-string (e.g., NaN) answers
                        if isinstance(answer, str):
                            return 1 if answer.lower() == "yes" else 0
                        elif pd.isna(answer):
                            return None  # Return None for NaN values
                        else:
                            # Handle other non-string values (e.g., numeric)
                            return int(answer) if answer else 0
        return None

    def _extract_labels_for_accessions(
        self, accessions: List[str], labels_data: dict
    ) -> Dict[str, np.ndarray]:
        """Extract labels for all questions from accessions."""
        labels_dict = {question: [] for question in self.questions}
        default_value = 0 if self.treat_na_as_no else np.nan

        logger.info(
            f"Extracting labels for {len(accessions)} accessions across {len(self.questions)} questions"
        )
        for acc in tqdm(accessions, desc="Extracting labels", unit="accession"):
            qa_results = labels_data.get(str(acc), {}).get("qa_results", {})

            for question in self.questions:
                answer = self._find_answer_for_question(question, qa_results)
                labels_dict[question].append(answer if answer is not None else default_value)

        return {q: np.array(labels) for q, labels in labels_dict.items()}

    def _get_class_distribution(self, labels: np.ndarray) -> Tuple[int, int]:
        """Get positive and negative sample counts."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        pos_samples = counts[unique_labels == 1][0] if 1 in unique_labels else 0
        neg_samples = counts[unique_labels == 0][0] if 0 in unique_labels else 0
        return int(pos_samples), int(neg_samples)

    def _calculate_split_metrics(
        self,
        all_question_metrics: List[Dict],
        threshold_name: str,
        threshold_percentage: float,
        total_samples: int,
    ) -> Dict[str, float]:
        """
        Split questions into common and rare groups based on threshold, calculate metrics for each.

        Args:
            all_question_metrics: List of metrics for all questions
            threshold_name: Name of the threshold (for logging)
            threshold_percentage: Percentage threshold for considering a question "rare"
            total_samples: Total number of test samples

        Returns:
            Dictionary with metrics for common and rare groups
        """
        if threshold_percentage == 0:
            # Special case: drop zeros - exclude questions with 0 samples in either class
            threshold_count = 1
        else:
            # Calculate minimum count based on percentage
            threshold_count = int((threshold_percentage / 100.0) * total_samples)

        # Split questions into common and rare groups
        common_questions = []
        rare_questions = []

        for q_metrics in all_question_metrics:
            num_positive = q_metrics["num_positive"]
            num_negative = q_metrics["num_negative"]

            # Skip questions with 0 samples in either class if threshold is 0
            if threshold_percentage == 0 and (num_positive == 0 or num_negative == 0):
                continue

            # Otherwise, classify as rare if either class is below threshold
            if num_positive < threshold_count or num_negative < threshold_count:
                rare_questions.append(q_metrics)
            else:
                common_questions.append(q_metrics)

        # Calculate metrics for each group
        results = {}

        # Common group metrics
        if common_questions:
            common_metrics = self._calculate_group_metrics(common_questions, "common")
            for metric, value in common_metrics.items():
                results[f"common_{metric}"] = value
        else:
            logger.info(f"No common questions for threshold {threshold_name}")
            for metric in ["accuracy", "precision", "recall", "f1", "auc", "specificity"]:
                results[f"common_{metric}"] = 0.0
            results["common_count"] = 0

        # Rare group metrics
        if rare_questions:
            rare_metrics = self._calculate_group_metrics(rare_questions, "rare")
            for metric, value in rare_metrics.items():
                results[f"rare_{metric}"] = value
        else:
            logger.info(f"No rare questions for threshold {threshold_name}")
            for metric in ["accuracy", "precision", "recall", "f1", "auc", "specificity"]:
                results[f"rare_{metric}"] = 0.0
            results["rare_count"] = 0

        # Add overall statistics
        results["threshold_name"] = threshold_name
        results["threshold_percentage"] = threshold_percentage
        results["threshold_count"] = threshold_count
        results["total_questions"] = len(common_questions) + len(rare_questions)

        logger.info(
            f"Threshold '{threshold_name}' ({threshold_percentage}%): "
            f"{len(common_questions)} common, {len(rare_questions)} rare questions"
        )

        return results

    def _calculate_group_metrics(self, questions: List[Dict], group_name: str) -> Dict[str, float]:
        """Calculate average metrics for a group of questions."""
        metric_sums = {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "auc": 0,
            "specificity": 0,
        }

        for q_metrics in questions:
            for metric in metric_sums:
                metric_sums[metric] += q_metrics[metric]

        # Average the metrics
        num_questions = len(questions)
        for metric in metric_sums:
            metric_sums[metric] /= num_questions

        # Add count
        metric_sums["count"] = num_questions

        return metric_sums

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate all evaluation metrics."""
        metrics = {}
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        metrics.update({"precision": precision, "recall": recall, "f1": f1})

        # AUC with fallback
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        except ValueError:
            metrics["auc"] = 0.0

        # Specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {k: float(v) for k, v in metrics.items()}

    def load_embeddings_from_checkpoint(
        self, checkpoint_dir: str, dataset_name: str, split: str, labels_json_path: str
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
        """
        Load embeddings from checkpoint directory and labels from JSON.

        Args:
            checkpoint_dir: Path to checkpoint directory
            dataset_name: Name of the dataset
            split: Dataset split (train/valid/test)
            labels_json_path: Path to JSON file with labels (qa_results format)

        Returns:
            Tuple of (embeddings array, labels dictionary, accessions list)
        """
        logger.info(f"Loading embeddings from checkpoint directory: {checkpoint_dir}")

        # Create checkpoint manager to load embeddings
        # Use a placeholder model name since we're loading from a specific directory
        from .common import SimpleCheckpointManager

        checkpoint_manager = SimpleCheckpointManager(
            model_name="auto",  # Placeholder - will load all embeddings from the directory
            dataset_name=dataset_name,
            split=split,
            cache_dir=checkpoint_dir,
        )

        # Load all embeddings from individual checkpoint files (without model filtering)
        cached_embeddings, cached_accessions = (
            checkpoint_manager.get_all_embeddings_from_directory()
        )

        if len(cached_embeddings) == 0:
            raise ValueError(f"No embeddings found in checkpoint directory: {checkpoint_dir}")

        # Handle extra dimensions (e.g., CNN extractors emitting shape (N, 1, D))
        if len(cached_embeddings.shape) == 3 and cached_embeddings.shape[1] == 1:
            cached_embeddings = cached_embeddings.squeeze(1)
            logger.info(f"Squeezed embeddings from 3D to 2D: {cached_embeddings.shape}")
        elif len(cached_embeddings.shape) > 2:
            logger.warning(
                f"Unexpected embedding shape: {cached_embeddings.shape}. Flattening to 2D."
            )
            cached_embeddings = cached_embeddings.reshape(cached_embeddings.shape[0], -1)

        # Discover questions if not already done
        if self.questions is None:
            self.questions = discover_questions_from_json(labels_json_path)
            logger.info(f"Using {len(self.questions)} questions for evaluation")

        logger.info(f"Loading labels from {labels_json_path}")
        with open(labels_json_path, "r") as f:
            labels_data = json.load(f)

        # Extract labels for cached accessions using discovered questions
        labels_dict = self._extract_labels_for_accessions(cached_accessions, labels_data)

        logger.info(
            f"Loaded {cached_embeddings.shape[0]} samples with {cached_embeddings.shape[1]} features"
        )
        logger.info(f"Sample accessions: {cached_accessions[:5]}")
        return cached_embeddings, labels_dict, cached_accessions

    def train_linear_classifiers(
        self, train_embeddings: np.ndarray, train_labels: Dict[str, np.ndarray]
    ):
        """
        Train classifiers for each question using PyTorch or sklearn.

        Args:
            train_embeddings: Training embeddings array (N, D)
            train_labels: Dictionary of training labels for each question

        Returns:
            Dictionary of trained classifiers or PyTorch model
        """
        if self.use_pytorch:
            return self._train_pytorch_classifiers(train_embeddings, train_labels)
        else:
            return self._train_sklearn_classifiers(train_embeddings, train_labels)

    def _train_pytorch_classifiers(
        self, train_embeddings: np.ndarray, train_labels: Dict[str, np.ndarray]
    ) -> MultiTaskLinearClassifier:
        """Train PyTorch multi-task classifier."""
        logger.info(f"Training PyTorch multi-task classifier for {len(self.questions)} questions")
        logger.info(
            f"Using batch_size={self.batch_size}, lr={self.learning_rate}, "
            f"epochs={self.num_epochs}, weight_decay={self.weight_decay}"
        )

        trainer = PyTorchClassifierTrainer(self.config)
        pytorch_model = trainer.train(train_embeddings, train_labels, self.questions)

        # Store training stats
        self.training_stats = self._calculate_pytorch_training_stats(train_labels)

        return pytorch_model

    def _train_sklearn_classifiers(
        self, train_embeddings: np.ndarray, train_labels: Dict[str, np.ndarray]
    ) -> Dict[str, LogisticRegression]:
        """Train individual sklearn classifiers (fallback)."""
        logger.info(f"Training sklearn classifiers for {len(self.questions)} questions")
        logger.info(f"Using solver: {self.solver}, C: {self.C}, max_iter: {self.max_iter}")

        classifiers = {}
        training_stats = {}

        for question in tqdm(self.questions, desc="Training sklearn classifiers", unit="question"):
            labels = train_labels[question]
            context = f"question '{question[:50]}...'"

            # Handle NaN values
            labels, embeddings, should_skip = self._handle_nan_labels(
                labels, train_embeddings, context
            )
            if should_skip:
                continue

            # Check class distribution
            pos_samples, neg_samples = self._get_class_distribution(labels)
            training_stats[question] = {
                "positive_samples": pos_samples,
                "negative_samples": neg_samples,
                "total_samples": len(labels),
            }

            # Skip if no positive samples
            if pos_samples == 0:
                logger.warning(f"Skipping {context}: no positive samples found")
                continue

            # Train logistic regression with configurable parameters
            try:
                clf = LogisticRegression(
                    solver=self.solver,
                    class_weight=self.class_weight,
                    C=self.C,
                    max_iter=self.max_iter,
                    fit_intercept=False,
                    random_state=42,
                )
                clf.fit(embeddings, labels)
                classifiers[question] = clf

                logger.debug(
                    f"Trained classifier for {context}: "
                    f"{pos_samples} pos, {neg_samples} neg samples"
                )

            except Exception as e:
                logger.error(f"Failed to train classifier for {context}: {e}")
                continue

        logger.info(f"Successfully trained {len(classifiers)} sklearn classifiers")
        self.training_stats = training_stats
        return classifiers

    def _calculate_pytorch_training_stats(
        self, train_labels: Dict[str, np.ndarray]
    ) -> Dict[str, Dict]:
        """Calculate training statistics for PyTorch training."""
        training_stats = {}

        for question in self.questions:
            if question not in train_labels:
                continue

            labels = train_labels[question]
            # Handle NaN values
            labels, _, should_skip = self._handle_nan_labels(
                labels, np.zeros((len(labels), 1)), f"question '{question[:50]}...'"
            )
            if should_skip:
                continue

            pos_samples, neg_samples = self._get_class_distribution(labels)
            training_stats[question] = {
                "positive_samples": pos_samples,
                "negative_samples": neg_samples,
                "total_samples": len(labels),
            }

        return training_stats

    def evaluate_classifiers(
        self,
        test_embeddings: np.ndarray,
        test_labels: Dict[str, np.ndarray],
        test_accessions: List[str],
        model_name: str = "model",
        pool_op: str = "mean",
    ) -> Dict[str, Any]:
        """
        Evaluate trained classifiers on test data.

        Args:
            test_embeddings: Test embeddings array (N, D)
            test_labels: Dictionary of test labels for each question
            test_accessions: List of accession IDs for test samples
            model_name: Name of the model being evaluated
            pool_op: Pooling operation used

        Returns:
            Dictionary with evaluation results
        """
        # Fixed threshold configurations for splitting common/rare questions
        threshold_configs = [
            ("full", -1),  # All questions (no threshold)
            ("drop_zeros", 0),  # Drop questions with 0 samples
        ]

        if self.use_pytorch and isinstance(self.models, MultiTaskLinearClassifier):
            return self._evaluate_pytorch_classifier(
                test_embeddings,
                test_labels,
                test_accessions,
                model_name,
                pool_op,
                threshold_configs,
            )
        else:
            return self._evaluate_sklearn_classifiers(
                test_embeddings,
                test_labels,
                test_accessions,
                model_name,
                pool_op,
                threshold_configs,
            )

    def _evaluate_pytorch_classifier(
        self,
        test_embeddings: np.ndarray,
        test_labels: Dict[str, np.ndarray],
        test_accessions: List[str],
        model_name: str,
        pool_op: str,
        threshold_configs: List[Tuple[str, float]],
    ) -> Dict[str, Any]:
        """Evaluate PyTorch multi-task classifier."""
        logger.info(f"Evaluating PyTorch classifier on {len(test_accessions)} test samples")

        # Use PyTorch evaluator
        evaluator = PyTorchEvaluator(self.config)
        pytorch_results = evaluator.evaluate_model(
            self.models, test_embeddings, test_labels, self.questions
        )

        # Convert to sklearn-compatible format for existing pipeline
        return self._convert_pytorch_results_to_sklearn_format(
            pytorch_results, test_accessions, model_name, pool_op, threshold_configs
        )

    def _evaluate_sklearn_classifiers(
        self,
        test_embeddings: np.ndarray,
        test_labels: Dict[str, np.ndarray],
        test_accessions: List[str],
        model_name: str,
        pool_op: str,
        threshold_configs: List[Tuple[str, float]],
    ) -> Dict[str, Any]:
        """Evaluate individual sklearn classifiers."""
        logger.info(f"Evaluating {len(self.questions)} sklearn classifiers on test data")

        detailed_results = []
        metric_sums = {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "auc": 0,
            "specificity": 0,
        }
        evaluated_count = 0

        # Store probabilities for each sample and question
        all_probabilities = []

        # Store metrics for all questions (unfiltered)
        all_question_metrics = []

        for question in tqdm(
            self.questions, desc="Evaluating sklearn classifiers", unit="question"
        ):
            if question not in self.models:
                continue

            clf = self.models[question]
            labels = test_labels[question]
            context = f"question '{question[:50]}...'"

            # Handle NaN values
            labels, embeddings, should_skip = self._handle_nan_labels(
                labels, test_embeddings, context
            )
            if should_skip:
                continue

            # Create mask for samples with valid labels (to match probabilities to accessions)
            if self.treat_na_as_no:
                valid_mask = np.ones(len(test_labels[question]), dtype=bool)
            else:
                valid_mask = ~pd.isna(test_labels[question])

            try:
                # Get predictions and probabilities
                y_pred = clf.predict(embeddings)
                y_prob = (
                    clf.predict_proba(embeddings)[:, 1]
                    if len(np.unique(labels)) > 1
                    else np.zeros_like(labels)
                )

                # Store probabilities for each sample
                for i, (prob, pred, true_label) in enumerate(zip(y_prob, y_pred, labels)):
                    # Find the original sample index in the full dataset
                    sample_idx = np.where(valid_mask)[0][i] if not self.treat_na_as_no else i
                    all_probabilities.append(
                        {
                            "accession": test_accessions[sample_idx],
                            "question": question,
                            "probability": float(prob),
                            "prediction": int(pred),
                            "true_label": int(true_label),
                        }
                    )

                # Calculate metrics
                metrics = self._calculate_metrics(labels, y_pred, y_prob)

                # Store detailed results
                question_result = {
                    "question": question,
                    "model_name": model_name,
                    "pool_op": pool_op,
                    **metrics,
                    "num_samples": len(labels),
                    "num_positive": int(np.sum(labels)),
                    "num_negative": int(len(labels) - np.sum(labels)),
                }
                detailed_results.append(question_result)
                all_question_metrics.append(question_result)

                # Add to running sums for averaging
                for metric in metric_sums:
                    metric_sums[metric] += metrics[metric]
                evaluated_count += 1

            except Exception as e:
                logger.error(f"Error evaluating {context}: {e}")
                continue

        # Calculate averages
        if evaluated_count > 0:
            for metric in metric_sums:
                metric_sums[metric] /= evaluated_count

        # Calculate summary stats
        summary_stats = {
            "model_name": model_name,
            "pool_op": pool_op,
            "total_findings": len(self.questions),
            "evaluated_findings": evaluated_count,
            **{f"avg_{metric}": metric_sums[metric] for metric in metric_sums},
        }

        # Add split metrics for different threshold configurations
        total_test_samples = len(test_accessions)
        for threshold_name, threshold_pct in threshold_configs:
            if threshold_pct == -1:
                # Special case for "full" - just use overall metrics
                for metric in ["accuracy", "precision", "recall", "f1", "auc", "specificity"]:
                    summary_stats[f"{threshold_name}_common_{metric}"] = summary_stats[
                        f"avg_{metric}"
                    ]
                    summary_stats[f"{threshold_name}_rare_{metric}"] = summary_stats[
                        f"avg_{metric}"
                    ]
                summary_stats[f"{threshold_name}_common_count"] = evaluated_count
                summary_stats[f"{threshold_name}_rare_count"] = evaluated_count
            else:
                split_stats = self._calculate_split_metrics(
                    all_question_metrics, threshold_name, threshold_pct, total_test_samples
                )
                # Add split stats to summary
                for key, value in split_stats.items():
                    if not key.startswith("threshold_"):  # Skip meta fields
                        summary_stats[f"{threshold_name}_{key}"] = value

        logger.info(f"Evaluated {evaluated_count} questions")
        logger.info(
            f"Average metrics - Accuracy: {summary_stats['avg_accuracy']:.3f}, "
            f"F1: {summary_stats['avg_f1']:.3f}, AUC: {summary_stats['avg_auc']:.3f}"
        )
        logger.info(f"Collected probabilities for {len(all_probabilities)} exam-question pairs")

        return {
            "detailed_results": detailed_results,
            "summary_stats": summary_stats,
            "training_stats": getattr(self, "training_stats", {}),
            "probabilities": all_probabilities,
        }

    def _convert_pytorch_results_to_sklearn_format(
        self,
        pytorch_results: Dict[str, Dict[str, np.ndarray]],
        test_accessions: List[str],
        model_name: str,
        pool_op: str,
        threshold_configs: List[Tuple[str, float]],
    ) -> Dict[str, Any]:
        """Convert PyTorch evaluation results to sklearn-compatible format."""
        detailed_results = []
        metric_sums = {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "auc": 0,
            "specificity": 0,
        }
        evaluated_count = 0

        # Store probabilities for each sample and question
        all_probabilities = []

        # Store metrics for all questions (unfiltered)
        all_question_metrics = []

        for question in tqdm(self.questions, desc="Converting PyTorch results", unit="question"):
            if question not in pytorch_results:
                continue

            result = pytorch_results[question]
            y_true = result["true_labels"]
            y_pred = result["predictions"]
            y_prob = result["probabilities"]

            # Store probabilities for each sample
            for i, (prob, pred, true_label) in enumerate(zip(y_prob, y_pred, y_true)):
                all_probabilities.append(
                    {
                        "accession": test_accessions[i],
                        "question": question,
                        "probability": float(prob),
                        "prediction": int(pred),
                        "true_label": int(true_label),
                    }
                )

            # Calculate metrics using existing helper
            metrics = self._calculate_metrics(y_true, y_pred, y_prob)

            # Store detailed results
            question_result = {
                "question": question,
                "model_name": model_name,
                "pool_op": pool_op,
                **metrics,
                "num_samples": len(y_true),
                "num_positive": int(np.sum(y_true)),
                "num_negative": int(len(y_true) - np.sum(y_true)),
            }
            detailed_results.append(question_result)
            all_question_metrics.append(question_result)

            # Add to running sums for averaging
            for metric in metric_sums:
                metric_sums[metric] += metrics[metric]
            evaluated_count += 1

        # Calculate averages
        if evaluated_count > 0:
            for metric in metric_sums:
                metric_sums[metric] /= evaluated_count

        # Calculate summary stats
        summary_stats = {
            "model_name": model_name,
            "pool_op": pool_op,
            "total_findings": len(self.questions),
            "evaluated_findings": evaluated_count,
            **{f"avg_{metric}": metric_sums[metric] for metric in metric_sums},
        }

        # Add split metrics for different threshold configurations
        total_test_samples = len(test_accessions)
        for threshold_name, threshold_pct in threshold_configs:
            if threshold_pct == -1:
                # Special case for "full" - just use overall metrics
                for metric in ["accuracy", "precision", "recall", "f1", "auc", "specificity"]:
                    summary_stats[f"{threshold_name}_common_{metric}"] = summary_stats[
                        f"avg_{metric}"
                    ]
                    summary_stats[f"{threshold_name}_rare_{metric}"] = summary_stats[
                        f"avg_{metric}"
                    ]
                summary_stats[f"{threshold_name}_common_count"] = evaluated_count
                summary_stats[f"{threshold_name}_rare_count"] = evaluated_count
            else:
                split_stats = self._calculate_split_metrics(
                    all_question_metrics, threshold_name, threshold_pct, total_test_samples
                )
                # Add split stats to summary
                for key, value in split_stats.items():
                    if not key.startswith("threshold_"):  # Skip meta fields
                        summary_stats[f"{threshold_name}_{key}"] = value

        logger.info(f"Converted PyTorch results for {evaluated_count} questions")
        logger.info(
            f"Average metrics - Accuracy: {summary_stats['avg_accuracy']:.3f}, "
            f"F1: {summary_stats['avg_f1']:.3f}, AUC: {summary_stats['avg_auc']:.3f}"
        )
        logger.info(f"Collected probabilities for {len(all_probabilities)} exam-question pairs")

        return {
            "detailed_results": detailed_results,
            "summary_stats": summary_stats,
            "training_stats": getattr(self, "training_stats", {}),
            "probabilities": all_probabilities,
        }

    def save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """
        Save evaluation results to CSV and JSON files.

        Args:
            results: Results dictionary from evaluate_classifiers
            output_dir: Output directory path
        """
        output_path = ensure_dir_exists(output_dir)

        # Save detailed results as CSV
        detailed_df = pd.DataFrame(results["detailed_results"])
        detailed_csv_path = output_path / "detailed_results.csv"
        detailed_df.to_csv(detailed_csv_path, index=False)
        logger.info(f"Saved detailed results to {detailed_csv_path}")

        # Save probabilities as CSV if available
        if "probabilities" in results and results["probabilities"]:
            probabilities_df = pd.DataFrame(results["probabilities"])
            probabilities_csv_path = output_path / "exam_probabilities.csv"
            probabilities_df.to_csv(probabilities_csv_path, index=False)
            logger.info(f"Saved exam probabilities to {probabilities_csv_path}")
            logger.info(f"Probabilities saved for {len(probabilities_df)} exam-question pairs")

        # Save summary stats as JSON
        summary_json_path = output_path / "summary_stats.json"
        with open(summary_json_path, "w") as f:
            json.dump(results["summary_stats"], f, indent=2)
        logger.info(f"Saved summary stats to {summary_json_path}")

        # Save training stats if available
        if "training_stats" in results:
            training_json_path = output_path / "training_stats.json"
            with open(training_json_path, "w") as f:
                json.dump(results["training_stats"], f, indent=2)
            logger.info(f"Saved training stats to {training_json_path}")

    def run_full_evaluation(
        self,
        checkpoint_dir: str,
        dataset_name: str,
        labels_json_path: str,
        pool_op: str = "mean",
        output_dir: str = "results",
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline from checkpoint directory.

        Args:
            checkpoint_dir: Path to checkpoint directory containing embeddings
            dataset_name: Name of the dataset
            labels_json_path: Path to JSON file with labels (qa_results format)
            pool_op: Pooling operation used
            output_dir: Directory to save results

        Returns:
            Dictionary with complete evaluation results
        """
        start_time = time.time()
        logger.info("Starting full embedding evaluation pipeline")

        try:
            # Load training data
            step_start = time.time()
            logger.info("Step 1/5: Loading training embeddings...")
            train_embeddings, train_labels, train_accessions = self.load_embeddings_from_checkpoint(
                checkpoint_dir, dataset_name, "train", labels_json_path
            )
            logger.info(f"Training data loaded in {time.time() - step_start:.1f}s")

            # Load test data
            step_start = time.time()
            logger.info("Step 2/5: Loading test embeddings...")
            test_embeddings, test_labels, test_accessions = self.load_embeddings_from_checkpoint(
                checkpoint_dir, dataset_name, "test", labels_json_path
            )
            logger.info(f"Test data loaded in {time.time() - step_start:.1f}s")

            # Train classifiers
            step_start = time.time()
            logger.info("Step 3/5: Training classifiers...")
            self.models = self.train_linear_classifiers(train_embeddings, train_labels)
            logger.info(f"Classifiers trained in {time.time() - step_start:.1f}s")

            # Evaluate classifiers
            step_start = time.time()
            logger.info("Step 4/5: Evaluating classifiers...")
            results = self.evaluate_classifiers(
                test_embeddings, test_labels, test_accessions, "auto", pool_op
            )
            logger.info(f"Evaluation completed in {time.time() - step_start:.1f}s")

            # Log evaluation metrics to WandB
            if self.use_wandb and wandb.run is not None and "summary_stats" in results:
                summary = results["summary_stats"]

                # Log summary metrics
                wandb_metrics = {
                    "evaluation/avg_accuracy": summary.get("avg_accuracy", 0),
                    "evaluation/avg_precision": summary.get("avg_precision", 0),
                    "evaluation/avg_recall": summary.get("avg_recall", 0),
                    "evaluation/avg_f1": summary.get("avg_f1", 0),
                    "evaluation/avg_auc": summary.get("avg_auc", 0),
                    "evaluation/avg_specificity": summary.get("avg_specificity", 0),
                    "evaluation/total_findings": summary.get("total_findings", 0),
                    "evaluation/evaluated_findings": summary.get("evaluated_findings", 0),
                    "data/num_train_samples": len(train_accessions),
                    "data/num_test_samples": len(test_accessions),
                    "data/embedding_dim": (
                        train_embeddings.shape[1] if len(train_embeddings.shape) > 1 else 0
                    ),
                }

                # Log split metrics for each threshold configuration
                # Known threshold names from our fixed configuration
                known_thresholds = ["full", "drop_zeros"]

                for key, value in summary.items():
                    # Check for threshold-specific metrics (format: thresholdname_group_metricname)
                    for threshold_name in known_thresholds:
                        prefix = f"{threshold_name}_"
                        if key.startswith(prefix):
                            # Extract group (common/rare) and metric name
                            remaining = key[len(prefix) :]
                            if remaining.startswith("common_"):
                                group = "common"
                                metric_name = remaining[7:]  # len("common_")
                            elif remaining.startswith("rare_"):
                                group = "rare"
                                metric_name = remaining[5:]  # len("rare_")
                            else:
                                continue

                            wandb_key = f"evaluation/{threshold_name}/{group}/{metric_name}"
                            wandb_metrics[wandb_key] = value

                wandb.log(wandb_metrics)
                logger.info("Logged evaluation metrics to WandB")

            # Save results
            step_start = time.time()
            logger.info("Step 5/5: Saving results...")
            self.save_results(results, output_dir)
            logger.info(f"Results saved in {time.time() - step_start:.1f}s")

            total_time = time.time() - start_time
            logger.info(
                f"Completed full embedding evaluation pipeline in {total_time:.1f}s ({total_time/60:.1f} minutes)"
            )
            return results

        except Exception as e:
            logger.error(f"Error in full evaluation pipeline: {e}")
            raise RATEEvalError(f"Evaluation failed: {e}")


def evaluate_embeddings(
    checkpoint_dir: str, dataset_name: str, labels_json_path: str, config: dict, **kwargs
) -> Dict[str, Any]:
    """
    Run embedding evaluation with simplified interface.

    Args:
        checkpoint_dir: Path to checkpoint directory containing embeddings
        dataset_name: Name of the dataset
        labels_json_path: Path to JSON file with labels (qa_results format)
        config: Configuration dictionary
        **kwargs: Additional arguments (pool_op, output_dir, etc.)

    Returns:
        Dictionary with evaluation results
    """
    evaluator = EmbeddingEvaluator(config)

    return evaluator.run_full_evaluation(
        checkpoint_dir=checkpoint_dir,
        dataset_name=dataset_name,
        labels_json_path=labels_json_path,
        pool_op=kwargs.get("pool_op", "mean"),
        output_dir=kwargs.get("output_dir", "results"),
    )
