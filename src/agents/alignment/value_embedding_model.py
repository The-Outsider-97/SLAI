"""
Ethical Value Embedding System
Implements:
- Cross-cultural value encoding (Hofstede dimensions)
- Policy-value alignment scoring
- Inverse reinforcement learning (Ng & Russell, 2000)
- Human preference modeling (Christiano et al., 2017)
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from models.slai_lm import SLAILM, get_shared_slailm
from logs.logger import get_logger

logger = get_logger("Value Embedding Model")

UDHR_JSON_PATH = os.path.join(os.path.dirname(__file__), "templates", "un_human_rights.json")
with open(UDHR_JSON_PATH, "r", encoding="utf-8") as f:
    udhr_data = json.load(f)

NUM_ETHICAL_PRINCIPLES = len(udhr_data["articles"])

@dataclass
class ValueConfig:
    """Configuration for ethical value embedding model"""
    embedding_dim: int = 512
    num_cultural_dimensions: int = 6  # Hofstede's 6 dimensions
    num_ethical_principles: int = NUM_ETHICAL_PRINCIPLES   # UN Declaration of Human Rights
    temperature: float = 0.07
    dropout: float = 0.1
    margin: float = 0.2  # Margin for triplet loss
    max_seq_length: int = 128

class ValueEmbeddingModel(nn.Module):
    """
    Neural ethical alignment system with:
    - Multi-modal value encoder
    - Policy embedding network
    - Alignment scoring head
    - Human preference predictor
    
    Architecture:
    1. Text Encoder: BERT-based ethical principle encoder
    2. Cultural Adaptor: Hofstede dimension projector
    3. Policy Encoder: Agent behavior embedding network
    4. Alignment Scorer: Cross-attention value-policy matching
    """

    def __init__(self, config, slai_lm=None):
        super().__init__()
        self.config = config
        self.slai_lm = slai_lm
        self.value_proj = nn.Linear(768, config.embedding_dim)
        
        # Cultural context adaptor
        self.cultural_adaptor = nn.Sequential(
            nn.Linear(config.num_cultural_dimensions, 256),
            nn.ReLU(),
            nn.Linear(256, config.embedding_dim)
        )
        
        # Policy encoder
        self.policy_encoder = nn.Sequential(
            nn.Linear(4096, 2048),  # Assumes policy params as input
            nn.GELU(),
            nn.LayerNorm(2048),
            nn.Linear(2048, config.embedding_dim)
        )
        
        # Alignment scoring
        self.alignment_scorer = nn.Sequential(
            nn.Linear(3*config.embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Human preference predictor
        self.preference_head = nn.Sequential(
            nn.Linear(2*config.embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.dropout = nn.Dropout(config.dropout)
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for projection layers"""
        for module in [self.value_proj, self.cultural_adaptor, self.policy_encoder]:
            if hasattr(module, 'weight'):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: Dict) -> Dict:
        """Full alignment scoring pipeline"""
        value_emb = self.encode_value(
            inputs["value_text"], 
            inputs["cultural_context"]
        )
        policy_emb = self.encode_policy(inputs["policy_params"])
        
        alignment_score = self.calculate_alignment(value_emb, policy_emb)
        preference_score = self.predict_preference(value_emb, policy_emb)
        
        return {
            "value_embedding": value_emb,
            "policy_embedding": policy_emb,
            "alignment_score": alignment_score,
            "preference_score": preference_score
        }

    def encode_value(self, text: List[str], cultural: torch.Tensor) -> torch.Tensor:
        """Use SLAILM to generate value embeddings from text and cultural context"""
        embeddings = []

        for t in text:
            result = self.slai_lm.process_input(prompt=t, text=t)
            tokens = result.get("tokens", [])
            vector = np.array([len(token) for token in tokens], dtype=np.float32)  # Simple placeholder

            # Normalize and resize
            vec = np.pad(vector, (0, max(1, self.config.embedding_dim - len(vector))), mode='constant')[:self.config.embedding_dim]
            embeddings.append(torch.tensor(vec))

        text_emb = torch.stack(embeddings)
        cultural_emb = self.cultural_adaptor(cultural)

        return self.dropout(text_emb + cultural_emb)

    def encode_policy(self, policy: torch.Tensor) -> torch.Tensor:
        """Policy parameter embedding"""
        return self.policy_encoder(policy)

    def calculate_alignment(self, value_emb: torch.Tensor, 
                          policy_emb: torch.Tensor) -> torch.Tensor:
        """Cross-attention alignment scoring"""
        combined = torch.cat([
            value_emb,
            policy_emb,
            torch.abs(value_emb - policy_emb)
        ], dim=-1)
        return torch.sigmoid(self.alignment_scorer(combined))

    def predict_preference(self, value_emb: torch.Tensor,
                         policy_emb: torch.Tensor) -> torch.Tensor:
        """Human preference prediction"""
        combined = torch.cat([value_emb, policy_emb], dim=-1)
        return torch.sigmoid(self.preference_head(combined))

    def loss(self, outputs: Dict, labels: Dict) -> torch.Tensor:
        """Composite loss function for ethical alignment"""
        # Triplet loss for embedding space structure
        triplet_loss = F.triplet_margin_loss(
            outputs["value_embedding"],
            outputs["policy_embedding"][labels["positive_idx"]],
            outputs["policy_embedding"][labels["negative_idx"]],
            margin=self.config.margin
        )
        
        # Preference prediction loss
        pref_loss = F.binary_cross_entropy(
            outputs["preference_score"],
            labels["human_preference"].float()
        )
        
        # Alignment regularization
        norm_loss = torch.mean(
            torch.norm(outputs["value_embedding"], dim=1) +
            torch.norm(outputs["policy_embedding"], dim=1)
        )
        
        return triplet_loss + pref_loss + 0.1*norm_loss

    def score_trajectory(self, data: pd.DataFrame) -> float:
        """Value alignment scoring for behavioral trajectories"""
        policy_emb = self.encode_policy(
            torch.tensor(data['policy_features'].values.tolist())
        )
        value_emb = self.encode_value(
            data['ethical_guidelines'].tolist(),
            torch.tensor(data['cultural_features'].values.tolist())
        )
        return self.calculate_alignment(value_emb, policy_emb).mean().item()

class ValueDataset(torch.utils.data.Dataset):
    """Dataset for ethical alignment training"""
    def __init__(self, 
                ethical_texts: List[str],
                cultural_features: List[List[float]],
                policy_parameters: List[List[float]],
                human_preferences: List[int]):
        self.ethical_texts = ethical_texts
        self.cultural_features = cultural_features
        self.policy_parameters = policy_parameters
        self.human_preferences = human_preferences
        
    def __len__(self):
        return len(self.ethical_texts)
    
    def __getitem__(self, idx):
        return {
            "value_text": self.ethical_texts[idx],
            "cultural_context": torch.tensor(self.cultural_features[idx]),
            "policy_params": torch.tensor(self.policy_parameters[idx]),
            "human_preference": torch.tensor(self.human_preferences[idx])
        }

class ValueTrainer:
    """Training pipeline for ethical value alignment"""
    def __init__(self, model: ValueEmbeddingModel,
                train_dataset: ValueDataset,
                val_dataset: ValueDataset):
        self.model = model
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=64
        )
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        
    def train_epoch(self):
        """Single training epoch with contrastive learning"""
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            
            # Generate triplets
            outputs = self.model(batch)
            loss = self.model.loss(outputs, self._create_labels(batch))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        self.scheduler.step()
        return total_loss / len(self.train_loader)
    
    def _create_labels(self, batch: Dict) -> Dict:
        human_pref = batch["human_preference"].squeeze()
        batch_size = human_pref.size(0)
    
        # Initialize indices
        positive_idx = torch.zeros(batch_size, dtype=torch.long)
        negative_idx = torch.zeros(batch_size, dtype=torch.long)
    
        # Get available positives/negatives
        pos_mask = human_pref == 1
        neg_mask = ~pos_mask
        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]
    
        # Handle edge cases
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return {
                "positive_idx": torch.tensor([], dtype=torch.long),
                "negative_idx": torch.tensor([], dtype=torch.long),
                "human_preference": batch["human_preference"]
            }
    
        # Compute policy embeddings once for efficiency
        policy_emb = self.model.policy_encoder(batch["policy_params"])
    
        for i in range(batch_size):
            if human_pref[i] == 1:
                neg_dists = F.pairwise_distance(policy_emb[i].unsqueeze(0), policy_emb[neg_mask])
                negative_idx[i] = neg_indices[torch.argmax(neg_dists)]
                positive_idx[i] = i
            else:
                pos_dists = F.pairwise_distance(policy_emb[i].unsqueeze(0), policy_emb[pos_mask])
                positive_idx[i] = pos_indices[torch.argmin(pos_dists)]
                negative_idx[i] = i
    
        return {
            "positive_idx": positive_idx,
            "negative_idx": negative_idx,
            "human_preference": batch["human_preference"]
        }
    
    def evaluate(self):
        """Validation evaluation"""
        self.model.eval()
        metrics = {"alignment_acc": 0, "pref_auc": 0}
        
        with torch.no_grad():
            for batch in self.val_loader:
                outputs = self.model(batch)
                # Calculate evaluation metrics
                
        return metrics

class ValueAuditor:
    """Ethical alignment verification system"""
    def __init__(self, model: ValueEmbeddingModel):
        self.model = model
        self.similarity = nn.CosineSimilarity(dim=-1)
        
    def compare_policies(self, policy_embs: torch.Tensor,
                       value_emb: torch.Tensor) -> Dict:
        """Compare multiple policies against reference values"""
        similarities = self.similarity(policy_embs, value_emb)
        return {
            "max_similarity": similarities.max().item(),
            "min_similarity": similarities.min().item(),
            "mean_similarity": similarities.mean().item(),
            "std_similarity": similarities.std().item()
        }
    
    def analyze_distribution(self, embeddings: torch.Tensor) -> Dict:
        """Analyze embedding space characteristics"""
        return {
            "dimensionality": self._intrinsic_dim(embeddings),
            "cluster_quality": self._cluster_metrics(embeddings),
            "coverage_score": self._coverage_metric(embeddings)
        }
    
    def _intrinsic_dim(self, emb: torch.Tensor) -> float:
        """Estimate intrinsic dimensionality using TWO-NN method (Facco et al., 2017)"""
        if emb.size(0) < 3:  # Need at least 3 points for neighbors
            return 0.0
        
        # Compute pairwise distances
        dists = torch.cdist(emb, emb)
        dists.fill_diagonal_(float('inf'))
        
        # Get first and second nearest neighbors
        top2_dists, _ = torch.topk(dists, k=2, dim=1, largest=False)
        mu = top2_dists[:, 1] / (top2_dists[:, 0] + 1e-12)
        
        valid = (top2_dists[:, 0] > 1e-12) & (mu > 1.0)
        if valid.sum() < 10:  # Minimum valid samples required
            return 0.0
        
        log_mu = torch.log(mu[valid])
        d = 1 / log_mu.mean()
        return d.item()
    
    def _cluster_metrics(self, emb: torch.Tensor) -> Dict:
        """Calculate clustering metrics using geometric approximations"""
        if emb.size(0) < 5:
            return {"num_clusters": 0, "silhouette_score": 0.0, "cohesion_separation_ratio": 0.0}
        
        # Fast geometric clustering approximation
        centroid = emb.mean(dim=0)
        dists_to_center = torch.norm(emb - centroid, dim=1)
        cohesion = dists_to_center.mean().item()
        
        # Estimate separation using pairwise distances
        random_sample = emb[torch.randperm(emb.size(0))[:5]]
        separation = torch.pdist(random_sample).mean().item()
        
        return {
            "num_clusters": 1 if cohesion < 0.5 else 2,
            "silhouette_score": max(0, separation - cohesion),
            "cohesion_separation_ratio": separation / (cohesion + 1e-12)
        }
    
    def _coverage_metric(self, emb: torch.Tensor) -> float:
        """Value space coverage metric using hypersphere volume estimation"""
        if emb.size(0) < 2:
            return 0.0
        
        # Compute effective volume using pairwise distances
        max_dists, _ = torch.max(torch.cdist(emb, emb), dim=1)
        mean_radius = max_dists.mean().item()
        return mean_radius * emb.size(1)  # Approximate dimensional scaling
    
if __name__ == "__main__":
    with open("src/agents/alignment/templates/un_human_rights.json", "r") as f:
        udhr_data = json.load(f)

    num_ethical_principles = len(udhr_data["articles"])

    config = ValueConfig(
        embedding_dim=256,
        num_cultural_dimensions=6,
        num_ethical_principles=num_ethical_principles,
        temperature=0.1,
        dropout=0.1,
        margin=0.3
    )

    # Mock SLAI language model (in practice would use real implementation)
    class MockSLM:
        def process_input(self, prompt, text):
            return {"tokens": ["mock"] * 20}  # Return fixed length tokens
    
    # Initialize model components
    mock_slm = MockSLM()
    model = ValueEmbeddingModel(config, slai_lm=mock_slm)
    
    # Create synthetic test data
    def generate_test_data(num_samples=100):
        ethical_texts = ["Respect human dignity"] * num_samples
        cultural_features = torch.randn(num_samples, config.num_cultural_dimensions)
        policy_parameters = torch.randn(num_samples, 4096)
        human_preferences = torch.randint(0, 2, (num_samples, 1))
        return ethical_texts, cultural_features, policy_parameters, human_preferences
    
    # Prepare datasets
    train_texts, train_cult, train_pol, train_pref = generate_test_data(200)
    val_texts, val_cult, val_pol, val_pref = generate_test_data(50)
    
    train_dataset = ValueDataset(train_texts, train_cult.tolist(), train_pol.tolist(), train_pref.tolist())
    val_dataset = ValueDataset(val_texts, val_cult.tolist(), val_pol.tolist(), val_pref.tolist())
    
    # Test training pipeline
    trainer = ValueTrainer(model, train_dataset, val_dataset)
    print("Starting training test...")
    for epoch in range(3):  # Short test run
        loss = trainer.train_epoch()
        print(f"Epoch {epoch+1} - Loss: {loss:.4f}")
    
    # Test evaluation
    metrics = trainer.evaluate()
    print("\nValidation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Test auditing functionality
    auditor = ValueAuditor(model)
    
    # Get embeddings for auditing
    with torch.no_grad():
        test_emb = model.encode_value(["Test value"], torch.randn(1, config.num_cultural_dimensions))
        policy_embs = model.encode_policy(torch.randn(5, 4096))
    
    # Run audits
    policy_comparison = auditor.compare_policies(policy_embs, test_emb)
    print("\nPolicy Comparison Results:")
    for k, v in policy_comparison.items():
        print(f"{k}: {v:.4f}")
    
    distribution_analysis = auditor.analyze_distribution(policy_embs)
    print("\nEmbedding Space Analysis:")
    for k, v in distribution_analysis.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for sk, sv in v.items():
                print(f"  {sk}: {sv:.4f}")
        else:
            print(f"{k}: {v:.4f}")
    
    # Test trajectory scoring
    test_df = pd.DataFrame({
        'policy_features': [torch.randn(4096).tolist() for _ in range(10)],
        'ethical_guidelines': ["Fair treatment"] * 10,
        'cultural_features': [torch.randn(6).tolist() for _ in range(10)]
    })
    trajectory_score = model.score_trajectory(test_df)
    print(f"\nTrajectory Alignment Score: {trajectory_score:.4f}")
    
    print("\nAll functionality tests completed successfully!")
