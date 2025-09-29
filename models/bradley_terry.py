import torch
import torch.nn as nn
import torch.optim as optim
from utils import device

class BradleyTerryModel:
    """
    Bradley-Terry model for pairwise comparisons.
    Learns a quality function q(x) such that P(x_i beats x_j) = sigmoid(q(x_i) - q(x_j))
    """
    def __init__(self, input_dim=64, hidden_dim=128, learning_rate=0.01):
        self.input_dim = input_dim
        self.device = device
        
        # Neural network to learn quality scores
        self.quality_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Single quality score output
        ).to(device)
        
        self.optimizer = optim.Adam(self.quality_model.parameters(), lr=learning_rate)
        self.comparisons_seen = 0
        
    def quality_score(self, x):
        """Compute quality score for embedding x"""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.quality_model(x)
    
    def comparison_probability(self, x_a, x_b):
        """
        P(A beats B) = sigmoid(quality(A) - quality(B))
        """
        quality_a = self.quality_score(x_a)
        quality_b = self.quality_score(x_b)
        return torch.sigmoid(quality_a - quality_b)
    
    def update(self, x_a, x_b, a_wins):
        """
        Update model based on comparison outcome
        
        Args:
            x_a: embedding of first image
            x_b: embedding of second image  
            a_wins: True if A was preferred over B, False otherwise
        """
        self.comparisons_seen += 1
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Compute probability that A beats B
        prob_a_wins = self.comparison_probability(x_a, x_b)
        
        # Binary cross-entropy loss - ensure tensor shapes match
        target = torch.tensor([[1.0 if a_wins else 0.0]], device=self.device)
        loss = nn.BCELoss()(prob_a_wins, target)
        
        # Backpropagate and update
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict_preferences(self, candidates):
        """
        Predict quality scores for a batch of candidates
        
        Args:
            candidates: tensor of shape (n, input_dim)
            
        Returns:
            quality_scores: tensor of shape (n,)
        """
        with torch.no_grad():
            if candidates.ndim == 1:
                candidates = candidates.unsqueeze(0)
            scores = self.quality_model(candidates).squeeze(-1)
            return scores
    
    def select_most_informative_pair(self, candidates, num_pairs_to_consider=100):
        """
        Select the pair of candidates that would be most informative to compare
        based on uncertainty (closest to 50% win probability)
        
        Args:
            candidates: tensor of shape (n, input_dim)
            num_pairs_to_consider: limit pairs to avoid O(n^2) computation
            
        Returns:
            (idx_a, idx_b): indices of the most informative pair
        """
        n_candidates = candidates.shape[0]
        
        if n_candidates < 2:
            return None, None
            
        # For efficiency, only consider a subset of pairs if there are many candidates
        if n_candidates > 50:  # If more than 50 candidates, sample pairs
            indices = torch.randperm(n_candidates)[:50]
            sub_candidates = candidates[indices]
        else:
            indices = torch.arange(n_candidates)
            sub_candidates = candidates
            
        best_entropy = -1
        best_pair = (None, None)
        
        n_sub = sub_candidates.shape[0]
        pairs_checked = 0
        
        for i in range(n_sub):
            for j in range(i + 1, n_sub):
                if pairs_checked >= num_pairs_to_consider:
                    break
                    
                prob = self.comparison_probability(sub_candidates[i], sub_candidates[j])
                prob_clamped = torch.clamp(prob, 1e-8, 1 - 1e-8)  # Numerical stability
                
                # Entropy of Bernoulli distribution
                entropy = -(prob_clamped * torch.log(prob_clamped) + 
                           (1 - prob_clamped) * torch.log(1 - prob_clamped))
                
                if entropy > best_entropy:
                    best_entropy = entropy
                    best_pair = (indices[i].item(), indices[j].item())
                    
                pairs_checked += 1
                
            if pairs_checked >= num_pairs_to_consider:
                break
        
        return best_pair
    
    def get_top_recommendations(self, candidates, k=10):
        """
        Get top k recommendations based on quality scores
        
        Args:
            candidates: tensor of shape (n, input_dim)
            k: number of top recommendations to return
            
        Returns:
            top_indices: indices of top k candidates
            top_scores: quality scores of top k candidates
        """
        scores = self.predict_preferences(candidates)
        top_k_values, top_k_indices = torch.topk(scores, min(k, len(scores)))
        return top_k_indices.cpu().numpy(), top_k_values.cpu().numpy()
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.quality_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'comparisons_seen': self.comparisons_seen,
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.quality_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.comparisons_seen = checkpoint['comparisons_seen']