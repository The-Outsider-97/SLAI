
import json
import math
import random
import numpy as np

from collections import defaultdict
from typing import List, Tuple, Optional

class BayesianNeuralNetwork:
    """
    Bayesian Neural Network with Variational Inference
    Implements a probabilistic neural network with uncertainty estimation
    
    Features:
    - Gaussian prior and posterior distributions for weights
    - Reparameterization trick for gradient estimation
    - Mini-batch training with ELBO optimization
    - Uncertainty quantification in predictions
    - Automatic differentiation for variational parameters
    
    Architecture:
    Input -> [Hidden Layers] -> Output
    """
    
    def __init__(self, layer_sizes: list, learning_rate: float = 0.01):
        """
        Initialize Bayesian Neural Network
        
        Args:
            layer_sizes: List of layer dimensions [input, hidden1, ..., output]
            learning_rate: Step size for gradient updates
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes) - 1
        
        # Initialize variational parameters (mean and log variance)
        self.weights_mu = []
        self.weights_logvar = []
        self.biases_mu = []
        self.biases_logvar = []
        
        # Initialize prior parameters
        self.prior_mu = 0.0
        self.prior_logvar = math.log(1.0)  # Prior variance = 1
        
        # Initialize network parameters
        for i in range(self.num_layers):
            # He initialization scaled by 0.1 for Bayesian networks
            fan_in = layer_sizes[i]
            scale = math.sqrt(2.0 / fan_in) * 0.1
            
            # Weight parameters
            self.weights_mu.append(
                np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            )
            self.weights_logvar.append(
                np.full((layer_sizes[i], layer_sizes[i+1]), 
                math.log(0.1)))  # Initial variance
            
            # Bias parameters
            self.biases_mu.append(
                np.zeros(layer_sizes[i+1])
            )
            self.biases_logvar.append(
                np.full(layer_sizes[i+1], math.log(0.1)))
    
    def sample_parameters(self):
        """Sample weights and biases from variational distribution"""
        weights, biases = [], []
        
        for i in range(self.num_layers):
            # Reparameterization trick: W = μ + ε * σ
            epsilon_w = np.random.randn(*self.weights_mu[i].shape)
            std_w = np.exp(0.5 * self.weights_logvar[i])
            weights.append(self.weights_mu[i] + epsilon_w * std_w)
            
            # Sample biases
            epsilon_b = np.random.randn(*self.biases_mu[i].shape)
            std_b = np.exp(0.5 * self.biases_logvar[i])
            biases.append(self.biases_mu[i] + epsilon_b * std_b)
            
        return weights, biases
    
    def forward(self, x: np.ndarray, weights: List[np.ndarray], 
                biases: List[np.ndarray]) -> np.ndarray:
        """
        Forward pass through the network
        
        Args:
            x: Input data (batch_size x input_dim)
            weights: Sampled weight matrices
            biases: Sampled bias vectors
            
        Returns:
            Network output (batch_size x output_dim)
        """
        a = x
        for i in range(self.num_layers - 1):
            # Linear transformation
            z = np.dot(a, weights[i]) + biases[i]
            # ReLU activation
            a = np.maximum(0, z)
        
        # Final layer (linear activation)
        output = np.dot(a, weights[-1]) + biases[-1]
        return output
    
    def elbo(self, x: np.ndarray, y: np.ndarray, num_samples: int = 1) -> float:
        """
        Compute Evidence Lower Bound (ELBO)
        
        Args:
            x: Input data
            y: Target labels
            num_samples: Number of MC samples
            
        Returns:
            ELBO value and KL divergence
        """
        log_likelihood = 0
        kl_divergence = 0
        batch_size = x.shape[0]
        
        # KL divergence term (independent of data)
        for i in range(self.num_layers):
            # KL divergence for weights
            kl_w = self._kl_divergence(
                self.weights_mu[i], self.weights_logvar[i],
                self.prior_mu, self.prior_logvar
            )
            
            # KL divergence for biases
            kl_b = self._kl_divergence(
                self.biases_mu[i], self.biases_logvar[i],
                self.prior_mu, self.prior_logvar
            )
            
            kl_divergence += kl_w + kl_b
        
        # Monte Carlo estimation of log likelihood
        for _ in range(num_samples):
            weights, biases = self.sample_parameters()
            outputs = self.forward(x, weights, biases)
            
            # Gaussian log-likelihood (σ=1)
            # L2 distance: ||y - f(x)||^2
            log_likelihood -= 0.5 * np.sum((y - outputs) ** 2)
        
        # Average over samples and scale by batch size
        log_likelihood /= num_samples
        elbo = log_likelihood - kl_divergence / batch_size
        return elbo, kl_divergence
    
    def _kl_divergence(self, mu: np.ndarray, logvar: np.ndarray, 
                      prior_mu: float, prior_logvar: float) -> float:
        """Compute KL divergence between two Gaussians"""
        var = np.exp(logvar)
        prior_var = np.exp(prior_logvar)
        
        kl = 0.5 * np.sum(
            (var + (mu - prior_mu) ** 2) / prior_var
            - logvar
            + prior_logvar
            - 1
        )
        return kl
    
    def train_step(self, x_batch: np.ndarray, y_batch: np.ndarray,
                  num_samples: int = 1) -> Tuple[float, float]:
        """
        Perform one training step
        
        Args:
            x_batch: Batch of input data
            y_batch: Batch of target labels
            num_samples: Number of MC samples
            
        Returns:
            ELBO value and KL divergence
        """
        # Compute gradients using automatic differentiation
        grads = self._compute_gradients(x_batch, y_batch, num_samples)
        
        # Update parameters with gradients
        for i in range(self.num_layers):
            # Update weights
            self.weights_mu[i] += self.learning_rate * grads['weights_mu'][i]
            self.weights_logvar[i] += self.learning_rate * grads['weights_logvar'][i]
            
            # Update biases
            self.biases_mu[i] += self.learning_rate * grads['biases_mu'][i]
            self.biases_logvar[i] += self.learning_rate * grads['biases_logvar'][i]
        
        return self.elbo(x_batch, y_batch, num_samples)
    
    def _compute_gradients(self, x: np.ndarray, y: np.ndarray,
                          num_samples: int = 1) -> dict:
        """Compute gradients of ELBO wrt variational parameters"""
        # Initialize gradient accumulators
        grads = {
            'weights_mu': [np.zeros_like(w) for w in self.weights_mu],
            'weights_logvar': [np.zeros_like(w) for w in self.weights_logvar],
            'biases_mu': [np.zeros_like(b) for b in self.biases_mu],
            'biases_logvar': [np.zeros_like(b) for b in self.biases_logvar],
        }
        
        batch_size = x.shape[0]
        
        for _ in range(num_samples):
            # Sample parameters and epsilons
            weights, biases = [], []
            epsilons_w, epsilons_b = [], []
            
            for i in range(self.num_layers):
                # Sample noise
                eps_w = np.random.randn(*self.weights_mu[i].shape)
                eps_b = np.random.randn(*self.biases_mu[i].shape)
                
                # Compute parameter samples
                std_w = np.exp(0.5 * self.weights_logvar[i])
                w = self.weights_mu[i] + eps_w * std_w
                
                std_b = np.exp(0.5 * self.biases_logvar[i])
                b = self.biases_mu[i] + eps_b * std_b
                
                weights.append(w)
                biases.append(b)
                epsilons_w.append(eps_w)
                epsilons_b.append(eps_b)
            
            # Forward pass
            activations = [x]
            pre_activations = []
            
            a = x
            for i in range(self.num_layers - 1):
                z = np.dot(a, weights[i]) + biases[i]
                pre_activations.append(z)
                a = np.maximum(0, z)  # ReLU
                activations.append(a)
            
            # Final layer
            z_final = np.dot(a, weights[-1]) + biases[-1]
            pre_activations.append(z_final)
            activations.append(z_final)
            
            # Backward pass (compute gradients of log-likelihood)
            # Start with output layer gradient
            delta = (activations[-1] - y) / num_samples
            
            # Backpropagate through layers
            for i in range(self.num_layers - 1, -1, -1):
                # Gradient wrt weights
                d_w = np.dot(activations[i].T, delta)
                
                # Gradient wrt biases
                d_b = np.sum(delta, axis=0)
                
                # Accumulate gradients for variational parameters
                # For μ: ∇μ L = ∇w L
                grads['weights_mu'][i] += d_w
                
                # For logvar: ∇logvar L = ∇w L * ε * 0.5 * exp(0.5 * logvar)
                grads['weights_logvar'][i] += d_w * epsilons_w[i] * 0.5 * np.exp(0.5 * self.weights_logvar[i])
                
                grads['biases_mu'][i] += d_b
                grads['biases_logvar'][i] += d_b * epsilons_b[i] * 0.5 * np.exp(0.5 * self.biases_logvar[i])
                
                # Propagate gradient to previous layer
                if i > 0:
                    # Gradient wrt previous activation
                    delta = np.dot(delta, weights[i].T)
                    
                    # Gradient through ReLU
                    delta = delta * (pre_activations[i-1] > 0).astype(float)
        
        # Add gradients from KL divergence term
        batch_scale = 1 / batch_size
        for i in range(self.num_layers):
            # KL gradients for weights
            d_kl_w_mu = (self.weights_mu[i] - self.prior_mu) / np.exp(self.prior_logvar)
            d_kl_w_logvar = 0.5 * (np.exp(self.weights_logvar[i]) / np.exp(self.prior_logvar) - 1)
            
            # KL gradients for biases
            d_kl_b_mu = (self.biases_mu[i] - self.prior_mu) / np.exp(self.prior_logvar)
            d_kl_b_logvar = 0.5 * (np.exp(self.biases_logvar[i]) / np.exp(self.prior_logvar) - 1)
            
            # Combine with likelihood gradients
            grads['weights_mu'][i] = grads['weights_mu'][i] - batch_scale * d_kl_w_mu
            grads['weights_logvar'][i] = grads['weights_logvar'][i] - batch_scale * d_kl_w_logvar
            
            grads['biases_mu'][i] = grads['biases_mu'][i] - batch_scale * d_kl_b_mu
            grads['biases_logvar'][i] = grads['biases_logvar'][i] - batch_scale * d_kl_b_logvar
        
        return grads
    
    def predict(self, x: np.ndarray, num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimation
        
        Args:
            x: Input data
            num_samples: Number of forward passes
            
        Returns:
            mean: Mean prediction
            std: Standard deviation of predictions
        """
        predictions = []
        
        for _ in range(num_samples):
            weights, biases = self.sample_parameters()
            pred = self.forward(x, weights, biases)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        return mean, std
    
    def save(self, filename: str):
        """Save model parameters to file"""
        params = {
            'layer_sizes': self.layer_sizes,
            'weights_mu': [w.tolist() for w in self.weights_mu],
            'weights_logvar': [w.tolist() for w in self.weights_logvar],
            'biases_mu': [b.tolist() for b in self.biases_mu],
            'biases_logvar': [b.tolist() for b in self.biases_logvar],
        }
        with open(filename, 'w') as f:
            json.dump(params, f)
    
    @classmethod
    def load(cls, filename: str):
        """Load model from file"""
        with open(filename, 'r') as f:
            params = json.load(f)
        
        model = cls(params['layer_sizes'])
        model.weights_mu = [np.array(w) for w in params['weights_mu']]
        model.weights_logvar = [np.array(w) for w in params['weights_logvar']]
        model.biases_mu = [np.array(b) for b in params['biases_mu']]
        model.biases_logvar = [np.array(b) for b in params['biases_logvar']]
        
        return model
