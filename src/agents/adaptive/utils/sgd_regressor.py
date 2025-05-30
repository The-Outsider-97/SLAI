
import numpy as np

class SGDRegressor:
    """Online linear regression with SGD and adaptive learning rates"""
    def __init__(self, eta0=0.01, learning_rate='constant', alpha=0.0001, 
                 power_t=0.25, max_iter=1000, tol=1e-4, random_state=None):
        """
        Args:
            eta0: Initial learning rate
            learning_rate: {'constant', 'invscaling', 'adaptive'}
            alpha: L2 regularization strength (ridge penalty)
            power_t: Exponent for inverse scaling learning rate
            max_iter: Max iterations per partial_fit call
            tol: Stopping tolerance for weight updates
            random_state: Random seed for weight initialization
        """
        self.eta0 = eta0
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.power_t = power_t
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = 0.0
        self.n_samples_seen = 0
        self.t_ = 0  # Time step counter
        self.loss_history = []
        
        if random_state is not None:
            np.random.seed(random_state)

    def partial_fit(self, X, y, sample_weight=None):
        """Incremental fit on batch of samples"""
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Initialize weights on first call
        if self.coef_ is None:
            self.coef_ = np.random.normal(0, 0.01, X.shape[1])
        
        # Handle single sample
        if len(y.shape) == 0:
            X = X.reshape(1, -1)
            y = y.reshape(1)
        
        # Track sample count
        self.n_samples_seen += X.shape[0]
        
        # Optimize using SGD
        prev_coef = self.coef_.copy()
        for _ in range(self.max_iter):
            converged = self._sgd_step(X, y, sample_weight)
            if converged:
                break
        
        # Update learning schedule
        self._update_learning_rate()
        return self

    def _sgd_step(self, X, y, sample_weight):
        """Single SGD epoch over all samples"""
        converged = True
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        
        for i in indices:
            x_i = X[i]
            y_i = y[i]
            weight = sample_weight[i] if sample_weight else 1.0
            
            # Compute prediction and error
            pred = np.dot(x_i, self.coef_) + self.intercept_
            error = pred - y_i
            
            # Compute gradients (with regularization)
            grad_coef = (error * x_i) + (self.alpha * self.coef_)
            grad_intercept = error
            
            # Update weights
            self.coef_ -= self.eta * weight * grad_coef
            self.intercept_ -= self.eta * weight * grad_intercept
            
            # Track loss
            self.loss_history.append(0.5 * error**2 + 0.5*self.alpha*np.sum(self.coef_**2))
            
            # Check convergence
            if np.linalg.norm(grad_coef) > self.tol:
                converged = False
        
        return converged

    @property
    def eta(self):
        """Current learning rate based on schedule"""
        if self.learning_rate == 'constant':
            return self.eta0
        elif self.learning_rate == 'invscaling':
            return self.eta0 / np.power(self.t_ + 1, self.power_t)
        elif self.learning_rate == 'adaptive':
            # Reduce learning rate when loss plateaus
            if len(self.loss_history) > 10:
                last_losses = np.array(self.loss_history[-10:])
                if np.std(last_losses) < 0.001 * np.mean(last_losses):
                    return self.eta0 * 0.5
            return self.eta0
        else:
            raise ValueError(f"Unsupported learning rate: {self.learning_rate}")

    def _update_learning_rate(self):
        """Update time counter for learning rate schedules"""
        self.t_ += 1
        if self.learning_rate == 'adaptive' and self.t_ % 100 == 0:
            # Reset learning rate periodically
            self.eta0 *= 0.9

    def predict(self, X):
        """Make predictions"""
        if self.coef_ is None:
            raise RuntimeError("Model not fitted yet")
        return np.dot(X, self.coef_) + self.intercept_

    def score(self, X, y):
        """Compute R^2 score"""
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - (u / v)

    def get_feature_importance(self):
        """Return normalized feature importance"""
        if self.coef_ is None:
            return None
        importance = np.abs(self.coef_)
        return importance / importance.sum()

    def reset(self):
        """Reset model for new training"""
        self.coef_ = None
        self.n_samples_seen = 0
        self.t_ = 0
        self.loss_history = []
