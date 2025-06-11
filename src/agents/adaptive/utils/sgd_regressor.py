
import numpy as np

from src.agents.adaptive.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("SGD Regressor")
printer = PrettyPrinter

class SGDRegressor:
    """Online linear regression with SGD and adaptive learning rates"""
    def __init__(self):
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
        self.config = load_global_config()
        self.sgd_config = get_config_section('sgd_regressor')
        self.eta0 = self.sgd_config.get('eta0')
        self.learning_rate = self.sgd_config.get('learning_rate')
        self.alpha = self.sgd_config.get('alpha')
        self.power_t = self.sgd_config.get('power_t')
        self.max_iter = self.sgd_config.get('max_iter')
        self.tol = self.sgd_config.get('tol')

        self.coef_ = None
        self.intercept_ = 0.0
        self.n_samples_seen = 0
        self.t_ = 0  # Time step counter
        self.loss_history = []

        self.random_state = None
        if self.random_state is not None:
            np.random.seed(self.random_state)

        logger.info(f"SGD Regressor succesfully initialized...")

    def partial_fit(self, X, y, sample_weight=None):
        """Incremental fit on batch of samples"""
        printer.status("INIT", "Partial fit succesfully initialized", "info")

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
        printer.status("INIT", "SGD step succesfully initialized", "info")

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
        printer.status("INIT", "Learning rate updater succesfully initialized", "info")

        self.t_ += 1
        if self.learning_rate == 'adaptive' and self.t_ % 100 == 0:
            # Reset learning rate periodically
            self.eta0 *= 0.9

    def predict(self, X):
        """Make predictions"""
        printer.status("INIT", "Predicor succesfully initialized", "info")

        if self.coef_ is None:
            raise RuntimeError("Model not fitted yet")
        return np.dot(X, self.coef_) + self.intercept_

    def score(self, X, y):
        """Compute R^2 score"""
        printer.status("INIT", "Scorer succesfully initialized", "info")

        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - (u / v)

    def get_feature_importance(self):
        """Return normalized feature importance"""
        printer.status("INIT", "Normalizer initialized", "info")

        if self.coef_ is None:
            return None
        importance = np.abs(self.coef_)
        return importance / importance.sum()

    def reset(self):
        """Reset model for new training"""
        printer.status("INIT", "Reseter initialized", "info")

        self.coef_ = None
        self.n_samples_seen = 0
        self.t_ = 0
        self.loss_history = []

if __name__ == "__main__":
    print("\n=== Running SGD Regressor ===\n")
    printer.status("TEST", "Starting SGD Regressor tests", "info")

    regressor = SGDRegressor()
    print(regressor)

    print("\n* * * * * Phase 2 * * * * *\n")
    X=np.array([[2]]) 
    y=np.array([3])
    sample_weight=None

    printer.status("FIT", regressor.partial_fit(X=X, y=y, sample_weight=sample_weight), "success")
    printer.status("predict", regressor.predict(X=X), "success")
    printer.status("score", regressor.score(X=X, y=y), "success")
    printer.status("feature", regressor.get_feature_importance(), "success")
    printer.status("reset", regressor.reset(), "success")

    print("\n=== Successfully Ran SGD Regressor ===\n")
