
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from logs.logger import get_logger

logger = get_logger("PyTorch Safety NN")

class SecurityNN(nn.Module):
    """
    PyTorch-based neural network for cyber-security tasks
    Replaces the custom NeuralNetwork implementation with PyTorch's robust framework
    """
    def __init__(self, input_size, layer_config, problem_type='binary_classification'):
        super(SecurityNN, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = []
        self.dropouts = []
        self.batch_norms = []
        self.problem_type = problem_type
        
        # Build network layers
        current_size = input_size
        for i, config in enumerate(layer_config):
            # Add linear layer
            layer = nn.Linear(current_size, config['neurons'])
            self.layers.append(layer)
            
            # Add activation
            if config['activation'] == 'relu':
                self.activations.append(nn.ReLU())
            elif config['activation'] == 'sigmoid':
                self.activations.append(nn.Sigmoid())
            elif config['activation'] == 'tanh':
                self.activations.append(nn.Tanh())
            elif config['activation'] == 'leaky_relu':
                self.activations.append(nn.LeakyReLU(0.01))
            elif config['activation'] == 'elu':
                self.activations.append(nn.ELU(alpha=1.0))
            elif config['activation'] == 'swish':
                self.activations.append(nn.SiLU())  # SiLU is Swish
            else:  # linear
                self.activations.append(nn.Identity())
            
            # Add dropout
            dropout_rate = config.get('dropout', 0.0)
            if dropout_rate > 0 and i < len(layer_config) - 1:  # No dropout on output layer
                self.dropouts.append(nn.Dropout(dropout_rate))
            else:
                self.dropouts.append(nn.Identity())
                
            # Add batch normalization
            use_bn = config.get('batch_norm', False)
            if use_bn and i < len(layer_config) - 1:  # No BN on output layer
                self.batch_norms.append(nn.BatchNorm1d(config['neurons']))
            else:
                self.batch_norms.append(nn.Identity())
                
            current_size = config['neurons']
        
        # Output activation
        if problem_type == 'binary_classification':
            self.output_activation = nn.Sigmoid()
        elif problem_type == 'multiclass_classification':
            self.output_activation = nn.Softmax(dim=1)
        else:  # regression
            self.output_activation = nn.Identity()

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.batch_norms[i](x)
            x = self.activations[i](x)
            x = self.dropouts[i](x)
        return self.output_activation(x)

class PyTorchSafetyModel:
    """
    Wrapper class for training and using security neural networks
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Set loss function based on problem type
        if model.problem_type == 'binary_classification':
            self.criterion = nn.BCELoss()
        elif model.problem_type == 'multiclass_classification':
            self.criterion = nn.CrossEntropyLoss()
        else:  # regression
            self.criterion = nn.MSELoss()
            
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, # verbose=True
        )

    def train_model(self, train_data, val_data=None, epochs=50, batch_size=32):
        """
        Train the security model with PyTorch's built-in capabilities
        """
        self.model.train()
        X_train, y_train = train_data
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * inputs.size(0)
            
            epoch_loss /= len(train_loader.dataset)
            
            # Validation
            val_loss = None
            if val_data:
                val_loss = self.evaluate_model(val_data, batch_size)
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict()
            
            logger.info(f'Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss or "N/A":.4f}')
        
        # Load best model weights
        if val_data:
            self.model.load_state_dict(best_model_state)

    def evaluate_model(self, eval_data, batch_size=32):
        """
        Evaluate model performance on validation/test data
        """
        self.model.eval()
        X_eval, y_eval = eval_data
        eval_dataset = TensorDataset(
            torch.tensor(X_eval, dtype=torch.float32),
            torch.tensor(y_eval, dtype=torch.float32)
        )
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
        
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in eval_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
        
        return total_loss / len(eval_loader.dataset)

    def predict(self, X):
        """
        Make predictions with the security model
        """
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(inputs)
            return outputs.cpu().numpy()

    def save_model(self, filepath):
        """
        Save model weights and configuration
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model_config
        }, filepath)
        logger.info(f"Security model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Load a saved security model
        """
        checkpoint = torch.load(filepath, map_location=device)
        model = SecurityNN(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        wrapper = cls(model, device)
        wrapper.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Security model loaded from {filepath}")
        return wrapper
