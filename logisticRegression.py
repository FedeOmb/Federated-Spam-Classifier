import numpy as np

class LogisticRegression:
    """Regressione Logistica con regolarizzazione L2"""
    
    def __init__(self, learning_rate: float = 0.01, lambda_reg: float = 0.1):
        self.lr = learning_rate
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def initialize_params(self, n_features: int):
        self.weights = np.zeros(n_features)
        self.bias = 0.0
    
    def compute_loss(self, X, y):
        """Compute cross-entropy loss con regolarizzazione L2"""
        m = len(y)
        z = X.dot(self.weights) + self.bias
        predictions = self.sigmoid(z)
        
        # Cross-entropy loss
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        ce_loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        
        # Regolarizzazione L2
        reg_loss = (self.lambda_reg / (2 * m)) * np.sum(self.weights ** 2)
        
        return ce_loss + reg_loss
    
    def compute_gradients(self, X, y):
        """Calcola i gradienti"""
        m = len(y)
        z = X.dot(self.weights) + self.bias
        predictions = self.sigmoid(z)
        
        dw = (1 / m) * X.T.dot(predictions - y) + (self.lambda_reg / m) * self.weights
        db = (1 / m) * np.sum(predictions - y)
        
        return dw, db
    
    def local_train(self, X, y, epochs: int = 10):
        """Training locale per un client"""
        if self.weights is None:
            self.initialize_params(X.shape[1])
        
        for epoch in range(epochs):
            dw, db = self.compute_gradients(X, y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        z = X.dot(self.weights) + self.bias
        return (self.sigmoid(z) >= 0.5).astype(int)
    
    def get_params(self):
        return {'weights': self.weights.copy(), 'bias': self.bias}
    
    def set_params(self, params):
        self.weights = params['weights'].copy()
        self.bias = params['bias']