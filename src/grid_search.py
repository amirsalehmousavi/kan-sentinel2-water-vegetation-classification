from sklearn.base import BaseEstimator, ClassifierMixin
from KANLayer import KAN
import torch
import torch.optim as optim


class PyTorchKANWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, scale_noise=0.1,
                 scale_base=1.0, scale_spline=1.0, base_activation=torch.nn.SiLU,
                 grid_eps=0.02, batch_size=64, epochs=10, learning_rate=0.001):
        self.layers_hidden = layers_hidden
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation
        self.grid_eps = grid_eps
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X, y):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = KAN(
            layers_hidden=self.layers_hidden,
            grid_size=self.grid_size,
            spline_order=self.spline_order,
            scale_noise=self.scale_noise,
            scale_base=self.scale_base,
            scale_spline=self.scale_spline,
            base_activation=self.base_activation,
            grid_eps=self.grid_eps
        ).to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        for epoch in range(self.epochs):
            self.model.train()
            for start in range(0, len(X_tensor), self.batch_size):
                end = min(start + self.batch_size, len(X_tensor))
                X_batch = X_tensor[start:end]
                y_batch = y_tensor[start:end]

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()
