import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split


class PytorchTrainer:

    def __init__(self, model: nn.Module, loss=nn.CrossEntropyLoss, optimizer=torch.optim.Adam):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer(self.model.parameters())
        self.test_loader = None
        self.valid_loader = None
        self.train_loader = None

    def set_data(self, data, batch_size, validation_rate=0.2):
        # train_data, test_data = train_test_split(data, test_size=test_rate)
        train_data, valid_data = train_test_split(data, test_size=validation_rate)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
        # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        # self.test_data=test_data

    def train(self, epochs=20, verbose=True):

        losses = []
        accuracy = []

        for epoch in range(epochs):
            for data, labels in self.train_loader:
                # images, labels = images.to(device), labels.to(device)
                outputs = self.model(data)
                loss = self.loss(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                N = 0
                tot_loss, correct = 0.0, 0.0

                for data, labels in self.valid_loader:
                    # images, labels = images.to(device), labels.to(device)
                    outputs = self.model(data)
                    N += data.shape[0]

                    tot_loss += data.shape[0] * self.loss(outputs, labels).item()

                    predicted_targets = outputs.argmax(dim=1)
                    correct += (predicted_targets == labels).sum().item()
                losses.append(tot_loss / N)
                accuracy.append(correct / N)
                if verbose:
                    print('Epoch ', epoch + 1, '==> Validation loss: ', tot_loss / N, ', Accuracy: ', correct / N)

        return losses, accuracy
