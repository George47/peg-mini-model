import torch
import torch.nn as nn
import torch.optim as optim

class RecommendationModel(nn.Module):
    def __init__(self):
        super(RecommendationModel, self).__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function (in train_model.py or train_model.ipynb)
def train_model():
    # Sample data
    data = torch.tensor([
        [0, 9, 1, 4], [1, 11, 1, 5], [2, 15, 1, 3],
        [0, 10, 7, 4], [1, 12, 7, 5], [2, 14, 7, 4],
        [0, 8, 30, 3], [1, 13, 30, 5], [2, 16, 30, 4],
        [0, 9, 365, 4], [1, 10, 365, 5], [2, 11, 365, 4]
    ], dtype=torch.float32)

    X = data[:, :3]
    y = data[:, 3].view(-1, 1)

    model = RecommendationModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/200], Loss: {loss.item():.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'models/recommendation_model.pth')

if __name__ == "__main__":
    train_model()
