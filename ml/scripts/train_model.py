import torch
from model import RecommendationModel

def train_model():
    # Load preprocessed data
    data = torch.tensor([
        [0, 9, 1, 4], [1, 11, 1, 5], [2, 15, 1, 3],
        [0, 10, 7, 4], [1, 12, 7, 5], [2, 14, 7, 4],
        [0, 8, 30, 3], [1, 13, 30, 5], [2, 16, 30, 4],
        [0, 9, 365, 4], [1, 10, 365, 5], [2, 11, 365, 4]
    ], dtype=torch.float32)

    X = data[:, :3]
    y = data[:, 3].view(-1, 1)

    model = RecommendationModel()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/200], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'models/recommendation_model.pth')

if __name__ == "__main__":
    train_model()
