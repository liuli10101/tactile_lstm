import torch
import torch.nn as nn
import numpy as np

# ===== 1️⃣ 构造假数据 =====
# 1000组样本
X = torch.randn(1000, 20, 312)
y = torch.randint(0, 2, (1000, 1)).float()

# ===== 2️⃣ 定义模型 =====
class SlipDetectionLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_reduce = nn.Linear(312, 128)

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feature_reduce(x)
        x = self.relu(x)

        out, _ = self.lstm(x)

        x = out[:, -1, :]

        x = self.dropout(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        return x


# ===== 3️⃣ 初始化 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SlipDetectionLSTM().to(device)
X = X.to(device)
y = y.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ===== 4️⃣ 训练 =====
epochs = 5

for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")