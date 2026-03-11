import torch
import torch.nn as nn

class SlipDetectionLSTM(nn.Module):
    def __init__(self):
        super(SlipDetectionLSTM, self).__init__()

        #  线性压缩层，将每个时间步的 312 维输入压缩为 128 维
        self.feature_reduce = nn.Linear(312, 128)

        #  LSTM 层，处理 128 维输入的时序信息，输出 64 维隐藏状态
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        #  Dropout 层，防止过拟合
        self.dropout = nn.Dropout(0.3)

        #  全连接层，64 → 32 维
        self.fc1 = nn.Linear(64, 32)

        #  输出层，将 32 维映射到 1 维，输出滑动判别概率
        self.fc2 = nn.Linear(32, 1)

        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x shape: (B, T, 312)  
        B: batch size, T: time steps (e.g. 20), 312: input dimension (52 * 3)
        """

        #  空间特征压缩层
        x = self.feature_reduce(x)   # (B, T, 128)
        x = self.relu(x)

        #  LSTM 层
        lstm_out, _ = self.lstm(x)   # (B, T, 64)

        #  只取最后一个时间步的输出
        x = lstm_out[:, -1, :]       # (B, 64)

        #  Dropout 层
        x = self.dropout(x)

        #  全连接层
        x = self.fc1(x)              # (B, 32)
        x = self.relu(x)

        #  输出层
        x = self.fc2(x)              # (B, 1)
        x = self.sigmoid(x)          # (B, 1), 概率输出

        return x