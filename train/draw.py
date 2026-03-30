import torch
import torch.nn as nn
from torchviz import make_dot

# ==========================================
# 1. 把你原来的模型定义原封不动复制过来
#    (或者你也可以从原来的 train_lstm2.py 里 import)
# ==========================================
class SlipDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入维度 624
        self.feature_proj = nn.Linear(624, 128)
        self.relu = nn.ReLU()
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        # Attention
        self.attn = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x (B,T,624)
        x = self.feature_proj(x)
        x = self.relu(x)
        lstm_out, _ = self.lstm(x)
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(lstm_out * weights, dim=1)
        out = self.dropout(context)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out.squeeze()

# ==========================================
# 2. 核心：生成结构图
# ==========================================
if __name__ == "__main__":
    print("正在初始化模型...")
    model = SlipDetectionModel()

    # --- 创建虚拟输入 (Dummy Input) ---

    dummy_x = torch.randn(1, 20, 624)

    # --- 前向传播一次 ---
    print("正在进行前向传播...")
    dummy_y = model(dummy_x)

    # --- 生成计算图 ---
    print("正在生成结构图...")
    graph = make_dot(
        dummy_y,                    # 模型的输出
        params=dict(model.named_parameters()), # 模型的参数
        # params=dict(),
        show_attrs=False,           # 关闭：不显示节点属性，让图更干净
        show_saved=False            # 关闭：不显示保存的中间变量
    )

    # --- 保存文件 ---
    # 1. 保存为 PNG (方便直接查看)
    graph.format = 'png'
    graph.render('torchviz_result_png') 
    print("✅ 已保存图片: torchviz_result_png.png")

    # 2. 保存为 SVG (矢量图，适合论文/二次编辑)
    graph.format = 'svg'
    graph.render('torchviz_result_svg')
    print("✅ 已保存矢量图: torchviz_result_svg.svg")

    print("\n🎉 完成！你可以在当前文件夹下找到生成的图片。")
