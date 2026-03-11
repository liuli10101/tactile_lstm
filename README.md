tactile_lstm/
├── data/                # 数据目录（原始数据、预处理后数据）
│   ├── raw/             # 原始数据集（如csv、txt、excel）
│   └── processed/       # 预处理后的数据（归一化、划分训练/测试集）
├── models/              # 模型目录
│   ├── lstm_model.py    # LSTM模型定义（核心）
│   └── model_weights/   # 训练好的模型权重文件（.h5/.pth）
├── scripts/             # 脚本目录
│   ├── data_preprocess.py  # 数据预处理（归一化、序列构造、划分数据集）
│   ├── train.py         # 模型训练脚本
│   └── predict.py       # 预测脚本（加载模型，输入数据输出预测结果）
├── utils/               # 工具函数
│   ├── metrics.py       # 评估指标（MAE、MSE、RMSE、MAPE等）
│   └── plot.py          # 结果可视化（真实值vs预测值对比图）
├── requirements.txt     # 依赖包清单（tensorflow/pytorch、pandas、numpy、matplotlib等）
└── README.md            # 项目说明（数据来源、运行步骤、参数说明）

输入 (B, T, 312)
→ Linear(312 → 128)   # 每个时间步独立做空间特征压缩
→ ReLU
→ LSTM(128 → 64)
→ Dropout
→ Linear(64 → 32)
→ ReLU
→ Linear(32 → 1)
→ Sigmoid
