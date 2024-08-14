import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import ast
from tslearn.clustering import KShape

# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# 数据准备
# 假设您的数据为 data，形状为 (n, m)
data = pd.read_csv("../data/cluster_data/用户过去n天的四级公司出勤模式序列-stage3.csv")
data["sequence_value"] = data["sequence_value"].apply(ast.literal_eval)
input_x = np.array(data["sequence_value"].tolist())
input_x = torch.Tensor(input_x)

# 定义超参数
input_dim = 60  # 输入数据的特征维度
latent_dim = 10  # 潜在特征的维度
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# 创建自编码器模型
model = Autoencoder(input_dim, latent_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for i in range(0, len(input_x), batch_size):
        # 准备数据批次
        inputs = input_x[i:i + batch_size]

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), 'autoencoder_model.pth')


# 使用自编码器对数据进行编码
encoded_data = model(torch.Tensor(input_x)).detach().numpy()
print(encoded_data)

# 创建并拟合 k-形状聚类模型
model = KShape(n_clusters=3, n_init=3, verbose=True, random_state=42)
y_pred = model.fit_predict(encoded_data)

# 将聚类结果添加到 DataFrame
data['cluster'] = y_pred

data.to_csv("../data/output_cluster.csv")
