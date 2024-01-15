import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from pprint import pprint
import random


def create_sliding_window_data(data, input_window, output_window, output_column):
    X, y = [], []
    for i in range(len(data) - input_window - output_window + 1):
        X.append(data[i:i + input_window])
        y.append(data[i + input_window:i + input_window + output_window, output_column])
    return np.array(X), np.array(y)


# 读取 CSV 文件
data = pd.read_csv('ETTh1.csv')
# data = pd.read_csv('test.csv')

# 数据预处理
# 特征工程
# data['date'] = pd.to_datetime(data['date'])
# data['year'] = data['date'].dt.year
# data['month'] = data['date'].dt.month
# data['day'] = data['date'].dt.day
# data['hour'] = data['date'].dt.hour
data.drop('date', axis=1, inplace=True)

# 参数设置
input_window = 96
output_window = 96
output_column = [i for i in range(7)]
# output_column = 6
valid_ratio = 0.2
test_ratio = 0.2

# 划分训练集和测试集
data_len = len(data)
valid_size = int(data_len * valid_ratio)
test_size = int(data_len * test_ratio)

# X_train, X_valid, X_test = data[:-(test_size + valid_size)], data[-(test_size + valid_size):-test_size], data[-test_size:]

scaler = StandardScaler()
data = scaler.fit_transform(data)

X_data_sliding, y_data_sliding = create_sliding_window_data(data, input_window, output_window, output_column)
data_sliding = list(zip(X_data_sliding, y_data_sliding))
random.shuffle(data_sliding)
X_data_sliding[:], y_data_sliding[:] = zip(*data_sliding)
X_train_sliding, y_train_sliding = X_data_sliding[:-(test_size + valid_size)], y_data_sliding[:-(test_size + valid_size)]
X_valid_sliding, y_valid_sliding = X_data_sliding[-(test_size + valid_size):-test_size], y_data_sliding[-(test_size + valid_size):-test_size]
X_test_sliding, y_test_sliding = X_data_sliding[-test_size:], y_data_sliding[-test_size:]

# # 数据归一化
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_valid = scaler.transform(X_valid)
#
# test_scaler = StandardScaler()
# X_test = test_scaler.fit_transform(X_test)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_sliding, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_sliding, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid_sliding, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid_sliding, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_sliding, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_sliding, dtype=torch.float32)
# print(y_test_tensor)
print(y_valid_tensor.size())

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)


# 定义模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional):
        super(LSTMModel, self).__init__()
        self.bidirectional_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                                          bidirectional=bidirectional)
        self.unidirectional_lstm = nn.LSTM(hidden_size * (2 if bidirectional else 1), hidden_size, num_layers,
                                           batch_first=True, bidirectional=False)
        # self.bidirectional_lstm_2 = nn.LSTM(hidden_size * 2, hidden_size, num_layers, batch_first=True,
        #                                   bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Pass the input through the bidirectional LSTM
        out_bidirectional, _ = self.bidirectional_lstm(x)
        # print(out_bidirectional)

        # Pass the output of the bidirectional LSTM through the unidirectional LSTM
        out_unidirectional, _ = self.unidirectional_lstm(out_bidirectional)
        # out_bidirectional_2, _ = self.bidirectional_lstm_2(out_bidirectional)


        # Apply ReLU activation on the output of the unidirectional LSTM
        # out = F.relu(out_unidirectional[:, -1, :])
        out = F.tanh(out_unidirectional[:, -1, :])
        # out = F.tanh(out_bidirectional_2[:, -1, :])

        # Pass the activated output through the fully connected layer
        out = self.fc(out)
        # print(out.size())
        return out


# 初始化模型、优化器和损失函数
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
input_size = data.shape[1]
print("input_size:", input_size)
hidden_size = 512
num_layers = 2
# output_size = output_window
output_size = 96 * 7
bidirectional = True

model = LSTMModel(input_size, hidden_size, num_layers, output_size, bidirectional).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
criterion = nn.MSELoss()
losses = []
# 训练模型
num_epochs = 300
patience = 10
p = 0
min_loss = float('inf')

# # Train
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for inputs, targets in train_loader:
#         # print(inputs)
#         inputs, targets = inputs.to(device), targets.to(device)  # [bz * 96 * features_num]
#         # print(targets.size())
#         # print(targets.view(32, -1).size())
#         optimizer.zero_grad()
#         # outputs = model(inputs).squeeze()
#         outputs = model(inputs)  # [bz * 672]
#         # print(outputs)
#         # print(targets)
#         # print("outputs:", outputs.size())
#         # loss = criterion(outputs, targets.squeeze(-1))
#
#         # print("inputs:")
#         # print(inputs.reshape(inputs.size(0), -1))
#         # print("outputs:")
#         # print(outputs)
#         # print("targets:")
#         # print(targets.reshape(targets.size(0), -1))
#
#         loss = criterion(outputs, targets.reshape(targets.size(0), -1))
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         losses.append(loss.item())
#     print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}', end=' ')
#     # dev
#     model.eval()
#     with torch.no_grad():
#         X_valid_tensor = X_valid_tensor.to(device)
#         y_valid_tensor = y_valid_tensor.to(device)
#         predictions = model(X_valid_tensor)
#         # predictions = predictions.reshape(predictions.shape[0], 96, -1)
#         dev_loss = criterion(predictions, y_valid_tensor.reshape(y_valid_tensor.size(0), -1))
#         print(f"Dev Loss: {dev_loss}")
#
#     if dev_loss > min_loss:
#         p += 1
#         if p == patience:
#             print(f"Epoch {epoch - 10} is best")
#             print(f"Best dev loss = {min_loss}")
#             print("Finish Training")
#             break
#     else:
#         print("Saving Model")
#         torch.save(model.state_dict(), f"models/LSTM_dev_best.v2.pth")
#         min_loss = dev_loss
#         p = 0
#     print("Saving Temp Model")
#     torch.save(model.state_dict(), f"models/LSTM_dev_temp.v2.pth")


# results = []
# plt.figure()
# plt.plot(losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.savefig('lossPhoto')

# 滑窗预测
model = LSTMModel(input_size, hidden_size, num_layers, output_size, bidirectional).to(device)
print("Loading Model")
model.load_state_dict(torch.load("models/LSTM_dev_best.v2.pth"))
model.eval()
with torch.no_grad():
    X_test_tensor = X_test_tensor.to(device)
    predictions = model(X_test_tensor).cpu()
    predictions = predictions.reshape(predictions.shape[0], 96, -1)

# print(results)
prediction = predictions[0]
prediction = scaler.inverse_transform(prediction)
print(prediction)
label = scaler.inverse_transform(y_test_tensor[0])
print(label)
# predictions = feature_scaler.inverse_transform(predictions.reshape(1, -1)).tolist()[0]
# results.append(predictions)
# print(predictions)
# print(len(predictions))