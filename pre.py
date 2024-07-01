import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

# file_path = 'train.csv'
# data = pd.read_csv(file_path)

# pixels = data.values

# labels = pixels[:, 0].astype(int)
# train_data = pixels[:, 1:]
# train_data = train_data.reshape(train_data.shape[0], 1, 28, 28)

# print(f"train_data.shape: {train_data.shape}")
# print(f"label.shape: {labels.shape}")

# test_data = train_data[3000:]
# train_data = train_data[:3000]
# test_labels = labels[3000:]
# train_labels = labels[:3000]

# tensor_x = torch.Tensor(train_data).float()
# tensor_y = torch.Tensor(train_labels).long()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# tensor_x = tensor_x.to(device)
# tensor_y = tensor_y.to(device)
# test_x = torch.Tensor(test_data).float().to(device)
# test_y = torch.Tensor(test_labels).long().to(device)

# dataset = TensorDataset(tensor_x, tensor_y)
# test_dataset = TensorDataset(test_x, test_y)
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.resnet18 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = nn.Linear(512, 10)
        
    def forward(self, x):
        return self.resnet18(x)


# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# epochs = 100

# writer = SummaryWriter()
# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
#     for i, (inputs, labels) in enumerate(train_loader):
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {running_loss}")
#     writer.add_scalar("Loss", running_loss, epoch)
    
#     model.eval()
#     correct = 0.0
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             correct += (predicted == labels).sum().item()
#     accuracy = correct / len(test_loader.dataset)
#     print(f"Accuracy: {accuracy}")
#     writer.add_scalar("Accuracy", accuracy, epoch)
    
#     torch.save(model.state_dict(), f'model/model{epoch+1}.pth')

file_path = 'test.csv'
data = pd.read_csv(file_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pixels = data.values
train_data = pixels.reshape(pixels.shape[0], 1, 28, 28)
#转化成tensor
train_data = torch.Tensor(train_data).float()

model = resnet18().to(device)
model.eval()
model.load_state_dict(torch.load('model/model30.pth'))

dataloader = DataLoader(train_data, batch_size=32, shuffle=False)
#预测
result = []
with torch.no_grad():
    for inputs in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        result.extend(predicted.cpu().numpy())
        
#写入结果
result = np.array(result)
result = pd.DataFrame(result)
result.index += 1
result.to_csv('result.csv', header=['Label'], index_label='ImageId')

