import torch
from Caltech101_Dataset import Caltech101Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from AlexNet import AlexNet
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_csv_dir = 'D:/DataBase/Calteck101/train.csv'
valid_csv_dir = 'D:/DataBase/Calteck101/valid.csv'
test_csv_dir = 'D:/DataBase/Calteck101/test.csv'

transform = transforms.Compose([
    transforms.Resize([227, 227]),
    transforms.ToTensor()
])

train_dataset = Caltech101Dataset(train_csv_dir, transform)
valid_dataset = Caltech101Dataset(valid_csv_dir, transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=128, shuffle=True)

net = AlexNet(num_class=101, init_weight=True)
net.to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)
save_path = './AlexNet.pth'
best_acc = 0.0

epoch = 0

# checkpoint = torch.load(save_path)
# net.load_state_dict(checkpoint)


for epoch in range(200):
    running_loss = 0.0
    time_start = time.time()
    for step, data in enumerate(train_loader):
        net.train()
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        predictions = net(images)
        loss = loss_function(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\r[epoch: {:d}]{:^3.0f}%[{}->{}]train loss: {:.3f}".format(epoch + 1, int(rate * 100), a, b, loss), end="")
    time_end = time.time()
    print()
    print("training time: %.3f" % (time_end - time_start))

    net.eval()
    acc = 0.0
    with torch.no_grad():
        for val_step, val_data in enumerate(valid_loader):
            val_images, val_labels = val_data
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            predictions = net(val_images)
            # print('predictions.size():', predictions.size())
            prediction = torch.max(predictions, dim=1)[1]
            # print('prediction.size():', prediction.size())
            # print(prediction)
            # print(val_labels)
            acc += (prediction == val_labels).sum().item()
        val_acc = acc / len(valid_dataset)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)
        print("train_loss: %.3f, test_accuracy: %.3f" % (running_loss / len(train_loader), val_acc))

    print("best_accuracy: %.3f" % best_acc)
