import torch



class AlexNet(torch.nn.Module):
    def __init__(self, num_class=100, init_weight=False):
        super(AlexNet, self).__init__()
        self.features = torch.nn.Sequential(
            # conv1 input:3*227*227  output:96*55*55 afterPooling:96*27*27
            torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            # conv2 input:96*27*27  output:256*27*27 afterPooling:256*13*13
            torch.nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            # conv3 input:256*13*13  output:384*13*13
            torch.nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            # conv4 input:384*13*13  output:384*13*13
            torch.nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            # conv5 input:384*13*13  output:256*13*13 afterPooling:256*6*6
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classify = torch.nn.Sequential(
            torch.nn.Linear(256*6*6, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, num_class),
        )
        if init_weight:
            pass

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classify(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 何教授方法
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)  # 正态分布赋值
                torch.nn.init.constant_(m.bias, 0)

