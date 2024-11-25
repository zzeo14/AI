class MyModel(nn.Module):
    def __init__(self, img_size=32, num_class=100):
        super(MyModel, self).__init__()

        self.img_size = img_size
        self.num_class = num_class

        # (TODO) Model definition
        # Define the layers of your model here
        # Stack multiple convolution layers with pooling and FC layers
        # You can consider existing CNN models (e.g., VGG, ResNet, DenseNet, ...)
        # Input: the number of classes (CIFAR-100 has 100 classes)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(512*(self.img_size//(2**4))**2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, self.num_class)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, img):
        # (TODO) Define forward pass through your model
        # Input: Images (batch_size * channels * image_size * image_size)
        # Output: Predicted labels
        batch_size = img.shape[0]

        out = self.relu((self.maxpool(self.conv1(img))))
        out = self.relu((self.maxpool(self.conv2(out))))
        out = self.relu((self.maxpool(self.conv3(out))))
        out = self.relu((self.maxpool(self.conv4(out))))

        out = out.reshape(batch_size, -1)

        out = self.fc2(self.relu(self.dropout(self.fc1(out))))
        return out
