import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from utils import get_mnist_loaders, train_model, test_model


# İyileştirilmiş LeNet-5 (Batch Normalization eklenmiş hali)
class ImprovedLeNet5(nn.Module):
    def __init__(self):
        super(ImprovedLeNet5, self).__init__()
        # Katman hiperparametreleri Model 1 ile aynı
        # Evrişimli Katmanlar + Batch Normalization
        self.conv_layers = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('bn1', nn.BatchNorm2d(6)),  # Batch Normalization eklendi
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('bn2', nn.BatchNorm2d(16)),  # Batch Normalization eklendi
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('bn3', nn.BatchNorm2d(120)),  # Batch Normalization eklendi
            ('relu5', nn.ReLU())
        ]))
        # Tam Bağlantılı Katmanlar + Dropout (Opsiyonel)
        self.fc_layers = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
            # ('drop1', nn.Dropout(0.5)), # Dropout eklenebilir [cite: 10, 35]
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.conv_layers(img)
        output = output.view(img.size(0), -1)  # Flatten
        output = self.fc_layers(output)
        return output


if __name__ == '__main__':
    # Hiperparametreler Model 1 ile aynı [cite: 9, 34, 36]
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")

    # Veri Yükleyiciler (Model 1 ile aynı set) [cite: 36]
    train_loader, test_loader = get_mnist_loaders(BATCH_SIZE)

    # Model, Kayıp Fonksiyonu ve Optimizatör
    model = ImprovedLeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nModel 2 (İyileştirilmiş LeNet-5) Eğitimi ve Testi")
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=EPOCHS)
    test_model(model, test_loader, device)

    # Modeli Kaydetme (Opsiyonel)
    # torch.save(model.state_dict(), 'model2_improved.pth')
    # print("Model 2 kaydedildi: model2_improved.pth")
