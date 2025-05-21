import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from utils import get_mnist_loaders, train_model, test_model


# LeNet-5 Model Tanımı
class LeNet5(nn.Module):
    """
    Input - 1x32x32 (MNIST için Pad(2) ile boyut ayarlandı)
    Output - 10 (MNIST sınıf sayısı)
    """

    def __init__(self):
        super(LeNet5, self).__init__()
        # Evrişimli Katmanlar
        self.conv_layers = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))
        # Tam Bağlantılı Katmanlar (Sınıflandırıcı)
        self.fc_layers = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))  # CrossEntropyLoss ile LogSoftmax kullanılır
        ]))

    def forward(self, img):
        output = self.conv_layers(img)
        # Düzleştirme Katmanı
        output = output.view(img.size(0), -1)
        output = self.fc_layers(output)
        return output


if __name__ == '__main__':
    # Hiperparametreler
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001  # 0.01 yerine daha stabil bir değer
    EPOCHS = 5  # Örnek için kısa tutuldu, ödev için artırılabilir

    # Cihaz Ayarı (GPU varsa kullan)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")

    # Veri Yükleyiciler
    train_loader, test_loader = get_mnist_loaders(BATCH_SIZE)

    # Model, Kayıp Fonksiyonu ve Optimizatör
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nModel 1 (LeNet-5 Benzeri) Eğitimi ve Testi")
    # Eğitim
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=EPOCHS)

    # Test
    test_model(model, test_loader, device)

    # Modeli Kaydetme (Opsiyonel)
    # torch.save(model.state_dict(), 'model1_lenet.pth')
    # print("Model 1 kaydedildi: model1_lenet.pth")
