import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from utils import get_cifar10_loaders, train_model, test_model  # CIFAR-10 kullanılacak

if __name__ == '__main__':
    # Hiperparametreler
    BATCH_SIZE = 32  # Daha büyük modeller için batch size küçültülebilir
    LEARNING_RATE = 0.001
    EPOCHS = 3  # Pretrained model için daha az epoch yeterli olabilir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")

    # Veri Yükleyiciler (CIFAR-10)
    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)
    num_classes = 10  # CIFAR-10 için sınıf sayısı

    # Model (Örn: ResNet18)
    # pretrained=True veya False seçilebilir
    model = models.resnet18(pretrained=True)

    # ResNet'in son katmanını CIFAR-10 sınıf sayısına göre ayarla
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)

    # Kayıp Fonksiyonu ve Optimizatör
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nModel 3 (Pretrained ResNet18) Eğitimi ve Testi (CIFAR-10)")
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=EPOCHS)
    test_model(model, test_loader, device)

    # Modeli Kaydetme (Opsiyonel)
    # torch.save(model.state_dict(), 'model3_pretrained.pth')
    # print("Model 3 kaydedildi: model3_pretrained.pth")
