import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from utils import get_cifar10_loaders, train_model, test_model

# Bu dosya, Model 4'te kullanılan *aynı* veri seti üzerinde
# tam bir CNN mimarisini (Model 1, 2 veya 3'teki gibi)
# eğitip test etmek için kullanılır.
# Eğer Model 4 için farklı bir veri seti kullanıldıysa bu adım zorunludur.
# Eğer Model 1, 2 veya 3'ten biri ve aynı veri seti seçildiyse bu dosyaya gerek yoktur.

# Örnek olarak Model 3 (ResNet18) mimarisi ve CIFAR-10 veri seti kullanılıyor.
# Model 4'te kullanılan CNN ve veri seti ile tutarlı olmalıdır.

if __name__ == '__main__':
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 5  # Model 4 ile karşılaştırmak için benzer sayıda epoch olabilir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")

    # Veri Yükleyiciler (Model 4 ile aynı set olmalı)
    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)
    num_classes = 10

    # Model (Model 4'teki özellik çıkarıcı ile aynı temel mimari)
    # Burada sıfırdan eğitiliyor (pretrained=False veya True seçilebilir)
    model = models.resnet18(pretrained=False)  # Sıfırdan eğitim
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nModel 5 (Karşılaştırma CNN - ResNet18) Eğitimi ve Testi (CIFAR-10)")
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=EPOCHS)
    test_model(model, test_loader, device)

    # Modeli Kaydetme (Opsiyonel)
    # torch.save(model.state_dict(), 'model5_comparison.pth')
    # print("Model 5 kaydedildi: model5_comparison.pth")
