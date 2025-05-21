import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_loaders(batch_size=64):
    """MNIST veri setini yükler ve DataLoader nesnelerini döndürür."""
    transform = transforms.Compose([
        transforms.Pad(2),  # LeNet-5 32x32 girdi bekler, MNIST 28x28'dir.
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"MNIST Veri Seti Yüklendi:")
    print(f"Eğitim seti boyutu: {len(train_data)}")
    print(f"Test seti boyutu: {len(test_data)}")

    return train_loader, test_loader


def get_cifar10_loaders(batch_size=64):
    """CIFAR-10 veri setini yükler ve DataLoader nesnelerini döndürür."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10 RGB'dir
    ])

    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"CIFAR-10 Veri Seti Yüklendi:")
    print(f"Eğitim seti boyutu: {len(train_data)}")
    print(f"Test seti boyutu: {len(test_data)}")

    return train_loader, test_loader


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5):
    """Modeli eğitir."""
    model.train()
    print("\nEğitim Başlatıldı...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        print(f'Epoch {epoch + 1} tamamlandı, Ortalama Kayıp: {running_loss / len(train_loader):.4f}')
    print("Eğitim Tamamlandı.")


def test_model(model, test_loader, device):
    """Modeli test eder ve doğruluğu döndürür."""
    model.eval()
    correct = 0
    total = 0
    print("\nTest Başlatıldı...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Doğruluğu: {accuracy:.2f}%')
    print("Test Tamamlandı.")
    return accuracy
