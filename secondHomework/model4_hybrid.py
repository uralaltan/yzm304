import torch
import torch.nn as nn
import numpy as np
from utils import get_cifar10_loaders  # Örnek olarak CIFAR-10
import torchvision.models as models
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib  # Model kaydetmek için


def extract_features(model, data_loader, device):
    """Verilen model ile özellik çıkarır."""
    model.eval()
    features_list = []
    labels_list = []
    print("Özellik Çıkarımı Başlatıldı...")
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            # Sınıflandırıcı katmanını çıkararak özellik vektörlerini al
            features = model(images)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    print("Özellik Çıkarımı Tamamlandı.")
    return np.concatenate(features_list), np.concatenate(labels_list)


if __name__ == '__main__':
    BATCH_SIZE = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")

    # Veri Yükleyiciler (CIFAR-10 veya başka uygun bir set)
    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)
    num_classes = 10

    # Özellik Çıkarıcı Model (Örn: Pretrained ResNet18'in evrişim katmanları)
    feature_extractor = models.resnet18(pretrained=True)
    # Son sınıflandırıcı katmanı çıkar
    feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])
    feature_extractor = feature_extractor.to(device)

    # Eğitim ve Test verilerinden özellik çıkar
    train_features, train_labels = extract_features(feature_extractor, train_loader, device)
    test_features, test_labels = extract_features(feature_extractor, test_loader, device)

    # Özellikleri düzleştir (Flatten)
    train_features = train_features.reshape(train_features.shape[0], -1)
    test_features = test_features.reshape(test_features.shape[0], -1)

    # Özellikleri ve etiketleri .npy olarak kaydet
    np.save('train_features.npy', train_features)
    np.save('train_labels.npy', train_labels)
    np.save('test_features.npy', test_features)
    np.save('test_labels.npy', test_labels)
    print("Özellikler ve etiketler .npy dosyalarına kaydedildi.")

    # Ölçeklendirme (SVM gibi modeller için önerilir)
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    # Kanonik Makine Öğrenmesi Modeli (Örn: SVM)
    print("\nSVM Modeli Eğitimi Başlatıldı...")
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)  # Örnek parametreler
    svm_model.fit(train_features_scaled, train_labels)
    print("SVM Modeli Eğitimi Tamamlandı.")

    # SVM Test
    print("\nSVM Modeli Test Ediliyor...")
    test_preds = svm_model.predict(test_features_scaled)
    svm_accuracy = accuracy_score(test_labels, test_preds) * 100
    print(f'Hibrit Model (ResNet18 Özellikleri + SVM) Test Doğruluğu: {svm_accuracy:.2f}%')

    # SVM Modelini Kaydetme (Opsiyonel)
    # joblib.dump(svm_model, 'svm_model.joblib')
    # joblib.dump(scaler, 'scaler.joblib')
    # print("SVM modeli ve ölçekleyici kaydedildi.")
