# Ankara Üniversitesi Yapay Zeka ve Veri Mühendisliği

## 2024 – 2025 Bahar Dönemi YZM304 Derin Öğrenme Dersi I. Proje Modülü – I. Proje Ödevi

### İçindekiler (Table of Contents)
- [Giriş (Introduction)](#giriş-introduction)
- [Yöntem (Methods)](#yöntem-methods)
  - [Veri Seti ve Ön İşleme](#veri-seti-ve-ön-i̇şleme)
  - [Model Mimarileri](#model-mimarileri)
  - [Ağırlıkların Başlatılması ve Eğitim](#ağırlıkların-başlatılması-ve-eğitim)
  - [Eğitim ve Değerlendirme Adımları](#eğitim-ve-değerlendirme-adımları)
  - [Scikit-learn ve Alternatif Kütüphaneler](#scikit-learn-ve-alternatif-kütüphaneler)
- [Bulgular (Results)](#bulgular-results)
  - [Model Performansları](#model-performansları)
  - [Karşılaştırma Analizi](#karşılaştırma-analizi)
- [Tartışma (Discussion)](#tartışma-discussion)
  - [Katman Sayısı ve İterasyon İlişkisi](#katman-sayısı-ve-i̇terasyon-i̇lişkisi)
  - [Scikit-learn Performansı](#scikit-learn-performansı)
  - [Model Seçimi Kriterleri](#model-seçimi-kriterleri)
  - [Aktivasyon Fonksiyonu Değişimi](#aktivasyon-fonksiyonu-değişimi)
- [Kaynaklar (References)](#kaynaklar-references)

---

## Giriş (Introduction)

Bu projede, Kaggle'dan alınan [BankNote_Authentication](https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data) veri seti kullanılarak ikili sınıflandırma (gerçek banknot ve sahte banknot ayrımı) için derin öğrenme tabanlı modeller geliştirilmiştir. Temel amaç, **2-Layer** (bir gizli katman ve bir çıkış katmanı) ve **3-Layer** (iki gizli katman ve bir çıkış katmanı) yapıları ile tanh aktivasyon fonksiyonunu kullanarak model eğitimi gerçekleştirmektir. Ayrıca, gereksinimler doğrultusunda **ReLU** aktivasyon fonksiyonu ile de benzer deneyler yürütülebilir. Bu rapor, proje kapsamında izlenen adımları ve elde edilen sonuçları IMRAD formatında özetlemektedir.

---

## Yöntem (Methods)

### Veri Seti ve Ön İşleme

- Veri seti, Kaggle üzerindeki "BankNote_Authentication" veri kümesinden elde edilmiştir.
- Veri ön işleme (feature engineering vb.) opsiyonel olup bu projede veri doğrudan kullanılmıştır.
- Veri, `train_test_split` fonksiyonu kullanılarak **%80** eğitim, **%20** test olarak ayrılmıştır. 
- Stratify parametresi ile sınıf dağılımı korunmuştur.

### Model Mimarileri

#### 2-Layer Model:
- Girdi katmanı (4 özellik)
- Gizli katman (6 nöron, **tanh** aktivasyonu)
- Çıkış katmanı (1 nöron, **sigmoid** aktivasyonu)

#### 3-Layer Model:
- Girdi katmanı (4 özellik)
- Birinci gizli katman (6 nöron, **tanh** aktivasyonu)
- İkinci gizli katman (6 nöron, **tanh** aktivasyonu)
- Çıkış katmanı (1 nöron, **sigmoid** aktivasyonu)

### Ağırlıkların Başlatılması ve Eğitim

- Başlangıç ağırlıkları, `np.random.seed(42)` kullanılarak sabitlenmiştir.
- Optimizasyon algoritması olarak **SGD** (Stokastik Gradient Descent) tercih edilmiştir.
- Öğrenme oranı (`learning_rate`) = 0.01 olarak sabit tutulmuştur.
- Kayıp fonksiyonu olarak **binary cross-entropy** (log loss) kullanılmıştır.

### Eğitim ve Değerlendirme Adımları

- Her bir adımda (epoch) ileri yayılım (forward propagation) ile çıkış hesaplanır.
- Binary cross-entropy kaybı üzerinden geri yayılım (backpropagation) gerçekleştirilir.
- Ağırlıklar ve bias değerleri güncellenir.
- Belirli adımlarda (ör. her 1000 iterasyonda) ara kayıp değeri ekrana yazdırılır.
- Test verisi üzerinde **accuracy**, **precision**, **recall**, **f1-score** ve **confusion matrix** hesaplanarak model performansı değerlendirilir.

### Scikit-learn ve Alternatif Kütüphaneler

- Aynı mimari, aynı hiperparametreler (öğrenme oranı, epoch sayısı vb.) ve aynı veri ayrımı ile **Scikit-learn MLPClassifier** üzerinden de eğitilmiştir.
- Ayrıca istenirse benzer yapı PyTorch, TensorFlow gibi kütüphanelerle de oluşturulabilir.

---

## Bulgular (Results)

### Model Performansları

#### 2-Layer Model (Tanh) Sonuçları:
- **n_steps = 800** iterasyonda:
  - Accuracy  : ~0.9818
  - Precision : 1
  - Recall    : ~0.9590
  - F1-Score  : ~0.9791

#### 3-Layer Model (Tanh) Sonuçları:
- **n_steps = 4200** iterasyonda:
  - Accuracy  : ~0.9818
  - Precision : 1
  - Recall    : ~0.9590
  - F1-Score  : ~0.9791

#### Scikit-learn MLPClassifier Sonuçları:
- Yaklaşık **110** iterasyonda benzer performansa ulaşmıştır:
  - Accuracy  : ~0.9818
  - Precision : 1
  - Recall    : ~0.9590
  - F1-Score  : ~0.9791

### Karşılaştırma Analizi

- 2 katmanlı ağ 800 iterasyonda %98 civarı başarıya ulaşabilirken, 3 katmanlı ağın aynı başarıya ulaşması için 4200 iterasyona ihtiyaç duyduğu gözlemlenmiştir.
- Scikit-learn MLPClassifier, benzer ağı daha hızlı bir şekilde (110 iterasyonda) eğitmiştir. Bu farklılık, kütüphanenin iç optimizasyonlarından ve varsayılan parametre ayarlarından kaynaklanıyor olabilir.
- Bütün modellerde sınıflandırma hataları oldukça az olup **True Positive**, **True Negative** değerleri yüksektir.
- **Precision**, **Recall** ve **F1-Score** metrikleri de 0.95 – 0.98 aralığında seyretmektedir.

---

## Tartışma (Discussion)

### Katman Sayısı ve İterasyon İlişkisi

3 katmanlı model, daha derin bir mimari olduğu için aynı öğrenme oranı ve benzer başlangıç koşulları altında 2 katmanlı modele göre daha uzun sürede yakınsamıştır. Parametre sayısının artması, eğitimin daha kararlı ancak daha yavaş ilerlemesine neden olabilmektedir.

### Scikit-learn Performansı

Scikit-learn'in MLPClassifier'ı, varsayılan olarak çeşitli optimizasyon yöntemleri (momentum, adaptif öğrenme oranı vb.) kullanabilir. Bu sayede daha az iterasyonla yüksek başarıya ulaşmak mümkündür.

### Model Seçimi Kriterleri

İstenilen doğruluk eşiği (%90 üzeri gibi) sağlandığında, **en düşük n_steps** ile sonuç veren model seçilebilir. Burada 2 katmanlı model 800 iterasyonda yüksek doğruluğa ulaşmışken, 3 katmanlı modelin aynı doğruluğa 4200 iterasyonda ulaşması zaman açısından dezavantaj yaratmaktadır. Ancak 3 katmanlı model daha karmaşık veri setlerinde potansiyel olarak daha iyi genelleme yapabilir.

### Aktivasyon Fonksiyonu Değişimi

Tanh yerine ReLU kullanıldığında, özellikle derin ağlarda öğrenme hızının artabileceği bilinmektedir. Bu projede ReLU aktivasyon fonksiyonu ile de benzer deneyler yapılarak, farklı aktivasyon fonksiyonlarının performansa etkisi gösterilebilir.

---

## Kaynaklar (References)

1. Kaggle BankNote_Authentication Dataset  
   [https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data](https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data)

2. Scikit-learn Documentation: MLPClassifier  
   [https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

3. PyTorch Documentation  
   [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

4. NumPy Documentation  
   [https://numpy.org/doc/](https://numpy.org/doc/)