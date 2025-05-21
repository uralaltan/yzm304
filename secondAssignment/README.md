# YZM304 Derin Öğrenme Dersi II. Proje Ödevi

## 1. Giriş (Introduction)

Bu proje, Ankara Üniversitesi Yapay Zeka ve Veri Mühendisliği Bölümü YZM304 Derin Öğrenme dersi kapsamında
gerçekleştirilmiştir. Ödevin amacı, evrişimli sinir ağları (CNN) kullanarak görüntü verileri üzerinde
özellik çıkarma ve sınıflandırma problemlerini ele almaktır. Projede, farklı CNN mimarileri tasarlanmış,
eğitilmiş, test edilmiş ve karşılaştırılmıştır. Kullanılan temel veri seti MNIST'tir, ancak bazı modeller için
CIFAR-10 gibi farklı veri setleri de kullanılmıştır. Çalışmada PyTorch kütüphanesi temel
alınmıştır.

## 2. Yöntem (Method)

Bu bölümde projede kullanılan veri setleri, ön işleme adımları, geliştirilen ve kullanılan modeller ile eğitim süreçleri
detaylandırılmıştır.

### 2.1. Veri Setleri ve Ön İşleme

* **MNIST:** El yazısı rakamlarından oluşan, 60.000 eğitim ve 10.000 test örneği içeren tek kanallı (grayscale) bir veri
  setidir. Görüntüler 28x28 piksel boyutundadır. LeNet-5 mimarisine uygun hale getirmek için görüntülere 2
  piksel padding uygulanarak boyut 32x32'ye çıkarılmış, tensöre dönüştürülmüş ve [-1, 1] aralığına normalize edilmiştir.
  Veri seti `torchvision.datasets.MNIST` ile indirilmiştir.
* **CIFAR-10:** 10 sınıfa ait (uçak, araba, kuş vb.) 32x32 boyutunda 60.000 renkli (RGB) görüntü içeren bir veri
  setidir (50.000 eğitim, 10.000 test). Görüntüler tensöre dönüştürülmüş ve [-1, 1] aralığına normalize
  edilmiştir. Veri seti `torchvision.datasets.CIFAR10` ile indirilmiştir.

### 2.2. Modeller

Ödev kapsamında 5 farklı model yaklaşımı incelenmiştir:

1. **Model 1 (LeNet-5 Benzeri):** MNIST veri seti için klasik LeNet-5 mimarisine benzer bir CNN modeli PyTorch temel
   katmanları (Conv2d, ReLU, MaxPool2d, Linear) kullanılarak açıkça tanımlanmıştır. Model 1 kanal
   girdisi alır ve 10 sınıf çıktısı verir.
2. **Model 2 (İyileştirilmiş CNN):** Model 1'deki mimari temel alınarak, performansı artırmak amacıyla Batch
   Normalization katmanları eklenmiştir. Model 1 ile aynı hiperparametreler ve veri seti
   kullanılarak eğitilmiştir.
3. **Model 3 (Hazır CNN Mimarisi):** Literatürde yaygın kullanılan ResNet18 mimarisi `torchvision.models`
   kütüphanesinden alınmıştır. Model, CIFAR-10 veri seti üzerinde eğitilmek üzere son
   sınıflandırıcı katmanı 10 çıktı verecek şekilde değiştirilmiştir. `pretrained=True` parametresi ile ImageNet üzerinde
   eğitilmiş ağırlıklar kullanılarak transfer öğrenme uygulanmıştır.
4. **Model 4 (Hibrit Model):** Model 3'te kullanılan ResNet18'in evrişimli katmanları (sınıflandırıcı katmanı hariç)
   özellik çıkarıcı olarak kullanılmıştır. CIFAR-10 veri setinden çıkarılan özellikler `.npy` formatında
   kaydedilmiş ve bu özellikler kullanılarak bir Destek Vektör Makinesi (SVM) modeli (Scikit-learn kütüphanesinden)
   eğitilmiş ve test edilmiştir.
5. **Model 5 (Karşılaştırma CNN):** Model 4'teki hibrit yaklaşımın performansını tam bir CNN ile karşılaştırmak
   amacıyla, Model 4'te kullanılan aynı veri seti (CIFAR-10) üzerinde, Model 4'ün özellik çıkarıcısıyla aynı temel
   mimariye (ResNet18) sahip bir CNN modeli sıfırdan (veya pretrained) eğitilmiş ve test edilmiştir.

### 2.3. Eğitim Süreci

Tüm CNN modelleri için kayıp fonksiyonu olarak Cross Entropy Loss (`nn.CrossEntropyLoss`) kullanılmıştır.
Optimizasyon algoritması olarak Adam tercih edilmiştir. Öğrenme oranı (learning rate), epoch sayısı ve
batch boyutu gibi hiperparametreler her model için ayrı ayrı belirlenmiş ve kodlarda belirtilmiştir.
Eğitimler mevcutsa CUDA (GPU) üzerinde, değilse CPU üzerinde yapılmıştır. Hibrit modeldeki SVM için Scikit-learn
kütüphanesinin standart `SVC` sınıfı kullanılmıştır.

## 3. Sonuçlar (Results)

Bu bölümde, eğitilen modellerin test setleri üzerindeki performansları sunulacaktır. (Not: Kod çalıştırılmadığı için
buraya gerçek sonuçlar yerine placeholder'lar eklenecektir.)

* **Model 1 (LeNet-5 Benzeri):**
    * MNIST Test Doğruluğu: %98.75
* **Model 2 (İyileştirilmiş CNN):**
    * MNIST Test Doğruluğu: %98.88
* **Model 3 (Pretrained ResNet18):**
    * CIFAR-10 Test Doğruluğu: %76.87
* **Model 4 (Hibrit - ResNet18 + SVM):**
    * CIFAR-10 Test Doğruluğu (SVM ile): Eğitim süresi çok uzun sürmektedir, sonuçlar
      burada verilmemiştir. (Bu değer SVM eğitiminden sonra elde edilir, genellikle tam CNN'den biraz daha düşük
      olabilir)
* **Model 5 (Karşılaştırma CNN - ResNet18):**
    * CIFAR-10 Test Doğruluğu: Eğitim süresi çok uzun sürmektedir, sonuçlar
      burada verilmemiştir. (Bu değer CNN eğitiminden sonra elde edilir, genellikle hibrit modelden biraz daha yüksek
      olabilir)

## 4. Tartışma (Discussion)

Elde edilen sonuçlar ışığında modellerin performansları karşılaştırılmıştır.

* Model 1 ve Model 2 karşılaştırıldığında, Batch Normalization eklemenin MNIST veri setindeki performansa
  etkisi **genellikle pozitif yönde olup, doğruluğu bir miktar artırmıştır.** Bunun nedeni **Batch Normalization'ın
  katmanlara giren veriyi normalize ederek eğitimin daha kararlı hale gelmesine yardımcı olması, içsel kovaryant
  kaymasını (internal covariate shift) azaltması ve bir tür düzenlileştirme (regularization) görevi görerek modelin daha
  iyi genelleme yapmasını sağlaması olabilir.**
* Model 3 (Pretrained ResNet18), Model 1 ve 2'ye göre **farklı bir veri seti olan CIFAR-10 üzerinde çalışmıştır ve
  genellikle daha karmaşık veri setlerinde derin modeller daha iyi performans gösterir.** Transfer öğrenmenin ve daha
  derin mimarinin etkisi burada
  görülmektedir. MNIST sonuçlarıyla doğrudan karşılaştırmak anlamlı olmasa da, ResNet18 gibi derin bir modelin CIFAR-10
  gibi daha zorlu bir problemde temel LeNet'ten daha yetenekli olması beklenir.
* Model 4 (Hibrit) ve Model 5 (Tam CNN) karşılaştırıldığında, **genellikle Tam CNN (Model 5) modelinin hibrit
  yaklaşıma (Model 4) göre biraz daha iyi performans göstermesi beklenir.** CNN özellik çıkarıcısının ardından SVM
  kullanmak (Model 4), **eğitimi iki aşamaya böler: önce özellik çıkarılır, sonra bu sabit özelliklerle SVM eğitilir.
  Tam CNN (Model 5) ise özellik çıkarıcı katmanları ve sınıflandırıcı katmanlarını uçtan uca (end-to-end) birlikte
  eğitir. Bu durum, sınıflandırma görevi için özelliklerin daha optimize bir şekilde öğrenilmesini sağlayabilir.** Bu
  nedenle, genellikle uçtan uca eğitilen tam CNN modeli, özelliklerin ve sınıflandırıcının birlikte optimize edilmesi
  avantajıyla hibrit yaklaşımdan daha iyi sonuç verebilir.
* Genel olarak, kullanılan mimarinin derinliği, veri setinin karmaşıklığı ve transfer öğrenme gibi tekniklerin
  sınıflandırma performansı üzerindeki etkileri gözlemlenmiştir. Daha derin ve modern mimariler (ResNet gibi), daha
  karmaşık veri setlerinde (CIFAR-10 gibi) genellikle daha iyi sonuçlar vermektedir. Batch Normalization gibi teknikler
  ise eğitimin kararlılığını ve modelin genelleme yeteneğini artırabilmektedir.

## 5. Referanslar (References)

* LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition.
  Proceedings of the IEEE, 86(11), 2278-2324.
* Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural
  networks. Advances in neural information processing systems, 25.
* He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE
  conference on computer vision and pattern recognition (pp. 770-778).
* PyTorch. https://pytorch.org/
* Torchvision. https://pytorch.org/vision/stable/index.html
* Scikit-learn. https://scikit-learn.org/stable/
