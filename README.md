# 🐾 Evcil ve Yırtıcı Hayvan Türlerinin Görüntü Tabanlı Sınıflandırılması

Bu proje, derin öğrenme ve bilgisayarlı görü (Computer Vision) teknikleri kullanılarak 5 farklı hayvan türünü (Kedi, Köpek, Aslan, Çita, Jaguar) sınıflandırmak amacıyla geliştirilmiş özgün bir Evrişimli Sinir Ağı (CNN) modelini içermektedir.

## 📌 Proje Özeti
Doğadaki vahşi kedilerin benzer kürk yapıları (özellikle Çita ve Jaguar) ile evcil hayvanlardaki yüksek ırk çeşitliliğinin bilgisayarlı görü algoritmalarında yarattığı zorlukları aşmak amacıyla PyTorch tabanlı bir CNN mimarisi tasarlanmıştır. Model, daha önce hiç görmediği test veri setinde **%78.80** genel doğruluk oranına ulaşarak başarılı bir genelleme kapasitesi sunmuştur.

## 📊 Veri Seti
Projede kullanılan 5000 görüntülük veri seti iki ana kaynaktan harmanlanmıştır:
* **Evcil Hayvanlar (Kedi, Köpek):** Literatürde standart kabul edilen `Oxford-IIIT Pet Dataset` içerisinden rastgele ırklar seçilerek oluşturulmuştur.
* **Vahşi Hayvanlar (Aslan, Çita, Jaguar):** Python tabanlı otonom web scraping (web kazıma) algoritmaları ile internet üzerinden toplanmıştır.

> **Veri Ön İşleme:** Mükerrer görüntülerin modelin ezberlemesine (overfitting) yol açmasını engellemek için **pHash (Algısal Özetleme)** algoritması ile temizlik yapılmış ve kalite kaybını önlemek adına tüm görüntüler `Lanczos` algoritmasıyla `300x300` piksel boyutunda standartlaştırılmıştır.

## 🧠 Model Mimarisi ve Eğitim Döngüsü
* **Mimari:** Özgün tasarımlı, 3 evrişim (Convolutional) bloklu derin sinir ağı.
* **Optimizasyon:** Adam Optimizer, CrossEntropyLoss.
* **Aşırı Öğrenmeyi Önleme (Regularization):** %50 Dropout, L2 Düzenlileştirme (Weight Decay) ve **Early Stopping** (Patience: 7).
* **Eğitim Sonucu:** Eğitim süreci, modelin ezberlemeye başladığının tespit edilmesiyle 14. epoch'ta Early Stopping tarafından otonom olarak kesilmiş ve modelin en sağlıklı olduğu **7. Epoch** ağırlıkları nihai model olarak kaydedilmiştir.

## 📈 Performans Metrikleri
Test veri seti üzerinde elde edilen sonuçlar:
* **Doğruluk (Accuracy):** %78.80
* **Kesinlik (Precision):** %79.45
* **Duyarlılık (Recall):** %78.80
* **F1-Skoru:** %78.53

## 🛠️ Kullanılan Teknolojiler
* **Dil:** Python 3.12
* **Derin Öğrenme Kütüphanesi:** PyTorch, Torchvision
* **Veri İşleme & Analiz:** NumPy, Matplotlib, Seaborn, Scikit-Learn
* **Ortam:** Jupyter Notebook / Google Colab

## 📁 Proje Dosyaları
* `model_egitimi.ipynb`: Veri ön işleme, veri çekme (scraping), model inşası ve eğitim süreçlerinin bulunduğu ana geliştirme kodları.
* `loss_vs_epoch.png` / `accuracy_vs_epoch.png`: Eğitim ve doğrulama optimizasyon grafiklerinin dökümü.
* `learning_curve.png`: Modelin öğrenme dinamiklerini gösteren bütünleşik eğri.
* `confusion_matrix.png`: Modelin vahşi doğa kamuflajları ve ırk çeşitliliği karşısındaki performansını gösteren sınıflar bazında detaylı hata analizi (Karmaşıklık Matrisi).
    

(VERİ SETİNİN BOYUTU BÜYÜK OLDUĞU İÇİN VERİ SETİNİ YÜKLENMİYOR)
