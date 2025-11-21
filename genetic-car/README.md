# 🚗 Genetic Algorithm - Self-Driving Car

Genetik algoritma kullanarak kendi kendine sürmeyi öğrenen arabalar simülasyonu.

## 📖 Proje Hakkında

Bu proje, **Genetic Algorithm (Genetik Algoritma)** ve **Neural Network (Sinir Ağı)** kombinasyonunu kullanarak arabaların bir pistte sürmeyi öğrenmesini sağlar.

### Bu Nedir?

- **Reinforcement Learning (RL) DEĞİL!** Bu bir **Neuroevolution** projesidir.
- Her araba kendi sinir ağına (beynine) sahiptir
- Arabalar yarışır, en başarılılar hayatta kalır
- Başarılı arabaların "genleri" (NN ağırlıkları) çocuklarına geçer
- Her nesil bir öncekinden daha iyi olur

### İlk Nesil vs 50. Nesil:

```
Nesil 1:  🚗💥 (hepsi anında çarpışır)
Nesil 10: 🚗→→💥 (biraz ilerliyorlar)
Nesil 30: 🚗→→→→🏁 (virajları dönebiliyorlar)
Nesil 50: 🚗🏁✅ (pisti tamamlıyorlar!)
```

## 🧬 Nasıl Çalışıyor?

### 1. Neural Network (Her Arabanın Beyni)

```
INPUT (6)          HIDDEN (8)        OUTPUT (3)
────────────────────────────────────────────────
[Sensor 1]  ───┐
[Sensor 2]  ───┤
[Sensor 3]  ───┼──→ [Neuron 1-8] ──→ [Acceleration]
[Sensor 4]  ───┤                     [Steering]
[Sensor 5]  ───┤                     [Brake]
[Speed]     ───┘
```

**Input Nedir?**
- 5 sensör: Duvara olan mesafe (ray-casting)
- 1 hız değeri

**Output Nedir?**
- Acceleration: Gaz (-1 ile +1)
- Steering: Direksiyon (-1 sol, +1 sağ)
- Brake: Fren (0 ile 1)

### 2. Genetic Algorithm Döngüsü

```python
# Her nesil:
1. 50 araba oluştur (ilk nesilde rastgele ağırlıklar)
2. Tüm arabaları pistte yarıştır
3. Fitness hesapla (ne kadar yol aldılar?)
4. En iyi 10 arabayı seç (Selection)
5. Bu 10'dan 50 yeni araba üret:
   - Crossover: İki ebeveynin genlerini karıştır
   - Mutation: %5 rastgele değişiklik
6. Yeni nesil → Adım 2'ye dön
```

### 3. Fitness Function

```python
fitness = distance_traveled + checkpoint_bonus - collision_penalty
```

Ne kadar uzağa gidersen o kadar yüksek fitness!

## 🎮 Kullanım

### Kurulum

```bash
cd genetic-car
pip install -r requirements.txt
```

### Çalıştırma

```bash
python main.py
```

### Kontroller

- **A**: Auto mode (otomatik nesil geçişi - izle ve keyfini çıkar! 🍿)
- **S**: Checkpoint kaydet
- **L**: Checkpoint yükle
- **R**: Sıfırdan başla
- **SPACE**: Duraklat/Devam
- **Q**: Çık

**💡 İpucu:** Auto mode'u aç, otur ve evrimi izle! Nesiller otomatik geçecek.

### Checkpoint Sistemi

Program her 10 nesilden bir otomatik kaydeder:
```
checkpoints/
├── gen_10_fitness_523.pkl
├── gen_20_fitness_1250.pkl
└── gen_30_fitness_1890.pkl
```

Programı kapattığın yerden devam edebilirsin! 🎯

## 📊 Görselleştirme

**Sol Panel**: Simülasyon
- Pist ve arabalar
- Sensör ışınları (yeşil)
- En iyi araba vurgulanır (kırmızı)

**Sağ Panel**: Neural Network Visualization
- En iyi arabanın beyin yapısı
- Layer'lar ve bağlantılar
- Ağırlık değerleri (renk kodlu)

**Alt Panel**: İstatistikler
- Generation (Nesil numarası)
- Best Fitness (En iyi skor)
- Average Fitness (Ortalama)
- Alive Cars (Yaşayan araba sayısı)

## 🔧 Teknik Detaylar

### Kullanılan Teknolojiler:
- **Python 3.8+**
- **NumPy**: Neural network hesaplamaları
- **Pygame**: Gerçek zamanlı görselleştirme
- **Pickle**: Checkpoint kaydetme

### Proje Yapısı:
```
genetic-car/
├── main.py                 # Ana program (checkpoint sistemi)
├── car.py                  # Araba fizik motoru + sensörler
├── neural_network.py       # Feedforward neural network
├── genetic_algorithm.py    # GA mantığı (selection, crossover, mutation)
├── track.py                # Pist tanımı
├── visualizer.py           # Pygame görselleştirme + NN panel
├── requirements.txt        # Bağımlılıklar
├── checkpoints/            # Otomatik kayıt klasörü
└── models/                 # En iyi model klasörü
```

## 📚 Öğrenme Kaynakları

### Bu Proje Hangi Kategoriye Giriyor?
- ✅ **Genetic Algorithm** (Evrimsel algoritma)
- ✅ **Neuroevolution** (Neural network + Evolution)
- ❌ **Reinforcement Learning değil** (gradient descent yok)
- ❌ **Supervised Learning değil** (labeled data yok)

### Benzer Projeler:
- NEAT (NeuroEvolution of Augmenting Topologies)
- Flappy Bird AI
- Snake AI
- Box2D Car Evolution

## 🎯 Sonraki Adımlar

Projeyi geliştirmek için fikirler:
- [ ] Daha kompleks pistler ekle
- [ ] Çoklu pist modu (her nesilde farklı pist)
- [ ] Replay system (en iyi turları kaydet ve izle)
- [ ] Network topology evolution (katman sayısı da evrilsin)
- [ ] Gerçek zamanlı grafik gösterimi
- [ ] Multi-threading ile hızlandırma

## 📝 Notlar

- İlk 10-20 nesil çok kötü performans gösterir (normal!)
- 30-50 nesil sonra ciddi gelişme görülür
- Mutation rate çok önemlİ (%5 ideal başlangıç)
- Populasyon büyüklüğü 50 iyi bir denge (hız vs çeşitlilik)

## 🤝 Katkıda Bulunma

Bu proje öğrenim amaçlı hazırlanmıştır. Geliştirme fikirleri her zaman hoş karşılanır!

---

**Eğlenceli öğrenmeler! 🚀**
