# 🔤 String Evolution - Genetic Algorithm

Genetik algoritma ile hedef string'e evrimleşme projesi.

## 🎯 Hedef

Rastgele harflerden başlayıp "HELLO WORLD" gibi bir hedef string'e evrimleşmek.

## 📋 Proje Durumu

🚧 **Geliştirme aşamasında** - Class yapıları planlanıyor...

### Planlanmış Modüller:
- [ ] `individual.py` - Birey class'ı (DNA taşıyıcı)
- [ ] `genetic_algorithm.py` - GA motoru (selection, crossover, mutation)
- [ ] `visualizer.py` - Terminal görselleştirme
- [ ] `main.py` - Ana program

## 🧬 Genetic Algorithm Prensipleri

Bu projede göreceğin GA teknikleri:
- **Selection:** Tournament Selection
- **Crossover:** Single-point veya Uniform
- **Mutation:** Random character mutation
- **Elitism:** En iyi bireyi koru

## 📊 Örnek Çıktı (Hedef)

```
Generation: 0
Best: "XKCBQ ZARTF" (Fitness: 0.0%)

Generation: 10
Best: "HEaLO asdfg" (Fitness: 45.5%)

Generation: 25
Best: "HELLO WOasD" (Fitness: 72.7%)

Generation: 42
Best: "HELLO WORLD" (Fitness: 100.0%) ✅
```

---

**Not:** Pure Python kullanılıyor - grafik kütüphanesi yok!
