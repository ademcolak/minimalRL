"""
Main modülü - Genetic Algorithm Car Simulator

Tüm parçaları bir araya getirir:
- Track
- Cars
- Neural Networks
- Genetic Algorithm
- Visualization
- Checkpoint sistemi
"""

import os
import pickle
from datetime import datetime
from car import Car
from track import Track
from genetic_algorithm import GeneticAlgorithm
from visualizer import Visualizer


class Trainer:
    """
    Ana trainer class - Eğitimi yönetir
    """

    def __init__(self):
        # Track
        self.track = Track(width=800, height=600)

        # Genetic Algorithm
        self.ga = GeneticAlgorithm(
            population_size=50,
            mutation_rate=0.05,
            crossover_rate=0.8,
            elitism_count=2
        )

        # Visualizer
        self.visualizer = Visualizer(width=1200, height=600)

        # Popülasyon
        start_x, start_y, start_angle = self.track.get_start_position()
        self.population = self.ga.create_initial_population(start_x, start_y, start_angle)

        # Durum
        self.running = True
        self.paused = False
        self.generation_time = 0
        self.max_generation_time = 600  # 10 saniye (60 fps * 10)

        # Checkpoint dizini
        os.makedirs('checkpoints', exist_ok=True)

    def run(self):
        """Ana döngü"""
        print("🚗 Genetic Algorithm - Self-Driving Car")
        print("=" * 60)
        print("Başlatılıyor...")
        print(f"Popülasyon: {self.ga.population_size}")
        print(f"Mutation rate: {self.ga.mutation_rate}")
        print(f"Nesil {self.ga.generation} başlıyor...")
        print()

        while self.running:
            # Event'leri işle
            events = self.visualizer.handle_events()

            if events['quit']:
                self.running = False
                break

            if events['pause']:
                self.paused = not self.paused
                print("⏸️  Durakladı" if self.paused else "▶️  Devam ediyor")

            if events['save']:
                self.save_checkpoint()

            if events['load']:
                self.load_checkpoint_interactive()

            if events['reset']:
                self.reset()

            # Eğer durakladıysa sadece çiz
            if self.paused:
                self.visualizer.render(self.track, self.population, self.ga)
                self.visualizer.tick(60)
                continue

            # Simülasyonu güncelle
            self.update_simulation()

            # Render
            self.visualizer.render(self.track, self.population, self.ga)
            self.visualizer.tick(60)

        # Temizlik
        self.visualizer.quit()
        print("\n👋 Görüşürüz!")

    def update_simulation(self):
        """Simülasyonu bir adım güncelle"""
        self.generation_time += 1

        # Tüm arabaları güncelle
        for car in self.population:
            if car.alive:
                car.update(self.track)

        # Tüm arabalar öldü mü veya süre doldu mu?
        alive_count = sum(1 for car in self.population if car.alive)

        if alive_count == 0 or self.generation_time >= self.max_generation_time:
            self.next_generation()

    def next_generation(self):
        """Yeni nesle geç"""
        # Fitness hesapla
        for car in self.population:
            car.calculate_fitness()

        # İstatistikler
        best_fitness = max(car.fitness for car in self.population)
        avg_fitness = sum(car.fitness for car in self.population) / len(self.population)

        print(f"✅ Nesil {self.ga.generation} tamamlandı!")
        print(f"   Best fitness: {int(best_fitness)}")
        print(f"   Avg fitness: {int(avg_fitness)}")

        # Yeni nesil oluştur
        start_x, start_y, start_angle = self.track.get_start_position()
        self.population = self.ga.evolve_population(self.population, start_x, start_y, start_angle)

        # Zamanı sıfırla
        self.generation_time = 0

        print(f"🧬 Nesil {self.ga.generation} başlıyor...")

        # Otomatik kaydetme (her 10 nesilde bir)
        if self.ga.generation % 10 == 0:
            self.save_checkpoint(auto=True)

    def save_checkpoint(self, auto=False):
        """
        Checkpoint kaydet

        Args:
            auto: Otomatik kaydetme mi?
        """
        best_fitness = int(self.ga.get_best_fitness())
        filename = f'checkpoints/gen_{self.ga.generation}_fitness_{best_fitness}.pkl'

        checkpoint = {
            'generation': self.ga.generation,
            'population': [car.nn.get_weights() for car in self.population],
            'best_fitness_history': self.ga.best_fitness_history,
            'avg_fitness_history': self.ga.avg_fitness_history,
            'timestamp': datetime.now().isoformat()
        }

        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)

        prefix = "💾 [AUTO]" if auto else "💾"
        print(f"{prefix} Checkpoint kaydedildi: {filename}")

    def load_checkpoint_interactive(self):
        """Checkpoint yükle (interaktif)"""
        checkpoints = self.list_checkpoints()

        if not checkpoints:
            print("❌ Checkpoint bulunamadı!")
            return

        print("\n📂 Mevcut checkpoints:")
        for i, cp in enumerate(checkpoints, 1):
            print(f"  [{i}] {cp['filename']} (Gen: {cp['generation']}, Fitness: {cp['fitness']})")

        # İlk checkpoint'i yükle (en yenisi)
        self.load_checkpoint(checkpoints[0]['filepath'])

    def load_checkpoint(self, filepath):
        """
        Checkpoint yükle

        Args:
            filepath: Checkpoint dosya yolu
        """
        try:
            with open(filepath, 'rb') as f:
                checkpoint = pickle.load(f)

            # GA state'i geri yükle
            self.ga.generation = checkpoint['generation']
            self.ga.best_fitness_history = checkpoint['best_fitness_history']
            self.ga.avg_fitness_history = checkpoint['avg_fitness_history']

            # Popülasyonu geri yükle
            start_x, start_y, start_angle = self.track.get_start_position()
            self.population = []

            for weights in checkpoint['population']:
                car = Car(start_x, start_y, start_angle)
                car.nn.set_weights(weights)
                self.population.append(car)

            self.generation_time = 0

            print(f"✅ Checkpoint yüklendi: {filepath}")
            print(f"   Nesil: {self.ga.generation}")
            print(f"   Best fitness: {int(self.ga.get_best_fitness())}")

        except Exception as e:
            print(f"❌ Checkpoint yüklenemedi: {e}")

    def list_checkpoints(self):
        """
        Checkpoint'leri listele

        Returns:
            list: Checkpoint bilgileri
        """
        checkpoints = []

        if not os.path.exists('checkpoints'):
            return checkpoints

        for filename in os.listdir('checkpoints'):
            if filename.endswith('.pkl'):
                filepath = os.path.join('checkpoints', filename)

                # Dosya adından bilgi çıkar
                # Format: gen_20_fitness_1250.pkl
                parts = filename.replace('.pkl', '').split('_')
                try:
                    generation = int(parts[1])
                    fitness = int(parts[3])

                    checkpoints.append({
                        'filename': filename,
                        'filepath': filepath,
                        'generation': generation,
                        'fitness': fitness
                    })
                except:
                    pass

        # Nesile göre sırala (en yeni önce)
        checkpoints.sort(key=lambda x: x['generation'], reverse=True)
        return checkpoints

    def reset(self):
        """Sıfırdan başla"""
        print("\n🔄 Sıfırdan başlatılıyor...")

        self.ga = GeneticAlgorithm(
            population_size=50,
            mutation_rate=0.05,
            crossover_rate=0.8,
            elitism_count=2
        )

        start_x, start_y, start_angle = self.track.get_start_position()
        self.population = self.ga.create_initial_population(start_x, start_y, start_angle)
        self.generation_time = 0

        print(f"✅ Nesil 0'dan başlatıldı!")


def main():
    """Ana fonksiyon"""
    print()
    print("=" * 60)
    print("🚗 GENETIC ALGORITHM - SELF-DRIVING CAR")
    print("=" * 60)
    print()

    # Checkpoint var mı kontrol et
    trainer = Trainer()
    checkpoints = trainer.list_checkpoints()

    if checkpoints:
        print("📂 Mevcut checkpoints bulundu:")
        for i, cp in enumerate(checkpoints[:5], 1):  # İlk 5'ini göster
            print(f"  [{i}] Gen: {cp['generation']}, Fitness: {cp['fitness']}")

        print()
        response = input("Checkpoint yüklemek ister misin? (y/n): ").strip().lower()

        if response == 'y':
            # En yeni checkpoint'i yükle
            trainer.load_checkpoint(checkpoints[0]['filepath'])

    print()
    print("▶️  Simülasyon başlıyor...")
    print()
    print("Kontroller:")
    print("  S - Save checkpoint")
    print("  L - Load checkpoint")
    print("  R - Reset (sıfırdan başla)")
    print("  SPACE - Pause/Resume")
    print("  Q - Quit")
    print()

    # Ana döngüyü başlat
    trainer.run()


if __name__ == "__main__":
    main()
