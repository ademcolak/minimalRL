"""
Genetic Algorithm modülü - Evrim motoru!

GA Döngüsü:
1. Selection: En iyi bireyleri seç
2. Crossover: İki ebeveynin genlerini karıştır
3. Mutation: Rastgele değişiklikler ekle
4. Yeni nesil oluştur
"""

import numpy as np
import random
from neural_network import NeuralNetwork
from car import Car


class GeneticAlgorithm:
    """
    Genetic Algorithm - Neural Network'leri evrimleştirir
    """

    def __init__(self,
                 population_size=50,
                 mutation_rate=0.05,
                 crossover_rate=0.8,
                 elitism_count=2):
        """
        GA parametrelerini ayarla

        Args:
            population_size: Popülasyon büyüklüğü (kaç araba)
            mutation_rate: Mutasyon oranı (0.05 = %5)
            crossover_rate: Crossover oranı (0.8 = %80)
            elitism_count: En iyileri direkt koru (elitism)
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count

        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []

    def create_initial_population(self, start_x, start_y, start_angle):
        """
        İlk nesli oluştur (rastgele ağırlıklar)

        Args:
            start_x: Başlangıç x pozisyonu
            start_y: Başlangıç y pozisyonu
            start_angle: Başlangıç açısı

        Returns:
            list: Car listesi
        """
        population = []
        for _ in range(self.population_size):
            car = Car(start_x, start_y, start_angle)
            population.append(car)

        return population

    def evolve_population(self, population, start_x, start_y, start_angle):
        """
        Yeni nesil oluştur (evrim!)

        Args:
            population: Mevcut popülasyon (Car listesi)
            start_x: Başlangıç x pozisyonu
            start_y: Başlangıç y pozisyonu
            start_angle: Başlangıç açısı

        Returns:
            list: Yeni nesil (Car listesi)
        """
        # Fitness'leri hesapla
        for car in population:
            car.calculate_fitness()

        # İstatistikleri kaydet
        fitnesses = [car.fitness for car in population]
        self.best_fitness_history.append(max(fitnesses))
        self.avg_fitness_history.append(np.mean(fitnesses))

        # Fitness'e göre sırala (en iyiden en kötüye)
        population = sorted(population, key=lambda c: c.fitness, reverse=True)

        # Yeni nesil
        new_population = []

        # 1. ELITISM: En iyi bireyleri direkt al
        for i in range(self.elitism_count):
            elite_car = Car(start_x, start_y, start_angle)
            elite_car.nn = population[i].nn.copy()
            new_population.append(elite_car)

        # 2. CROSSOVER + MUTATION ile geri kalanı oluştur
        while len(new_population) < self.population_size:
            # İki ebeveyn seç (tournament selection)
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)

            # Crossover
            if random.random() < self.crossover_rate:
                child_nn = self._crossover(parent1.nn, parent2.nn)
            else:
                child_nn = parent1.nn.copy()

            # Mutation
            child_nn = self._mutate(child_nn)

            # Yeni araba oluştur
            child_car = Car(start_x, start_y, start_angle)
            child_car.nn = child_nn
            new_population.append(child_car)

        self.generation += 1
        return new_population

    def _tournament_selection(self, population, tournament_size=5):
        """
        Tournament selection - Rastgele birkaç birey seç, en iyisini al

        Args:
            population: Popülasyon listesi
            tournament_size: Turnuva büyüklüğü

        Returns:
            Car: Kazanan birey
        """
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda c: c.fitness)

    def _crossover(self, nn1, nn2):
        """
        Crossover - İki ebeveynin genlerini karıştır

        Uniform crossover: Her ağırlık için rastgele ebeveyn seç

        Args:
            nn1: Neural Network 1
            nn2: Neural Network 2

        Returns:
            NeuralNetwork: Çocuk NN
        """
        child_nn = NeuralNetwork(nn1.input_size, nn1.hidden_size, nn1.output_size)

        # Her katman için crossover
        weights1 = nn1.get_weights()
        weights2 = nn2.get_weights()
        child_weights = {}

        for key in weights1.keys():
            # Rastgele maske oluştur (0 veya 1)
            mask = np.random.randint(0, 2, weights1[key].shape)

            # Mask'e göre ebeveynlerden al
            child_weights[key] = np.where(mask, weights1[key], weights2[key])

        child_nn.set_weights(child_weights)
        return child_nn

    def _mutate(self, nn):
        """
        Mutation - Rastgele değişiklikler ekle

        Args:
            nn: Neural Network

        Returns:
            NeuralNetwork: Mutasyona uğramış NN
        """
        weights = nn.get_weights()

        for key in weights.keys():
            # Her ağırlık için mutation_rate olasılıkla mutasyon uygula
            mutation_mask = np.random.random(weights[key].shape) < self.mutation_rate

            # Küçük rastgele değişiklik ekle
            mutation_values = np.random.randn(*weights[key].shape) * 0.3

            # Mutation uygula
            weights[key] = np.where(mutation_mask,
                                   weights[key] + mutation_values,
                                   weights[key])

            # Sınırla (-2 ile +2 arası)
            weights[key] = np.clip(weights[key], -2, 2)

        nn.set_weights(weights)
        return nn

    def get_best_fitness(self):
        """En iyi fitness'i döndür"""
        return self.best_fitness_history[-1] if self.best_fitness_history else 0

    def get_avg_fitness(self):
        """Ortalama fitness'i döndür"""
        return self.avg_fitness_history[-1] if self.avg_fitness_history else 0


def test_genetic_algorithm():
    """Test fonksiyonu"""
    print("🧬 Genetic Algorithm Test")
    print("-" * 50)

    ga = GeneticAlgorithm(population_size=10, mutation_rate=0.05)

    # İlk nesil
    population = ga.create_initial_population(100, 100, 0)
    print(f"Nesil {ga.generation}: {len(population)} araba oluşturuldu")

    # Dummy fitness ata
    for i, car in enumerate(population):
        car.fitness = random.uniform(10, 100)

    # Yeni nesil
    population = ga.evolve_population(population, 100, 100, 0)
    print(f"Nesil {ga.generation}: Evrim tamamlandı")
    print(f"  Best fitness: {ga.get_best_fitness():.1f}")
    print(f"  Avg fitness: {ga.get_avg_fitness():.1f}")

    # Bir nesil daha
    for car in population:
        car.fitness = random.uniform(20, 150)

    population = ga.evolve_population(population, 100, 100, 0)
    print(f"Nesil {ga.generation}: Evrim tamamlandı")
    print(f"  Best fitness: {ga.get_best_fitness():.1f}")
    print(f"  Avg fitness: {ga.get_avg_fitness():.1f}")

    print("\n✅ Genetic Algorithm çalışıyor!")


if __name__ == "__main__":
    test_genetic_algorithm()
