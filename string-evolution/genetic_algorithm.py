"""
Genetic Algorithm - Evrim motoru

GA döngüsü:
1. Fitness hesapla
2. Selection (en iyileri seç)
3. Crossover (çaprazlama)
4. Mutation (mutasyon)
5. Yeni nesil
"""

import random
from individual import Individual


class GeneticAlgorithm:
    """Genetic Algorithm motoru"""

    def __init__(self, target, population_size=200, mutation_rate=0.01, elitism_count=1):
        """
        GA'yi başlat

        Args:
            target: Hedef string
            population_size: Popülasyon büyüklüğü
            mutation_rate: Mutasyon oranı (0.0-1.0)
            elitism_count: En iyi kaç birey direkt geçsin
        """
        self.target = target
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count

        self.population = []
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []

    def create_initial_population(self):
        """İlk popülasyonu oluştur (rastgele)"""
        self.population = [
            Individual(target_length=len(self.target))
            for _ in range(self.population_size)
        ]

    def calculate_all_fitness(self):
        """Tüm bireylerin fitness'ini hesapla"""
        for individual in self.population:
            individual.calculate_fitness(self.target)

    def get_best_individual(self):
        """En iyi bireyi bul"""
        return max(self.population, key=lambda ind: ind.fitness)

    def get_average_fitness(self):
        """Ortalama fitness"""
        return sum(ind.fitness for ind in self.population) / len(self.population)

    def tournament_selection(self, tournament_size=5):
        """
        Tournament selection - Rastgele birkaç birey seç, en iyisini al

        Args:
            tournament_size: Turnuva büyüklüğü

        Returns:
            Individual: Kazanan birey
        """
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)

    def crossover(self, parent1, parent2):
        """
        Crossover (çaprazlama) - İki ebeveynden çocuk oluştur

        Args:
            parent1: Ebeveyn 1
            parent2: Ebeveyn 2

        Returns:
            Individual: Çocuk
        """
        # Single-point crossover
        crossover_point = random.randint(1, len(self.target) - 1)

        child_dna = parent1.dna[:crossover_point] + parent2.dna[crossover_point:]
        return Individual(target_length=len(self.target), dna=child_dna)

    def evolve(self):
        """Yeni nesil oluştur!"""
        # Fitness hesapla
        self.calculate_all_fitness()

        # İstatistikleri kaydet
        best_fitness = self.get_best_individual().fitness
        avg_fitness = self.get_average_fitness()
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)

        # Popülasyonu fitness'e göre sırala
        sorted_population = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)

        # Yeni popülasyon
        new_population = []

        # 1. ELITISM - En iyi bireyleri direkt al
        for i in range(self.elitism_count):
            new_population.append(
                Individual(target_length=len(self.target), dna=sorted_population[i].dna)
            )

        # 2. SELECTION + CROSSOVER + MUTATION ile geri kalanı oluştur
        while len(new_population) < self.population_size:
            # Ebeveyn seç (tournament selection)
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            # Crossover
            child = self.crossover(parent1, parent2)

            # Mutation
            child.mutate(self.mutation_rate)

            new_population.append(child)

        # Yeni nesil
        self.population = new_population
        self.generation += 1

    def is_target_found(self):
        """Hedef bulundu mu?"""
        best = self.get_best_individual()
        return best.fitness >= 100.0


def test_genetic_algorithm():
    """Test fonksiyonu"""
    print("🧬 Genetic Algorithm Test")
    print("-" * 50)

    target = "HELLO"
    ga = GeneticAlgorithm(target=target, population_size=50, mutation_rate=0.05)

    print(f"Target: {target}")
    print(f"Population: {ga.population_size}")
    print(f"Mutation rate: {ga.mutation_rate}")
    print()

    # İlk popülasyon
    ga.create_initial_population()
    ga.calculate_all_fitness()

    print(f"Gen {ga.generation}: Best = {ga.get_best_individual()}")

    # 5 nesil çalıştır
    for _ in range(5):
        ga.evolve()
        best = ga.get_best_individual()
        print(f"Gen {ga.generation}: Best = {best}")

        if ga.is_target_found():
            print("\n🎉 Hedef bulundu!")
            break

    print("\n✅ GA test tamamlandı!")


if __name__ == "__main__":
    test_genetic_algorithm()
