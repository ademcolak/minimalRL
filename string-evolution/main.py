"""
Main - String Evolution Genetic Algorithm

Tüm parçaları bir araya getirir ve çalıştırır
"""

import time
from genetic_algorithm import GeneticAlgorithm
from visualizer import Visualizer


def main():
    """Ana fonksiyon"""
    # Hedef string
    target = "Bu Bir Deneme Yazısı"

    # Parametreler
    population_size = 400
    mutation_rate = 0.02  # %2
    elitism_count = 2

    # GA ve Visualizer oluştur
    ga = GeneticAlgorithm(
        target=target,
        population_size=population_size,
        mutation_rate=mutation_rate,
        elitism_count=elitism_count
    )
    viz = Visualizer()

    # Başlangıç
    viz.clear_screen()
    viz.display_header(target)

    print(f"Population Size: {population_size}")
    print(f"Mutation Rate: {mutation_rate * 100}%")
    print(f"Elitism: {elitism_count}")
    print()
    input("Press ENTER to start evolution...")

    # İlk popülasyon
    ga.create_initial_population()
    ga.calculate_all_fitness()  # İlk fitness'leri hesapla

    # Ana döngü
    while not ga.is_target_found():
        # Yeni nesil
        ga.evolve()

        # Görselleştir
        viz.clear_screen()
        viz.display_header(target)
        viz.display_generation(ga)

        # Biraz bekle (animasyon gibi)
        time.sleep(0.05)

        # Maksimum nesil kontrolü (sonsuz döngüyü önle)
        if ga.generation > 10000:
            print("⚠️  10000 nesil aşıldı, durduruluyor...")
            break

    # Başarı!
    if ga.is_target_found():
        viz.display_success(ga, ga.generation)
        viz.display_fitness_graph(ga.best_fitness_history)

    # Final istatistikleri
    print("Final Statistics:")
    print("-" * 70)
    print(f"Total Generations: {ga.generation}")
    print(f"Best Fitness: {ga.get_best_individual().fitness:.1f}%")
    print(f"Average Fitness: {ga.get_average_fitness():.1f}%")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️  Interrupted by user")
        print("Goodbye! 👋")
