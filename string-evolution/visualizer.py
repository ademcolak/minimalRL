"""
Visualizer - Terminal görselleştirme

Terminal'de güzel çıktılar üretir
"""

import sys


class Visualizer:
    """Terminal görselleştirme"""

    def __init__(self):
        self.colors = {
            'green': '\033[92m',
            'red': '\033[91m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'bold': '\033[1m',
            'end': '\033[0m'
        }

    def clear_screen(self):
        """Ekranı temizle"""
        sys.stdout.write('\033[2J\033[H')
        sys.stdout.flush()

    def display_header(self, target):
        """Başlık göster"""
        print("=" * 70)
        print(f"{self.colors['bold']}🧬 STRING EVOLUTION - Genetic Algorithm{self.colors['end']}")
        print("=" * 70)
        print(f"Target: {self.colors['green']}{self.colors['bold']}\"{target}\"{self.colors['end']}")
        print("=" * 70)
        print()

    def display_generation(self, ga):
        """Nesil bilgisini göster"""
        best = ga.get_best_individual()
        avg_fitness = ga.get_average_fitness()

        print(f"{self.colors['cyan']}Generation {ga.generation}{self.colors['end']}")
        print("-" * 70)

        # En iyi birey
        self._display_individual(best, ga.target)

        # Progress bar
        self._display_progress_bar(best.fitness)

        # İstatistikler
        print(f"Average Fitness: {avg_fitness:.1f}%")
        print()

    def _display_individual(self, individual, target):
        """Bireyi hedef ile karşılaştırarak göster"""
        dna = individual.dna
        result = []

        for i, char in enumerate(dna):
            if char == target[i]:
                # Doğru - yeşil
                result.append(f"{self.colors['green']}{char}{self.colors['end']}")
            else:
                # Yanlış - kırmızı
                result.append(f"{self.colors['red']}{char}{self.colors['end']}")

        print(f"Best: \"{''.join(result)}\" ({individual.fitness:.1f}%)")

    def _display_progress_bar(self, fitness, width=50):
        """Progress bar göster"""
        filled = int((fitness / 100) * width)
        empty = width - filled

        bar = (
            self.colors['green'] + '█' * filled +
            self.colors['white'] + '░' * empty +
            self.colors['end']
        )

        print(f"Progress: [{bar}] {fitness:.1f}%")

    def display_success(self, ga, generations):
        """Başarı mesajı"""
        print()
        print("=" * 70)
        print(f"{self.colors['green']}{self.colors['bold']}🎉 SUCCESS!{self.colors['end']}")
        print(f"Found target in {generations} generations!")
        print("=" * 70)
        print()

    def display_fitness_graph(self, history, width=60, height=15):
        """ASCII fitness grafiği"""
        if len(history) < 2:
            return

        print("Fitness History:")
        print("-" * 70)

        max_fitness = max(history)
        min_fitness = min(history)
        fitness_range = max_fitness - min_fitness if max_fitness > min_fitness else 1

        # Y ekseni değerleri
        y_labels = [int(min_fitness + (fitness_range * i / height)) for i in range(height, -1, -1)]

        # Grafiği çiz
        for i, y_val in enumerate(y_labels):
            line = f"{y_val:3d}│"

            for gen_idx in range(len(history)):
                x_pos = int((gen_idx / (len(history) - 1)) * width)

                if i == len(y_labels) - 1:
                    # Alt çizgi
                    line += "─"
                elif abs(history[gen_idx] - y_val) <= (fitness_range / height):
                    # Veri noktası
                    line += self.colors['green'] + "●" + self.colors['end']
                else:
                    line += " "

            print(line)

        # X ekseni
        x_axis = "   └" + "─" * width
        print(x_axis)

        # X eksen labels
        x_labels = f"    0{' ' * (width - 10)}{len(history) - 1}"
        print(x_labels)
        print()


def test_visualizer():
    """Test fonksiyonu"""
    from genetic_algorithm import GeneticAlgorithm

    viz = Visualizer()
    target = "HELLO"
    ga = GeneticAlgorithm(target=target, population_size=50)

    viz.clear_screen()
    viz.display_header(target)

    ga.create_initial_population()

    for _ in range(3):
        ga.evolve()
        viz.display_generation(ga)

        if ga.is_target_found():
            viz.display_success(ga, ga.generation)
            break

    viz.display_fitness_graph(ga.best_fitness_history)

    print("✅ Visualizer test tamamlandı!")


if __name__ == "__main__":
    test_visualizer()
