"""
Individual class - Tek bir bireyi (DNA) temsil eder

Her birey:
- Bir string (DNA) taşır
- Fitness değerine sahiptir
- Mutasyona uğrayabilir
"""

import random
import string


class Individual:
    """Genetic Algorithm'da tek bir birey"""

    # Kullanılabilir karakterler (A-Z ve boşluk)
    CHARS = string.ascii_uppercase + ' '

    def __init__(self, target_length=11, dna=None):
        """
        Yeni birey oluştur

        Args:
            target_length: DNA uzunluğu
            dna: DNA string'i (None ise rastgele oluşturulur)
        """
        self.target_length = target_length

        if dna is None:
            # Rastgele DNA oluştur
            self.dna = ''.join(random.choice(self.CHARS) for _ in range(target_length))
        else:
            self.dna = dna

        self.fitness = 0.0

    def calculate_fitness(self, target):
        """
        Fitness hesapla (ne kadar hedefe yakın?)

        Args:
            target: Hedef string

        Returns:
            float: Fitness değeri (0-100 arası)
        """
        score = 0
        for i in range(len(target)):
            if self.dna[i] == target[i]:
                score += 1

        self.fitness = (score / len(target)) * 100
        return self.fitness

    def mutate(self, mutation_rate):
        """
        DNA'yı mutasyona uğrat

        Args:
            mutation_rate: Mutasyon oranı (0.0-1.0)
        """
        new_dna = []
        for char in self.dna:
            if random.random() < mutation_rate:
                # Mutasyon! Yeni rastgele karakter
                new_dna.append(random.choice(self.CHARS))
            else:
                # Aynı kal
                new_dna.append(char)

        self.dna = ''.join(new_dna)

    def __str__(self):
        """String representation"""
        return f'"{self.dna}" (Fitness: {self.fitness:.1f}%)'

    def __repr__(self):
        return self.__str__()


def test_individual():
    """Test fonksiyonu"""
    print("🧬 Individual Test")
    print("-" * 50)

    # Rastgele birey
    ind = Individual(target_length=11)
    print(f"Rastgele DNA: {ind}")

    # Fitness hesapla
    target = "HELLO WORLD"
    ind.calculate_fitness(target)
    print(f"Target: {target}")
    print(f"Fitness: {ind.fitness:.1f}%")

    # Mutasyon
    print(f"\nÖnce: {ind.dna}")
    ind.mutate(mutation_rate=0.3)
    print(f"Sonra (30% mutasyon): {ind.dna}")

    # Mükemmel birey
    perfect = Individual(dna="HELLO WORLD")
    perfect.calculate_fitness(target)
    print(f"\nMükemmel birey: {perfect}")

    print("\n✅ Individual test tamamlandı!")


if __name__ == "__main__":
    test_individual()
