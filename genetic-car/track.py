"""
Track modülü - Pist tanımı ve çarpışma kontrolü

Pist:
- Dış ve iç duvarlar
- Checkpoint'ler (ilerleme takibi için)
- Başlangıç pozisyonu
"""

import math


class Track:
    """
    Yarış pisti

    Basit oval/dikdörtgen pist ile başlıyoruz
    """

    def __init__(self, width=800, height=600):
        """
        Pist oluştur

        Args:
            width: Ekran genişliği
            height: Ekran yüksekliği
        """
        self.width = width
        self.height = height

        # Pist kenarları (dış duvar)
        self.outer_margin = 50
        self.outer_rect = (
            self.outer_margin,
            self.outer_margin,
            width - 2 * self.outer_margin,
            height - 2 * self.outer_margin
        )

        # İç duvar
        self.inner_margin = 150
        self.inner_rect = (
            self.inner_margin,
            self.inner_margin,
            width - 2 * self.inner_margin,
            height - 2 * self.inner_margin
        )

        # Başlangıç pozisyonu (pistin solunda, ortada)
        self.start_x = self.outer_margin + 30
        self.start_y = height / 2
        self.start_angle = 0  # Sağa bakıyor

        # Checkpoint'ler (ilerleme takibi için)
        self.checkpoints = self._create_checkpoints()

    def _create_checkpoints(self):
        """
        Checkpoint noktaları oluştur

        Returns:
            list: [(x, y, radius), ...] checkpoint listesi
        """
        checkpoints = []

        # Pistin çevresinde 8 checkpoint koy
        cx = self.width / 2
        cy = self.height / 2
        rx = (self.width - 2 * self.outer_margin - 2 * self.inner_margin) / 4 + self.inner_margin
        ry = (self.height - 2 * self.outer_margin - 2 * self.inner_margin) / 4 + self.inner_margin

        num_checkpoints = 8
        for i in range(num_checkpoints):
            angle = (i / num_checkpoints) * 2 * math.pi
            x = cx + rx * math.cos(angle)
            y = cy + ry * math.sin(angle)
            checkpoints.append((x, y, 40))  # (x, y, radius)

        return checkpoints

    def check_collision(self, x, y):
        """
        Nokta duvara çarpıyor mu kontrol et

        Args:
            x: X koordinatı
            y: Y koordinatı

        Returns:
            bool: True ise çarpışma var
        """
        # Dış duvar kontrolü (pist dışına çıktı mı?)
        outer_x, outer_y, outer_w, outer_h = self.outer_rect
        if (x < outer_x or x > outer_x + outer_w or
            y < outer_y or y > outer_y + outer_h):
            return True

        # İç duvar kontrolü (içerideki engele çarptı mı?)
        inner_x, inner_y, inner_w, inner_h = self.inner_rect
        if (x > inner_x and x < inner_x + inner_w and
            y > inner_y and y < inner_y + inner_h):
            return True

        return False

    def check_checkpoint(self, x, y, current_checkpoint):
        """
        Araba checkpoint'ten geçti mi kontrol et

        Args:
            x: X koordinatı
            y: Y koordinatı
            current_checkpoint: Şu anki checkpoint indexi

        Returns:
            bool: True ise yeni checkpoint'ten geçti
        """
        if current_checkpoint >= len(self.checkpoints):
            return False  # Tüm checkpoint'ler geçildi

        cp_x, cp_y, cp_radius = self.checkpoints[current_checkpoint]
        distance = math.sqrt((x - cp_x)**2 + (y - cp_y)**2)

        return distance < cp_radius

    def get_start_position(self):
        """
        Başlangıç pozisyonunu döndür

        Returns:
            tuple: (x, y, angle)
        """
        return (self.start_x, self.start_y, self.start_angle)

    def get_walls(self):
        """
        Duvar koordinatlarını döndür (çizim için)

        Returns:
            dict: {'outer': rect, 'inner': rect}
        """
        return {
            'outer': self.outer_rect,
            'inner': self.inner_rect
        }

    def get_checkpoints(self):
        """
        Checkpoint'leri döndür

        Returns:
            list: [(x, y, radius), ...]
        """
        return self.checkpoints


class CircularTrack(Track):
    """
    Dairesel pist (daha zor!)

    İleride eklenebilir - şimdilik basit dikdörtgen pist kullanıyoruz
    """
    pass


def test_track():
    """Test fonksiyonu"""
    print("🏁 Track Test")
    print("-" * 50)

    track = Track(800, 600)

    # Başlangıç pozisyonu
    start_x, start_y, start_angle = track.get_start_position()
    print(f"Start: x={start_x}, y={start_y}, angle={start_angle}")

    # Çarpışma testleri
    print(f"\nÇarpışma testleri:")
    print(f"  (100, 100) - İçeride: {not track.check_collision(100, 100)}")
    print(f"  (10, 10) - Dışarıda: {track.check_collision(10, 10)}")
    print(f"  (400, 300) - İç duvarda: {track.check_collision(400, 300)}")

    # Checkpoint sayısı
    print(f"\nCheckpoint sayısı: {len(track.get_checkpoints())}")

    print("\n✅ Track çalışıyor!")


if __name__ == "__main__":
    test_track()
