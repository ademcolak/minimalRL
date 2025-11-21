"""
Car modülü - Araba fiziği ve sensörler

Her araba:
- Kendi neural network'üne (beynine) sahiptir
- 5 sensör ile duvarları algılar (ray-casting)
- NN'den gelen komutlara göre hareket eder
"""

import numpy as np
import math
from neural_network import NeuralNetwork


class Car:
    """
    Self-driving car with neural network brain
    """

    def __init__(self, x, y, angle=0, nn=None):
        """
        Araba oluştur

        Args:
            x: Başlangıç x pozisyonu
            y: Başlangıç y pozisyonu
            angle: Başlangıç açısı (derece)
            nn: Neural Network (None ise yeni oluştur)
        """
        # Pozisyon ve hareket
        self.x = x
        self.y = y
        self.angle = angle  # Derece cinsinden
        self.speed = 0

        # Araba özellikleri
        self.width = 20
        self.height = 10
        self.max_speed = 5
        self.acceleration = 0.2
        self.brake_strength = 0.3
        self.turn_speed = 5

        # Sensörler (5 adet ray-casting)
        self.sensor_angles = [-60, -30, 0, 30, 60]  # Derece
        self.sensor_max_distance = 200
        self.sensor_readings = [1.0] * 5  # Normalize edilmiş (0-1)

        # Neural Network (beyin)
        self.nn = nn if nn else NeuralNetwork()

        # Durum
        self.alive = True
        self.fitness = 0
        self.distance_traveled = 0
        self.checkpoints_passed = 0
        self.time_alive = 0

    def update(self, track, dt=1):
        """
        Arabayı güncelle (her frame)

        Args:
            track: Track objesi (duvarlara çarpma kontrolü için)
            dt: Delta time (zaman adımı)
        """
        if not self.alive:
            return

        self.time_alive += dt

        # Sensörleri güncelle
        self.update_sensors(track)

        # Neural Network'ten karar al
        inputs = np.array(self.sensor_readings + [self.speed / self.max_speed])
        outputs = self.nn.forward(inputs)

        # Çıktıları yorumla
        acceleration_cmd = outputs[0]  # -1 ile +1
        steering_cmd = outputs[1]      # -1 ile +1
        brake_cmd = outputs[2]         # -1 ile +1

        # Hızı güncelle
        if acceleration_cmd > 0:
            self.speed += self.acceleration * acceleration_cmd
        if brake_cmd > 0:
            self.speed -= self.brake_strength * brake_cmd

        # Hız limitleri
        self.speed = max(0, min(self.speed, self.max_speed))

        # Dönüş
        if self.speed > 0.1:  # Sadece hareket ederken dön
            self.angle += self.turn_speed * steering_cmd

        # Pozisyon güncelle
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)

        # Mesafe hesapla
        self.distance_traveled += self.speed

        # Çarpışma kontrolü
        if track.check_collision(self.x, self.y):
            self.alive = False

        # Checkpoint kontrolü
        checkpoint_passed = track.check_checkpoint(self.x, self.y, self.checkpoints_passed)
        if checkpoint_passed:
            self.checkpoints_passed += 1

    def update_sensors(self, track):
        """
        Sensörleri güncelle (ray-casting ile duvara olan mesafe)

        Args:
            track: Track objesi
        """
        for i, sensor_angle in enumerate(self.sensor_angles):
            # Sensör açısını hesapla (araba açısı + sensör açısı)
            angle = self.angle + sensor_angle
            rad = math.radians(angle)

            # Ray-casting: Sensör ışınını at
            distance = 0
            step = 5  # Her adımda 5 pixel ilerle

            for distance in range(0, self.sensor_max_distance, step):
                # Işının ulaştığı nokta
                check_x = self.x + distance * math.cos(rad)
                check_y = self.y + distance * math.sin(rad)

                # Duvara çarptı mı?
                if track.check_collision(check_x, check_y):
                    break

            # Normalize et (0-1 arası)
            self.sensor_readings[i] = distance / self.sensor_max_distance

    def calculate_fitness(self):
        """
        Fitness hesapla (Genetic Algorithm için)

        Fitness = mesafe + checkpoint bonusu
        """
        self.fitness = (
            self.distance_traveled +
            self.checkpoints_passed * 500 +
            self.time_alive * 0.1
        )
        return self.fitness

    def get_sensor_endpoints(self):
        """
        Sensör uç noktalarını döndür (görselleştirme için)

        Returns:
            list: [(x1, y1), (x2, y2), ...] - 5 sensör ucu
        """
        endpoints = []
        for i, sensor_angle in enumerate(self.sensor_angles):
            angle = self.angle + sensor_angle
            rad = math.radians(angle)

            # Sensörün ulaştığı mesafe
            distance = self.sensor_readings[i] * self.sensor_max_distance

            # Uç nokta
            end_x = self.x + distance * math.cos(rad)
            end_y = self.y + distance * math.sin(rad)

            endpoints.append((end_x, end_y))

        return endpoints

    def get_corners(self):
        """
        Arabanın köşe noktalarını döndür (çizim için)

        Returns:
            list: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        """
        rad = math.radians(self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        # Merkeze göre köşeler
        half_w = self.width / 2
        half_h = self.height / 2

        corners = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h)
        ]

        # Döndür ve pozisyona ekle
        rotated_corners = []
        for cx, cy in corners:
            rx = cx * cos_a - cy * sin_a + self.x
            ry = cx * sin_a + cy * cos_a + self.y
            rotated_corners.append((rx, ry))

        return rotated_corners

    def clone_with_brain(self, nn):
        """
        Aynı konumda ama farklı beyin ile araba kopyası oluştur

        Args:
            nn: Neural Network

        Returns:
            Car: Yeni araba
        """
        return Car(self.x, self.y, self.angle, nn)


def test_car():
    """Test fonksiyonu"""
    print("🚗 Car Test")
    print("-" * 50)

    # Dummy track for testing
    class DummyTrack:
        def check_collision(self, x, y):
            # Basit sınır kontrolü
            return x < 0 or x > 800 or y < 0 or y > 600

        def check_checkpoint(self, x, y, current):
            return False

    track = DummyTrack()
    car = Car(400, 300, 0)

    print(f"Başlangıç: x={car.x}, y={car.y}, angle={car.angle}")
    print(f"Sensörler: {car.sensor_readings}")

    # 10 frame simüle et
    for i in range(10):
        car.update(track)

    print(f"10 frame sonra: x={car.x:.1f}, y={car.y:.1f}, speed={car.speed:.2f}")
    print(f"Fitness: {car.calculate_fitness():.1f}")
    print(f"Alive: {car.alive}")

    print("\n✅ Car çalışıyor!")


if __name__ == "__main__":
    test_car()
