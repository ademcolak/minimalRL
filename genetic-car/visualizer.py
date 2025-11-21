"""
Visualizer modülü - Pygame ile görselleştirme

Ekran düzeni:
- Sol: Simülasyon (pist + arabalar + sensörler)
- Sağ: Neural Network görselleştirme
- Alt: İstatistikler
"""

import pygame
import math
import numpy as np


class Visualizer:
    """
    Pygame visualizer - Simülasyonu görselleştirir
    """

    def __init__(self, width=1200, height=600):
        """
        Visualizer başlat

        Args:
            width: Ekran genişliği
            height: Ekran yüksekliği
        """
        pygame.init()

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("🚗 Genetic Algorithm - Self-Driving Car")

        # Simülasyon ve NN panel'leri
        self.sim_width = 800
        self.nn_panel_x = self.sim_width
        self.nn_panel_width = width - self.sim_width

        # Renkler
        self.colors = {
            'bg': (40, 44, 52),
            'track': (60, 63, 65),
            'wall': (255, 255, 255),
            'checkpoint': (100, 100, 255, 100),
            'car_alive': (0, 255, 0),
            'car_dead': (100, 100, 100),
            'car_best': (255, 0, 0),
            'sensor': (0, 255, 0, 150),
            'text': (255, 255, 255),
            'nn_node': (100, 200, 255),
            'nn_connection': (50, 50, 50),
            'nn_panel_bg': (30, 34, 42)
        }

        # Font
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)

        self.clock = pygame.time.Clock()

    def draw_track(self, track):
        """
        Pisti çiz

        Args:
            track: Track objesi
        """
        walls = track.get_walls()

        # Dış duvar
        pygame.draw.rect(self.screen, self.colors['wall'], walls['outer'], 3)

        # İç duvar
        pygame.draw.rect(self.screen, self.colors['wall'], walls['inner'], 3)

        # Checkpoint'ler (hafif mavi daireler)
        for cp_x, cp_y, cp_radius in track.get_checkpoints():
            s = pygame.Surface((cp_radius * 2, cp_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, self.colors['checkpoint'],
                             (cp_radius, cp_radius), cp_radius)
            self.screen.blit(s, (cp_x - cp_radius, cp_y - cp_radius))

    def draw_cars(self, cars, show_sensors=True):
        """
        Arabaları çiz

        Args:
            cars: Car listesi
            show_sensors: Sensörleri göster mi?
        """
        # En iyi arabayı bul (kırmızı ile vurgulamak için)
        best_car = max(cars, key=lambda c: c.fitness, default=None)

        for car in cars:
            if not car.alive:
                continue

            # Renk seç
            if car == best_car:
                color = self.colors['car_best']
            else:
                color = self.colors['car_alive']

            # Sensörleri çiz
            if show_sensors and car == best_car:
                endpoints = car.get_sensor_endpoints()
                for end_x, end_y in endpoints:
                    pygame.draw.line(self.screen, self.colors['sensor'],
                                   (car.x, car.y), (end_x, end_y), 1)
                    pygame.draw.circle(self.screen, self.colors['sensor'],
                                     (int(end_x), int(end_y)), 3)

            # Araba gövdesini çiz
            corners = car.get_corners()
            pygame.draw.polygon(self.screen, color, corners, 0)
            pygame.draw.polygon(self.screen, (255, 255, 255), corners, 1)

            # Ön kısmı göster (yön)
            front_corners = corners[1:3]  # Sağ ön köşeler
            mid_x = sum(c[0] for c in front_corners) / 2
            mid_y = sum(c[1] for c in front_corners) / 2
            pygame.draw.circle(self.screen, (255, 255, 0),
                             (int(mid_x), int(mid_y)), 3)

    def draw_neural_network(self, nn, car):
        """
        Neural Network'ü görselleştir (sağ panel)

        Args:
            nn: NeuralNetwork objesi
            car: Car objesi (input değerleri için)
        """
        panel_x = self.nn_panel_x
        panel_width = self.nn_panel_width
        panel_height = self.height

        # Panel arka planı
        pygame.draw.rect(self.screen, self.colors['nn_panel_bg'],
                        (panel_x, 0, panel_width, panel_height))

        # Başlık
        title = self.font_medium.render("Neural Network", True, self.colors['text'])
        self.screen.blit(title, (panel_x + 20, 20))

        # Layer pozisyonları
        layer_x_positions = [
            panel_x + 50,
            panel_x + panel_width // 2,
            panel_x + panel_width - 50
        ]

        node_radius = 8

        # Input layer (6 nöron)
        input_nodes = 6
        input_y_start = 100
        input_y_spacing = 50
        input_positions = []

        for i in range(input_nodes):
            y = input_y_start + i * input_y_spacing
            input_positions.append((layer_x_positions[0], y))

        # Hidden layer (8 nöron)
        hidden_nodes = nn.hidden_size
        hidden_y_start = 100
        hidden_y_spacing = 40
        hidden_positions = []

        for i in range(hidden_nodes):
            y = hidden_y_start + i * hidden_y_spacing
            hidden_positions.append((layer_x_positions[1], y))

        # Output layer (3 nöron)
        output_nodes = 3
        output_y_start = 150
        output_y_spacing = 60
        output_positions = []

        for i in range(output_nodes):
            y = output_y_start + i * output_y_spacing
            output_positions.append((layer_x_positions[2], y))

        # Bağlantıları çiz (Input → Hidden)
        for inp_pos in input_positions:
            for hid_pos in hidden_positions:
                pygame.draw.line(self.screen, self.colors['nn_connection'],
                               inp_pos, hid_pos, 1)

        # Bağlantıları çiz (Hidden → Output)
        for hid_pos in hidden_positions:
            for out_pos in output_positions:
                pygame.draw.line(self.screen, self.colors['nn_connection'],
                               hid_pos, out_pos, 1)

        # Input nöronları çiz
        input_labels = ['S1', 'S2', 'S3', 'S4', 'S5', 'Speed']
        for i, (x, y) in enumerate(input_positions):
            pygame.draw.circle(self.screen, self.colors['nn_node'], (x, y), node_radius)
            label = self.font_small.render(input_labels[i], True, self.colors['text'])
            self.screen.blit(label, (x - 15, y - 25))

        # Hidden nöronları çiz
        for x, y in hidden_positions:
            pygame.draw.circle(self.screen, self.colors['nn_node'], (x, y), node_radius)

        # Output nöronları çiz
        output_labels = ['Gas', 'Steer', 'Brake']
        for i, (x, y) in enumerate(output_positions):
            pygame.draw.circle(self.screen, self.colors['nn_node'], (x, y), node_radius)
            label = self.font_small.render(output_labels[i], True, self.colors['text'])
            self.screen.blit(label, (x - 20, y + 15))

        # Layer başlıkları
        layer_names = ['Input', 'Hidden', 'Output']
        for i, x in enumerate(layer_x_positions):
            label = self.font_small.render(layer_names[i], True, self.colors['text'])
            self.screen.blit(label, (x - 20, 60))

    def draw_stats(self, generation, alive_cars, total_cars, best_fitness, avg_fitness):
        """
        İstatistikleri çiz (alt kısım)

        Args:
            generation: Nesil numarası
            alive_cars: Yaşayan araba sayısı
            total_cars: Toplam araba sayısı
            best_fitness: En iyi fitness
            avg_fitness: Ortalama fitness
        """
        stats_y = self.height - 60

        # Arka plan
        pygame.draw.rect(self.screen, (20, 20, 20),
                        (0, stats_y, self.sim_width, 60))

        # İstatistikler
        stats_text = [
            f"Generation: {generation}",
            f"Alive: {alive_cars}/{total_cars}",
            f"Best: {int(best_fitness)}",
            f"Avg: {int(avg_fitness)}"
        ]

        x_offset = 20
        for text in stats_text:
            label = self.font_medium.render(text, True, self.colors['text'])
            self.screen.blit(label, (x_offset, stats_y + 20))
            x_offset += 200

    def draw_controls(self):
        """
        Kontrol tuşlarını göster (sağ panel altı)
        """
        controls_y = self.height - 150
        controls_x = self.nn_panel_x + 20

        controls = [
            "Controls:",
            "S - Save",
            "L - Load",
            "R - Reset",
            "SPACE - Pause",
            "Q - Quit"
        ]

        for i, text in enumerate(controls):
            label = self.font_small.render(text, True, self.colors['text'])
            self.screen.blit(label, (controls_x, controls_y + i * 20))

    def render(self, track, cars, ga, show_sensors=True):
        """
        Tam ekranı render et

        Args:
            track: Track objesi
            cars: Car listesi
            ga: GeneticAlgorithm objesi
            show_sensors: Sensörleri göster mi?
        """
        # Arka plan
        self.screen.fill(self.colors['bg'])

        # Pist
        self.draw_track(track)

        # Arabalar
        self.draw_cars(cars, show_sensors)

        # En iyi arabayı bul ve NN'ini göster
        best_car = max(cars, key=lambda c: c.fitness, default=None)
        if best_car:
            self.draw_neural_network(best_car.nn, best_car)

        # İstatistikler
        alive_count = sum(1 for car in cars if car.alive)
        self.draw_stats(
            ga.generation,
            alive_count,
            len(cars),
            ga.get_best_fitness(),
            ga.get_avg_fitness()
        )

        # Kontroller
        self.draw_controls()

        # Ekranı güncelle
        pygame.display.flip()

    def handle_events(self):
        """
        Pygame event'lerini işle

        Returns:
            dict: Event bilgileri {'quit': bool, 'pause': bool, ...}
        """
        events = {
            'quit': False,
            'pause': False,
            'save': False,
            'load': False,
            'reset': False
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                events['quit'] = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    events['quit'] = True
                elif event.key == pygame.K_SPACE:
                    events['pause'] = True
                elif event.key == pygame.K_s:
                    events['save'] = True
                elif event.key == pygame.K_l:
                    events['load'] = True
                elif event.key == pygame.K_r:
                    events['reset'] = True

        return events

    def tick(self, fps=60):
        """
        FPS'i kontrol et

        Args:
            fps: Hedef FPS
        """
        self.clock.tick(fps)

    def quit(self):
        """Pygame'i kapat"""
        pygame.quit()
