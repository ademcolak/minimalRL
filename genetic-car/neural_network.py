"""
Neural Network modülü - Her arabanın beyni!

Basit bir Feedforward Neural Network:
- Input Layer: 6 nöron (5 sensör + 1 hız)
- Hidden Layer: 8 nöron
- Output Layer: 3 nöron (gaz, direksiyon, fren)
"""

import numpy as np


class NeuralNetwork:
    """
    Basit Feedforward Neural Network

    Genetic Algorithm tarafından eğitilecek (gradient descent YOK!)
    """

    def __init__(self, input_size=6, hidden_size=8, output_size=3):
        """
        Neural Network'ü başlat

        Args:
            input_size: Input nöron sayısı (varsayılan: 6)
            hidden_size: Hidden layer nöron sayısı (varsayılan: 8)
            output_size: Output nöron sayısı (varsayılan: 3)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Ağırlıkları rastgele başlat (-1 ile +1 arası)
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))

        # Bias'lar
        self.bias_hidden = np.random.uniform(-1, 1, hidden_size)
        self.bias_output = np.random.uniform(-1, 1, output_size)

    def forward(self, inputs):
        """
        Forward pass - Girdileri işleyip çıktıları üret

        Args:
            inputs: numpy array (örn: [0.8, 0.3, 0.5, 0.9, 0.4, 0.2])

        Returns:
            outputs: numpy array (örn: [0.6, -0.3, 0.1])
        """
        # Input → Hidden Layer
        hidden = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden = np.tanh(hidden)  # Activation function

        # Hidden → Output Layer
        output = np.dot(hidden, self.weights_hidden_output) + self.bias_output
        output = np.tanh(output)  # Activation function

        return output

    def get_weights(self):
        """
        Tüm ağırlıkları ve bias'ları döndür (Genetic Algorithm için)

        Returns:
            dict: Tüm ağırlıklar ve bias'lar
        """
        return {
            'weights_input_hidden': self.weights_input_hidden.copy(),
            'weights_hidden_output': self.weights_hidden_output.copy(),
            'bias_hidden': self.bias_hidden.copy(),
            'bias_output': self.bias_output.copy()
        }

    def set_weights(self, weights):
        """
        Ağırlıkları ve bias'ları dışarıdan ayarla (Genetic Algorithm için)

        Args:
            weights: dict - get_weights() ile döndürülen format
        """
        self.weights_input_hidden = weights['weights_input_hidden'].copy()
        self.weights_hidden_output = weights['weights_hidden_output'].copy()
        self.bias_hidden = weights['bias_hidden'].copy()
        self.bias_output = weights['bias_output'].copy()

    def copy(self):
        """
        Bu neural network'ün kopyasını oluştur

        Returns:
            NeuralNetwork: Yeni kopya
        """
        new_nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        new_nn.set_weights(self.get_weights())
        return new_nn


def test_neural_network():
    """Test fonksiyonu - NN'in çalışıp çalışmadığını kontrol et"""
    print("🧠 Neural Network Test")
    print("-" * 50)

    # NN oluştur
    nn = NeuralNetwork(input_size=6, hidden_size=8, output_size=3)

    # Test input (5 sensör + hız)
    test_input = np.array([0.8, 0.3, 0.5, 0.9, 0.4, 0.2])

    # Forward pass
    output = nn.forward(test_input)

    print(f"Input: {test_input}")
    print(f"Output: {output}")
    print(f"  - Acceleration: {output[0]:.3f}")
    print(f"  - Steering: {output[1]:.3f}")
    print(f"  - Brake: {output[2]:.3f}")

    # Copy test
    nn2 = nn.copy()
    output2 = nn2.forward(test_input)

    print(f"\nCopy test - Same output? {np.allclose(output, output2)}")

    print("\n✅ Neural Network çalışıyor!")


if __name__ == "__main__":
    test_neural_network()
