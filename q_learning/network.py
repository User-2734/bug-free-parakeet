import tensorflow as tf

class QNetwork(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(24, activation='relu')
        self.fc2 = tf.keras.layers.Dense(16, activation='relu')
        self.out = tf.keras.layers.Dense(output_dim)

    def call(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)