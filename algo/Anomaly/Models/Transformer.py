import dataclasses
from dataclasses import field
from tensorflow.keras import layers
from tensorflow.keras import models, ops
from tensorflow import keras


@dataclasses.dataclass
class TransformerConfig:
    model_name: str = "transformer"
    model_path: str = ""
    output_activation: str = "sigmoid"
    patch_size: int = 32
    projection_dim: int = 64
    num_heads: int = 4
    transformer_layers: int = 4
    mlp_head_units: list= field(default_factory=lambda:[2048, 1024, 512, 64, 32])


class TransformerModel:
    def __init__(self, params: TransformerConfig, input_shape, output_shape):
        self.params = params
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.output_activation = self.params.output_activation
        self.num_patches = (input_shape[0] // self.params.patch_size) ** 2
        self.transformer_units = [self.params.projection_dim * 2, self.params.projection_dim]

        self.model = self._build_model()
        if self.params.model_path != "":
            self.model.load_weights(self.params.model_path)
            print(f"Model loaded from {self.params.model_path}")

    def _build_model(self):
        inputs = keras.Input(shape=self.input_shape)
        # Create patches
        patches = Patches(self.params.patch_size)(inputs)
        # Encode patches
        encoded_patches = PatchEncoder(self.num_patches, self.params.projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(self.params.transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=self.params.num_heads, key_dim=self.params.projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP
            x3 = mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.3)(representation)
        # Add MLP.
        features = mlp(representation, hidden_units=self.params.mlp_head_units, dropout_rate=0.3)

        prob = layers.Dense(1, activation=self.params.output_activation)(
            features
        )  # Final neuron that outputs classification probability

        # return Keras model.
        return keras.Model(inputs=inputs, outputs=prob)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )


    def call(self, patch):
        positions =\
            ops.expand_dims(ops.arange(start=0, stop=self.num_patches, step=1), axis=0)
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded


if __name__ == '__main__':
    model_config = TransformerConfig()
    model = TransformerModel(model_config, (400, 400, 3), 1)
    model.model.summary()
