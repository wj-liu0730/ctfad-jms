import dataclasses
from tensorflow.keras import layers
from tensorflow.keras import models


@dataclasses.dataclass
class AutoEncoderConfig:
    model_name: str = "ConvAutoEncoder"
    model_path: str = ""
    output_activation: str = "sigmoid"
    latent_dim: int = 512


class AutoEncoderModel:
    def __init__(self, params: AutoEncoderConfig, input_shape, output_shape):
        self.params = params
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.latent_dim = self.params.latent_dim
        self.output_activation = self.params.output_activation
        self.model = self._build_model()

        if self.params.model_path != "":
            self.model.load_weights(self.params.model_path)
            print(f"Model loaded from {self.params.model_path}")

    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        image_shape = x.shape

        x = layers.Flatten()(x)
        latent = layers.Dense(self.latent_dim, activation='relu')(x)

        x = layers.Dense(image_shape[1] * image_shape[2] * image_shape[3], activation='relu')(latent)
        x = layers.Reshape((image_shape[1], image_shape[2], -1))(x)

        x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
        x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
        x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
        x = layers.Conv2DTranspose(3, (3, 3), activation=self.output_activation, strides=(2, 2), padding='same')(x)

        return models.Model(inputs, x)


if __name__ == '__main__':
    model_config = AutoEncoderConfig()
    model = AutoEncoderModel(model_config, (400, 400, 3), (400, 400, 3))
    model.model.summary()





