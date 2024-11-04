import dataclasses
from tensorflow.keras import layers
import keras_cv
from tensorflow.keras import models


@dataclasses.dataclass
class ResNetConfig:
    model_name: str = "resnet18_v2"
    model_path: str = ""
    output_activation: str = "sigmoid"


class ResNetModel:
    def __init__(self, params: ResNetConfig, input_shape, output_shape):
        self.params = params
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.output_activation = self.params.output_activation
        self.model = self._build_model()

        if self.params.model_path != "":
            self.model.load_weights(self.params.model_path)
            print(f"Model loaded from {self.params.model_path}")

    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        x \
            = keras_cv.models.ResNetV2Backbone.from_preset(self.params.model_name)(inputs)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.AveragePooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(self.output_shape, activation=self.output_activation)(x)
        return models.Model(inputs, x)


if __name__ == '__main__':
    model_config = ResNetConfig()
    model = ResNetModel(model_config, (400, 400, 3), 1)
    model.model.summary()
