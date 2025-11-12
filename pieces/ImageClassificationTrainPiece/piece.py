import logging
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from domino.base_piece import BasePiece
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Conv2D, GlobalAveragePooling2D, Dense, Input, ReLU, Dropout
from tensorflow.keras.models import Sequential

from .models import InputModel, OutputModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set default level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Timestamp format
    handlers=[
        logging.FileHandler("app.log"),  # Log to file
        logging.StreamHandler()  # Also log to console
    ]
)

logger = logging.getLogger(__name__)


class ImageClassificationTrainPiece(BasePiece):
    def _read_image_dataset(self, filename, validation_split=None, image_size=(256, 256), batch_size=32):
        assert os.path.exists(filename), "path to dataset does not exist"

        train, validation = keras.utils.image_dataset_from_directory(
            filename,
            validation_split=validation_split,
            subset="both" if validation_split else None,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            seed=42,
        )

        return train, validation

    def _build_model(
        self,
        input_shape: tuple,
        num_classes,
        num_layers,
        filters_per_layer,
        kernel_sizes,
        dropout_rate,
    ):
        """
        Builds a generic, parameterized 1D CNN model.

        Args:
            input_shape (tuple): Shape of input data (timesteps, features).
            num_layers (int): Number of convolutional layers.
            num_classes (int): Number of output classes.
            filters_per_layer (list): Number of filters for each conv layer.
            kernel_sizes (list): Kernel size for each conv layer.
            dropout_rate (float): Dropout rate.
        Returns:
            keras.Model: Compiled Keras 1D CNN model.
        """

        model = Sequential(name="Generic2DCNN")
        model.add(Input(shape=input_shape))

        # Convolutional layers
        for i, no_filters in enumerate(filters_per_layer):
            for j in range(num_layers):
                model.add(Conv2D(
                    filters=no_filters,
                    kernel_size=kernel_sizes[i],
                    padding='same',
                    name=f'conv_{i + 1}_{j + 1}'
                ))
                model.add(ReLU(name=f'relu_{i + 1}_{j + 1}'))

            model.add(BatchNormalization(name=f'bn_{i + 1}'))
            model.add(ReLU(name=f'relu_{i + 1}'))
            model.add(Dropout(rate=dropout_rate))

        model.add(GlobalAveragePooling2D())

        # Dense head
        model.add(Dense(num_classes, activation='softmax', name='dense_1'))

        # Compile
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy'],
        )

        return model

    def piece_function(self, input_data: InputModel):
        logger.debug('piece function')

        if bool(input_data.validation_split) == bool(input_data.validation_data_path):
            raise ValueError(
                "Exactly one method of creating validation set must be specified: "
                "either 'validation_split' or 'validation_data_path'."
            )

        if input_data.validation_split:
            train, validation = self._read_image_dataset(
                input_data.train_data_path, validation_split=0.2, batch_size=input_data.batch_size
            )
        else:
            train = self._read_image_dataset(input_data.train_data_path, batch_size=input_data.batch_size)
            validation = self._read_image_dataset(input_data.validation_data_path, batch_size=input_data.batch_size)

        input_shape = train.element_spec[0].shape[1:]  # (height, width, channels)
        num_classes = len(train.class_names)
        class_names = train.class_names

        train = train.prefetch(tf.data.AUTOTUNE)

        m = self._build_model(
            input_shape=input_shape,
            num_classes=num_classes,
            num_layers=input_data.num_layers,
            filters_per_layer=input_data.filters_per_layer,
            kernel_sizes=input_data.kernel_sizes,
            dropout_rate=input_data.dropout_rate
        )

        best_model_file_path = os.path.join(Path(self.results_path), 'trained_model', 'best_model.keras')
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                best_model_file_path, save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=100, min_lr=0.0001
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=input_data.early_stop_patience, verbose=1),
        ]

        history = m.fit(
            train,
            validation_data=validation,
            batch_size=input_data.batch_size,
            epochs=input_data.epochs,
            callbacks=callbacks,
            verbose=1,
        )

        last_model_file_path = os.path.join(Path(self.results_path), 'trained_model', 'last_model.keras')
        m.save(last_model_file_path)

        config_path = os.path.join(self.results_path, 'trained_model', 'config.json')
        with open(config_path, "w") as f:
            cfg = dict(input_data)
            cfg['class_mapping'] = {i: name for i, name in enumerate(class_names)}
            json.dump(cfg, f)

        metric = "sparse_categorical_accuracy"
        plt.figure()
        plt.plot(history.history[metric])
        plt.plot(history.history["val_" + metric])
        plt.title("Model " + metric)
        plt.ylabel(metric, fontsize="large")
        plt.xlabel("epoch", fontsize="large")
        plt.legend(["train", "val"], loc="best")

        fig_path = os.path.join(Path(self.results_path), f"training_{metric}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Set display result
        self.display_result = {
            'file_type': 'png',
            'file_path': fig_path
        }

        # Return output
        return OutputModel(
            best_model_file_path=best_model_file_path,
            last_model_file_path=last_model_file_path,
            config_path=config_path,
        )
