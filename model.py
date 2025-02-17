from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, concatenate, BatchNormalization, Lambda, Input, Reshape, ReLU
from tensorflow.python.keras import regularizers
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

RESIZE_FACTOR = 2


def resize_bilinear(x):
    return tf.image.resize(
        x,
        size=[K.shape(x)[1]*RESIZE_FACTOR, K.shape(x)[2]*RESIZE_FACTOR],
        method=tf.image.ResizeMethod.BILINEAR)


def resize_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4
    shape[1] *= RESIZE_FACTOR
    shape[2] *= RESIZE_FACTOR
    return tuple(shape)


class EASTModel:

    def __init__(self, input_size=512):
        scaled_input_size = input_size / 2
        input_image = Input(shape=(input_size, input_size, 3), name='input_image')
        overly_small_text_region_training_mask = Input(shape=(input_size // 4, input_size // 4, 1), name='overly_small_text_region_training_mask')
        text_region_boundary_training_mask = Input(shape=(input_size // 4, input_size // 4, 1), name='text_region_boundary_training_mask')
        target_score_map = Input(shape=(input_size // 4, input_size // 4, 1), name='target_score_map')
        mobilenetv2 = MobileNetV2(input_tensor=input_image, input_shape=(input_size, input_size, 3), weights='imagenet', include_top=False, pooling=None)
        x = mobilenetv2.get_layer('out_relu').output

        x = Lambda(resize_bilinear, output_shape=(input_size // 16, input_size // 16, 1280), name='resize_1')(x)
        x = Reshape((input_size // 16, input_size // 16, 1280))(x)
        x = concatenate([x, mobilenetv2.get_layer('block_13_expand_relu').output], axis=3)
        x = Conv2D(128, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True, fused=False)(x)
        x = ReLU(6.)(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True, fused=False)(x)
        x = ReLU(6.)(x)

        x = Lambda(resize_bilinear, output_shape=(input_size // 8, input_size // 8, 128), name='resize_2')(x)
        x = Reshape((input_size // 8, input_size // 8, 128))(x)
        x = concatenate([x, mobilenetv2.get_layer('block_6_expand_relu').output], axis=3)
        x = Conv2D(64, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True, fused=False)(x)
        x = ReLU(6.)(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True, fused=False)(x)
        x = ReLU(6.)(x)

        x = Lambda(resize_bilinear, output_shape=(input_size // 4, input_size // 4, 64), name='resize_3')(x)
        x = Reshape((input_size // 4, input_size // 4, 64))(x)
        x = concatenate([x, mobilenetv2.get_layer('block_3_expand_relu').output], axis=3)
        x = Conv2D(32, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True, fused=False)(x)
        x = ReLU(6.)(x)
        x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True, fused=False)(x)
        x = ReLU(6.)(x)

        x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True, fused=False)(x)
        x = ReLU(6.)(x)

        pred_score_map = Conv2D(1, (1, 1), activation=tf.nn.tanh, name='pred_score_map')(x)
        pred_score_map = Lambda(lambda x: (x + 1) * 0.5)(pred_score_map)
        rbox_geo_map = Conv2D(4, (1, 1), activation=tf.nn.tanh, name='rbox_geo_map')(x)
        rbox_geo_map = Lambda(lambda x: (x + 1) * scaled_input_size)(rbox_geo_map)
        angle_map = Conv2D(1, (1, 1), activation=tf.nn.tanh, name='rbox_angle_map')(x)
        angle_map = Lambda(lambda x: 0.7853981633974483 * x)(angle_map)
        pred_geo_map = concatenate([rbox_geo_map, angle_map], axis=3, name='pred_geo_map')

        model = Model(inputs=[input_image, overly_small_text_region_training_mask, text_region_boundary_training_mask, target_score_map], outputs=[pred_score_map, pred_geo_map])

        self.model = model
        self.input_image = input_image
        self.overly_small_text_region_training_mask = overly_small_text_region_training_mask
        self.text_region_boundary_training_mask = text_region_boundary_training_mask
        self.target_score_map = target_score_map
        self.pred_score_map = pred_score_map
        self.pred_geo_map = pred_geo_map

