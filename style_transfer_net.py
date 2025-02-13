# Style Transfer Network
# Encoder -> AdaIN -> Decoder

import tensorflow as tf

from encoder import Encoder
from decoder import Decoder
from adaptive_instance_norm import AdaIN


class StyleTransferNet(object):

    def __init__(self, encoder_weights_path):
        self.encoder = Encoder(encoder_weights_path)
        self.decoder = Decoder()

    def transform(self, content, style):
        # switch RGB to BGR
        content = tf.reverse(content, axis=[-1])
        style = tf.reverse(style, axis=[-1])

        # preprocess image
        content = self.encoder.preprocess(content)
        style = self.encoder.preprocess(style)

        # encode image
        enc_c, self.encoded_content_layers = self.encoder.encode(content)
        enc_s, self.encoded_style_layers = self.encoder.encode(style)

        # pass the encoded images to AdaIN
        self.target_features = AdaIN(enc_c, enc_s)

        # decode target features back to image
        generated_img = self.decoder.decode(self.target_features)

        # deprocess image
        generated_img = self.encoder.deprocess(generated_img)

        # switch BGR back to RGB
        generated_img = tf.reverse(generated_img, axis=[-1])

        # clip to 0..255
        generated_img = tf.clip_by_value(generated_img, 0.0, 255.0)

        return generated_img

