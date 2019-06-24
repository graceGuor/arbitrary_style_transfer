# Demo - train the style transfer network & use it to generate an image


from __future__ import print_function
import time
from train import train
from infer import stylize
from utils import list_images


IS_TRAINING = True  # False  #

# for training
# local
# TRAINING_CONTENT_DIR = '../../data/style_transfer/MSCOCO/val2017'
# TRAINING_STYLE_DIR = '../../data/style_transfer/wikiart/test_cases'
# serving
TRAINING_CONTENT_DIR = '../../data/style_transfer/MSCOCO/train2017'
# TRAINING_CONTENT_DIR = '../../data/style_transfer/MSCOCO/val2017'
# TRAINING_STYLE_DIR = '../../data/style_transfer/wikiart/test_cases'
TRAINING_STYLE_DIR = '../../data/style_transfer/wikiart/train'
ENCODER_WEIGHTS_PATH = 'vgg19_normalised.npz'
LOGGING_PERIOD = 20

# 在测试时无法随意调节风格化的程度
STYLE_WEIGHTS = [2]
MODEL_SAVE_PATHS = [
    'models/style_weight_2e0.ckpt',
    # 'models_my/style_weight_2e0.ckpt',
]

# for inferring (stylize)
INFERRING_CONTENT_DIR = '../../data/style_transfer/cases/content'
INFERRING_STYLE_DIR = '../../data/style_transfer/cases/styles'
OUTPUTS_DIR = 'outputs_my'


def main():

    if IS_TRAINING:

        content_imgs_path = list_images(TRAINING_CONTENT_DIR)
        style_imgs_path   = list_images(TRAINING_STYLE_DIR)

        for style_weight, model_save_path in zip(STYLE_WEIGHTS, MODEL_SAVE_PATHS):
            print('\n>>> Begin to train the network with the style weight: %.2f\n' % style_weight)

            train(style_weight, content_imgs_path, style_imgs_path, ENCODER_WEIGHTS_PATH, 
                  model_save_path, logging_period=LOGGING_PERIOD, debug=True)

        print('\n>>> Successfully! Done all training...\n')

    else:
        # load all images at a time, so OOM error will occur
        # when content images size and style images size are too big.

        content_imgs_path = list_images(INFERRING_CONTENT_DIR)
        print("content_imgs_path:", content_imgs_path)

        content_imgs_path = list_images(INFERRING_CONTENT_DIR)[18]
        style_imgs_path   = list_images(INFERRING_STYLE_DIR)[:12]

        for style_weight, model_save_path in zip(STYLE_WEIGHTS, MODEL_SAVE_PATHS):
            print('\n>>> Begin to stylize images with style weight: %.2f\n' % style_weight)


            start_time = time.time()
            outputs = stylize(content_imgs_path, style_imgs_path, OUTPUTS_DIR,
                    ENCODER_WEIGHTS_PATH, model_save_path, 
                    suffix='-' + str(style_weight))
            end_time = time.time()
            sum_time = end_time - start_time
            avg_time = sum_time / (len(content_imgs_path) * len(style_imgs_path))
            print("sum_time:", sum_time,
                  "content_imgs size:", len(content_imgs_path),
                  "style_imgs size:", len(style_imgs_path))

        print('\n>>> Successfully! Done all stylizing in {}s'.format(avg_time))


if __name__ == '__main__':
    main()

