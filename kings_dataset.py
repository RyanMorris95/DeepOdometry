import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, sys

HEIGHT = 1080
WIDTH = 1920
DEPTH = 3


class KingsDataset(object):
    def __init__(self, data_dir, subset='train', use_distortion=True, abs_and_rel=False):
        self.data_dir = data_dir
        self.subset = subset
        self.use_distortion = use_distortion
        self.abs_and_rel = abs_and_rel

    def get_filenames(self):
        if self.subset in ['train', 'validation', 'test']:
            return [os.path.join(self.data_dir, self.subset + '.tfrecord')]
        elif self.subset in 'pred':
            return [os.path.join(self.data_dir, 'train_absrel_scale.tfrecord')]
        else:
            raise ValueError('Invalid data subset "%s"' % self.subset)

    def parser(self, record):
        keys_to_features = {
            "pose_rel": tf.FixedLenFeature((), tf.string, default_value=""),
            "pose_abs": tf.FixedLenFeature((), tf.string, default_value=""),
            "image": tf.FixedLenFeature((), tf.string, default_value=""),
        }

        parsed = tf.parse_single_example(record, keys_to_features)

        image = tf.image.decode_jpeg(parsed["image"])
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.reshape(image, [HEIGHT, WIDTH, 3])  # image will be in 1 - 0 format
        image = tf.image.resize_image_with_crop_or_pad(image, 1020, 1020)

        label_abs = tf.decode_raw(parsed["pose_abs"], tf.float64)
        label_abs = tf.reshape(label_abs, [7])

        label_rel = tf.decode_raw(parsed["pose_rel"], tf.float64)
        label_rel = tf.reshape(label_rel, [7])

        return image, label_rel, label_abs

    def make_batch(self, batch_size):
        filenames = self.get_filenames()

        dataset = tf.data.TFRecordDataset(filenames).repeat()

        # parse records
        dataset = dataset.map(self.parser, num_parallel_calls=8)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        image_batch, labels_rel, labels_abs = iterator.get_next()

        image_batch = self.preprocess(image_batch)

        if self.abs_and_rel:
            labels_relabs = tf.concat([labels_rel, labels_abs], axis=1)
            return image_batch, labels_relabs
        else:
            return image_batch, labels_abs

    def _image_augmentation(self, image, seed=42):
        # Perform additional preprocessing on the parsed data.
        # hue_delta = tf.random_uniform([1], minval=0.9, maxval=1.1, seed=seed)
        # sat_delta = tf.random_uniform([1], minval=0.9, maxval=1.1, seed=seed)
        # gam_delta = tf.random_uniform([1], minval=0.9, maxval=1.1, seed=seed)
        #
        # image = tf.image.random_brightness(image, 0.1, seed=seed)
        # image = tf.image.random_contrast(image, lower=0.9, upper=1.1, seed=seed)
        # image = tf.image.adjust_hue(image, hue_delta)
        # image = tf.image.adjust_saturation(image, sat_delta)
        return image

    def preprocess(self, image):
        if self.subset in 'train':
            image = self._image_augmentation(image, seed=42)

        image = tf.image.resize_area(image, (224, 224))
        image = tf.divide(tf.subtract(image, 127.), 127.)

        return image




