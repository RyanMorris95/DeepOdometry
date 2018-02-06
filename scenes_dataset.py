import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, sys

HEIGHT = 480
WIDTH = 640
DEPTH = 3


class ScenesDataset(object):
    def __init__(self, data_dir, target_height=HEIGHT, target_width=WIDTH, mean=127., std=127.,
                 subset='train', use_distortion=True, abs_and_rel=False):
        self.data_dir = data_dir
        self.subset = subset
        self.use_distortion = use_distortion
        self.abs_and_rel = abs_and_rel
        self.target_height = target_height
        self.target_width = target_width
        self.mean = mean
        self.std = std

    def get_filenames(self):
        if self.subset in ['train', 'validation', 'test']:
            return [os.path.join(self.data_dir, self.subset + '_7scenes.tfrecord')]
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

        image = tf.reshape(image, [HEIGHT, WIDTH, 3])
        image = tf.image.convert_image_dtype(image, tf.float32)  # IMAGE BETWEEN 1 and 0

        label_abs = tf.decode_raw(parsed["pose_abs"], tf.float64)
        label_abs = tf.reshape(label_abs, [6])

        label_rel = tf.decode_raw(parsed["pose_rel"], tf.float64)
        label_rel = tf.reshape(label_rel, [6])

        return image, label_rel, label_abs

    def make_batch(self, batch_size):
        filenames = self.get_filenames()
        print (filenames)

        dataset = tf.data.TFRecordDataset(filenames)
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

        # image = tf.image.random_brightness(image, max_delta=32. / 255.)
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        # image = tf.image.random_hue(image, max_delta=0.2)
        # image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        # image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    def preprocess(self, image):
        if self.subset in 'train':
            image = self._image_augmentation(image, seed=42)

        image = tf.divide(tf.subtract(image, self.mean), self.std)
        image = tf.image.resize_area(image, (self.target_height, self.target_width))

        return image


if __name__ == '__main__':
    import cv2

    MEAN = [0.5001157, 0.4440133, 0.43354344]
    STD = [0.22229797, 0.23737237, 0.2185392]

    scenes_dataset = ScenesDataset('/media/ryan/E4DE46CCDE4696A8/7-scenes/', subset='test',
                                   mean=MEAN, std=STD, target_height=251, target_width=341)
    image_batch, image_labels = scenes_dataset.make_batch(1)
    means = []
    stds = []

    with tf.Session() as sess:
        for i in range(10):
            images, labels = sess.run([image_batch, image_labels])
            img = images[0].reshape((images[0].shape[0], images[0].shape[1], 3)).copy()

            quat = labels[0, 3:7]
            u, v = quat[0], quat[1:4]

            plt.imshow(img)
            plt.show()
            log_q = np.dot(v/np.linalg.norm(v), np.arccos(u))

    ### FOR FINDING THE MEAN AND STD OF TRAINING SET
    # with tf.Session() as sess:
    #     for i in range(25609):
    #         if i % 1000 == 0:
    #             print (i)
    #         images, labels = sess.run([image_batch, image_labels])
    #         img = images[0].reshape((images[0].shape[0], images[0].shape[1], 3)).copy()
    #         mean = img.reshape(-1, img.shape[-1]).mean(0)
    #         _std = img.reshape(-1, img.shape[-1]).std(0)
    #
    #         means.append(mean.copy())
    #         stds.append(_std.copy())
    #
    # means = np.array(means)
    # stds = np.array(stds)
    #
    # import pickle as p
    # with open('tmp.p', 'wb') as fp:
    #     p.dump([means, stds], fp)
    #
    # final_mean = means.reshape(-1, means.shape[-1]).mean(0)
    # final_stds = stds.reshape(-1, stds.shape[-1]).mean(0)
    # print (final_mean, final_stds)