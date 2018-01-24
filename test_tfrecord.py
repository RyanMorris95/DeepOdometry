import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

width = 1920
height = 1080

def parser(record):
    keys_to_features = {
        "pose_rel": tf.FixedLenFeature((), tf.string, default_value=""),
        "pose_abs": tf.FixedLenFeature((), tf.string, default_value=""),
        "image": tf.FixedLenFeature((), tf.string, default_value=""),
    }

    parsed = tf.parse_single_example(record, keys_to_features)

    image = tf.image.decode_jpeg(parsed["image"])
    image = tf.reshape(image, [height, width, 3])
    image = tf.image.resize_image_with_crop_or_pad(image, 1020, 1020)
    #image = tf.image.resize_area(image, (224, 224))

    label_abs = tf.decode_raw(parsed["pose_abs"], tf.float64)
    label_abs = tf.reshape(label_abs, [7])

    label_rel = tf.decode_raw(parsed["pose_rel"], tf.float64)
    label_rel = tf.reshape(label_rel, [7])

    return image, label_rel, label_abs

if __name__ == '__main__':
    import cv2
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parser)
    dataset = dataset.batch(32)
    iterator = dataset.make_initializable_iterator()

    with tf.Session() as sess:
        # Initialize 'iterator' with training data
        training_filenames = ['/media/ryan/E4DE46CCDE4696A8/KingsCollege/train.tfrecord']

        sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
        images, labels_rel, labels_abs = iterator.get_next()
        images, labels_rel, labels_abs = sess.run([images, labels_rel, labels_abs])
        print (images[0].shape)
        img = images[8].reshape((images[0].shape[0], images[0].shape[1], 3))
        img = cv2.resize(img, (224, 224))

        plt.imshow(img)
        plt.show()