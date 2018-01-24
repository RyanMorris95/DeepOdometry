import tensorflow as tf
from tensorflow.contrib import slim

keras = tf.keras


def build_inception(inputs, is_training, scope='inception_v1'):
    with tf.variable_scope(scope):
        inception_v1 = slim.nets.inception_v1
        predictions, _ = inception_v1.inception_v1(inputs, num_classes=7, prediction_fn=None)
    return predictions


def build_sfmlearner(inputs, is_training, outputs, scope='sfmlearner'):
    slim = tf.contrib.slim
    with tf.variable_scope('pose_exp_net') as sc:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            cnv1 = slim.conv2d(inputs, 16, [7, 7], stride=2, scope='cnv1')
            cnv2 = slim.conv2d(cnv1, 32, [5, 5], stride=2, scope='cnv2')
            cnv3 = slim.conv2d(cnv2, 64, [3, 3], stride=2, scope='cnv3')
            cnv4 = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
            cnv5 = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
            # Pose specific layers
            with tf.variable_scope('pose'):
                cnv6 = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
                cnv7 = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                flat = slim.flatten(cnv7)
                fc1 = slim.fully_connected(flat, 512, activation_fn=tf.nn.relu)
                pose = slim.fully_connected(fc1, outputs)
    return pose


def build_resnet34(inputs, is_training, num_output, scope='resnet34'):
    import resnet
    with tf.variable_scope(scope):
        network = resnet.imagenet_resnet_v2(34, num_output)
        predictions = network(inputs, is_training=is_training)
    return predictions
