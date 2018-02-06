import tensorflow as tf
from tensorflow.contrib import slim

keras = tf.keras


# Thanks, https://github.com/tensorflow/tensorflow/issues/4079
def LeakyReLU(x, leak=0.1, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1.0 + leak)
        f2 = 0.5 * (1.0 - leak)
        return f1 * x + f2 * abs(x)


def average_endpoint_error(labels, predictions):
    """
    Given labels and predictions of size (N, H, W, 2), calculates average endpoint error:
        sqrt[sum_across_channels{(X - Y)^2}]
    """
    num_samples = predictions.shape.as_list()[0]
    with tf.name_scope(None, "average_endpoint_error", (predictions, labels)) as scope:
        predictions = tf.to_float(predictions)
        labels = tf.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

        squared_difference = tf.square(tf.subtract(predictions, labels))
        # sum across channels: sum[(X - Y)^2] -> N, H, W, 1
        loss = tf.reduce_sum(squared_difference, 3, keep_dims=True)
        loss = tf.sqrt(loss)
        return tf.reduce_sum(loss) / num_samples


def pad(tensor, num=1):
    """
    Pads the given tensor along the height and width dimensions with `num` 0s on each side
    """
    return tf.pad(tensor, [[0, 0], [num, num], [num, num], [0, 0]], "CONSTANT")


def antipad(tensor, num=1):
    """
    Performs a crop. "padding" for a deconvolutional layer (conv2d tranpose) removes
    padding from the output rather than adding it to the input.
    """
    batch, h, w, c = tensor.shape.as_list()
    return tf.slice(tensor, begin=[0, num, num, 0], size=[batch, h - 2 * num, w - 2 * num, c])


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
                # flat = slim.flatten(cnv7)
                # fc1 = slim.fully_connected(flat, 512, activation_fn=tf.nn.relu)
                #pose = slim.fully_connected(fc1, outputs)
    return cnv7


slim = tf.contrib.slim
def build_flownet(inputs, is_training, scope='flownet'):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        # Only backprop this network if trainable
                        trainable=False,
                        # He (aka MSRA) weight initialization
                        weights_initializer=slim.variance_scaling_initializer(),
                        activation_fn=LeakyReLU,
                        # We will do our own padding to match the original Caffe code
                        padding='VALID'):
        weights_regularizer = slim.l2_regularizer(0.0004)
        with slim.arg_scope([slim.conv2d], weights_regularizer=weights_regularizer):
            with slim.arg_scope([slim.conv2d], stride=2):
                conv_1 = slim.conv2d(pad(inputs, 3), 64, 7, scope='conv1')
                conv_2 = slim.conv2d(pad(conv_1, 2), 128, 5, scope='conv2')
                conv_3 = slim.conv2d(pad(conv_2, 2), 256, 5, scope='conv3')

            conv3_1 = slim.conv2d(pad(conv_3), 256, 3, scope='conv3_1')
            with slim.arg_scope([slim.conv2d], num_outputs=512, kernel_size=3):
                conv4 = slim.conv2d(pad(conv3_1), stride=2, scope='conv4')
                conv4_1 = slim.conv2d(pad(conv4), scope='conv4_1')
                conv5 = slim.conv2d(pad(conv4_1), stride=2, scope='conv5')
                conv5_1 = slim.conv2d(pad(conv5), scope='conv5_1')
            conv6 = slim.conv2d(pad(conv5_1), 1024, 3, stride=2, scope='conv6')
            #conv6_1 = slim.conv2d(pad(conv6), 1024, 3, scope='conv6_1')

    return conv6


from tflearn.layers import core, conv, recurrent
def build_deepvo(inputs, is_training, seq_length, height, width, scope='deepvo'):
    # print (inputs)
    # inputs_list = tf.unstack(inputs)
    # for inputs in inputs_list:
    #     pred = build_flownet(inputs, is_training=is_training)
    #
    with tf.variable_scope('timedistruted_cnn', reuse=tf.AUTO_REUSE):
        print (inputs)
        inputs = tf.reshape(inputs, (1, seq_length, height, width, 2))
        net = core.input_data(shape=(None, seq_length, height, width, 2), placeholder=inputs)
        net = core.time_distributed(net, build_flownet, [inputs])
        # net = core.time_distributed(net, conv.conv_2d, [32, 7, 2, 'same', 'relu'])
        # net = core.time_distributed(net, conv.conv_2d, [64, 5, 2, 'same', 'relu'])
        # net = core.time_distributed(net, conv.conv_2d, [128, 3, 2, 'same', 'relu'])
        # net = core.time_distributed(net, conv.conv_2d, [256, 3, 2, 'same', 'relu'])
        # net = core.time_distributed(net, conv.conv_2d, [256, 3, 2, 'same', 'relu'])
        # net = core.time_distributed(net, conv.conv_2d, [256, 3, 2, 'same', 'relu'])
        net = core.time_distributed(net, conv.global_max_pool)
    with tf.variable_scope('rnn'):
        net = recurrent.lstm(net, n_units=124, activation=tf.nn.relu, return_seq=True, name='lstm1')
        net = recurrent.lstm(net, n_units=124, activation=tf.nn.relu, return_seq=True, name='lstm2')

        net = core.time_distributed(net, core.fully_connected, [128, 'relu'])
        net = core.time_distributed(net, core.fully_connected, [12])
        pose, uncertainty = tf.split(net, 2, axis=2)
        pose = tf.cast(pose, tf.float64)

        print ("pose output shape")
        print (pose)

        #pose = core.fully_connected(net, activation=tf.nn.relu, n_units=128)
        #pose = core.fully_connected(net, n_units=6)
        #pose = tf.cast(pose, tf.float64)
        #print (pose)
        #print (tf.trainable_variables())

    return pose, uncertainty


def build_resnet34(inputs, is_training, num_output, scope='resnet34'):
    """
    Building the mapnet architecture from https://arxiv.org/pdf/1712.03342.pdf
    :param inputs:
    :param is_training:
    :param num_output:
    :param scope:
    :return:
    """
    import resnet
    with tf.variable_scope(scope):
        network = resnet.imagenet_resnet_v2(34, num_output)
        predictions = network(inputs, is_training=is_training)
    return predictions
