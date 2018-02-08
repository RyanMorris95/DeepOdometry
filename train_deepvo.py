import functools

import tensorflow as tf
import os
import architectures
import memory_saving_gradients

from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn import learn_runner
from kitti_dataset import KittiDataset, se3, compose_poses_tf
slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "/media/ryan/E4DE46CCDE4696A8/kitti_tfrecords/", "Dataset directory")
flags.DEFINE_string("optimizer", "adam", "Optimizer type")
flags.DEFINE_string("loss", "l2", "type of norm loss")
flags.DEFINE_string("save_dir", 'experiments/deepvo2', "Directory to save checkpoints")
flags.DEFINE_float("grad_clip", 60, "Gradient Norm Clipping")
flags.DEFINE_float("learning_rate", 0.002, "learing rate of optimizer")
flags.DEFINE_float("gpu_fraction", 1.0, "% usage of gpu")
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")
flags.DEFINE_integer("target_height", 64, "Target Image height")
flags.DEFINE_integer("target_width", 208, "Target Image width")
flags.DEFINE_integer("train_size", 15960, "Size of training set")
flags.DEFINE_integer("eval_size", 7230, "Size of eval set")
flags.DEFINE_integer("epochs", 100, "Number of runs through dataset")
flags.DEFINE_integer("seq_length", 20, "Sequence Length")
flags.DEFINE_bool("mem_save_grad", True, "Use memory saving gradients.")
FLAGS = flags.FLAGS

MEAN = [-0.14064756]
STD = [0.2851444]

TRAIN_STEPS = int(FLAGS.train_size / (FLAGS.batch_size*FLAGS.seq_length))
EVAL_STEPS = int(FLAGS.eval_size / (FLAGS.batch_size*FLAGS.seq_length))

print("Train steps: ", str(TRAIN_STEPS*FLAGS.epochs))
print("Eval steps: ", str(EVAL_STEPS))

if FLAGS.mem_save_grad:
    tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory


def model_fn(features, labels, mode, params):
    is_training = mode == ModeKeys.TRAIN
    global_step = tf.train.get_global_step()

    # predictions = architectures.build_sfmlearner(features, is_training=is_training, outputs=num_outputs)
    # predictions = architectures.build_resnet34(features, num_output=num_outputs, is_training=is_training)
    predictions, uncertainty = architectures.build_deepvo(features, is_training=is_training, seq_length=FLAGS.seq_length,
                                             height=FLAGS.target_height, width=FLAGS.target_width)

    # compose the relative pose predictions into abs poses
    with tf.variable_scope('compose_poses'):
        predictions = tf.reshape(predictions, (FLAGS.seq_length, 6))
        predictions = compose_poses_tf(predictions, FLAGS.seq_length)        
        predictions = tf.reshape(predictions, (FLAGS.seq_length, 6))


    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


    with tf.variable_scope('label_pred_sclicing'):
        pred_trans, pred_rot = tf.split(predictions, 2, axis=1)
        label_trans, label_rot = tf.split(labels, 2, axis=1)

    eval_metric_ops = {"rmse_trans": tf.metrics.root_mean_squared_error(tf.cast(label_trans, tf.float64),
                                                                  tf.cast(pred_trans, tf.float64)),
                        "rmse_rot": tf.metrics.root_mean_squared_error(tf.cast(label_rot, tf.float64),
                                                                  tf.cast(pred_rot, tf.float64))}
    loss = None
    train_op = None
    train_hooks = None

    if mode != ModeKeys.INFER:
        chk = '/home/ryan/DeepOdometry/FlowNetS/flownet-S.ckpt-0'
        variables = ['conv1', 'conv2', 'conv3', 'conv3_1', 'conv4', 'conv4_1', 'conv5', 'conv6', 'conv6_1']
        variables_to_restore = slim.get_variables_to_restore(variables)
        tf.train.init_from_checkpoint(chk, {v.name.split(':')[0]: v for v in variables_to_restore})

        sx = tf.Variable(0, dtype=tf.float32, name='optimal_trans_weight')
        sq = tf.Variable(-3, dtype=tf.float32, name='optimal_rot_weight')


        with tf.variable_scope('calc_loss'):
            if FLAGS.loss == 'l1':
                loss_trans = tf.reduce_mean(tf.losses.absolute_difference(label_trans, pred_trans), name='l1_trans')
                loss_rot = tf.reduce_mean(tf.losses.absolute_difference(label_rot, pred_rot), name='l1_rot')
            elif FLAGS.loss == 'l2':
                loss_trans = tf.reduce_mean(tf.squared_difference(label_trans, pred_trans), name='l2_trans')
                loss_rot = tf.reduce_mean(tf.squared_difference(label_rot, pred_rot), name='l2_rot')

            loss = tf.add(loss_trans, tf.multiply(loss_rot, 100), name='loss')

        lr = FLAGS.learning_rate

        # train_vars = None
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        train_vars.append(sx)
        train_vars.append(sq)

        if FLAGS.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        elif FLAGS.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer
        elif FLAGS.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer

        train_op = tf.contrib.layers.optimize_loss(loss=loss, global_step=tf.contrib.framework.get_global_step(),
                                                   optimizer=optimizer, learning_rate=lr, variables=train_vars,
                                                   clip_gradients=FLAGS.grad_clip)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('lr', lr)
        tf.summary.scalar('optimal_trans_weight', sx)
        tf.summary.scalar('optimal_rot_weight', sq)
        #ux, uy, uz, urx, ury, urz = tf.split(uncertainty, 6)
        #tf.summary.scalar('uncertainty', uncertainty)


        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)

        tensors_to_log = {'loss': loss}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10)
        train_hooks = [logging_hook]

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss,
                                      train_op=train_op, training_hooks=train_hooks,
                                      eval_metric_ops=eval_metric_ops)


def input_fn(data_dir, subset, batch_size, use_distortion_for_training=True):
    with tf.device('/cpu:0'):
        dataset = KittiDataset(data_dir, subset=subset,
                                target_height=FLAGS.target_height,
                                target_width=FLAGS.target_width,
                                mean=MEAN,
                                std=STD)
        image_batch, label_batch = dataset.make_batch(batch_size, FLAGS.seq_length)
        return image_batch, label_batch


def get_experiment_fn(data_dir):
    def _experiment_fn(run_config, hparams):
        train_input_fn = functools.partial(input_fn, data_dir, subset='train', batch_size=FLAGS.batch_size,
                                           use_distortion_for_training=True)
        eval_input_fn = functools.partial(input_fn, data_dir, subset='test', batch_size=FLAGS.batch_size)

        train_steps = TRAIN_STEPS * FLAGS.epochs
        eval_steps = EVAL_STEPS

        classifier = tf.estimator.Estimator(model_fn=model_fn, config=run_config, params=hparams)

        return tf.contrib.learn.Experiment(classifier, train_input_fn=train_input_fn,
                                           eval_input_fn=eval_input_fn, train_steps=train_steps,
                                           eval_steps=eval_steps)

    return _experiment_fn


def main(argv):
    """Run the training experiment."""
    params = tf.contrib.training.HParams(
        learning_rate=FLAGS.learning_rate,
        n_classes=6,
        train_steps=TRAIN_STEPS * FLAGS.epochs,
        min_eval_frequency=TRAIN_STEPS * 5  # every x epoch evaluate
    )

    # Session config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction

    # Set the run_config and the directory to save the model and stats
    run_config = tf.contrib.learn.RunConfig(session_config=config)
    run_config = run_config.replace(model_dir='./'+FLAGS.save_dir)

    learn_runner.run(
        experiment_fn=get_experiment_fn(FLAGS.dataset_dir),  # First-class function
        run_config=run_config,  # RunConfig
        schedule="train_and_evaluate",  # What to run
        hparams=params  # HParams
    )


if __name__ == '__main__':
    tf.app.run()
