import functools

import tensorflow as tf
import os
import architectures
import memory_saving_gradients

from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn import learn_runner
from kings_dataset import KingsDataset
from scenes_dataset import ScenesDataset

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "/media/ryan/E4DE46CCDE4696A8/KingsCollege/", "Dataset directory")
flags.DEFINE_string("optimizer", "adam", "Optimizer type")
flags.DEFINE_string("loss", "l1", "type of norm loss")
flags.DEFINE_string("save_dir", 'experiments/current_model', "Directory to save checkpoints")
flags.DEFINE_float("grad_clip", None, "Gradient Norm Clipping")
flags.DEFINE_float("learning_rate", 0.001, "learing rate of optimizer")
flags.DEFINE_float("gpu_fraction", 1.0, "% usage of gpu")
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("target_height", 256, "Target Image height")
flags.DEFINE_integer("target_width", 341, "Target Image width")
flags.DEFINE_integer("train_size", 25609, "Size of training set")
flags.DEFINE_integer("eval_size", 16000, "Size of eval set")
flags.DEFINE_integer("epochs", 300, "Number of runs through dataset")
flags.DEFINE_bool("abs_and_rel", True, "Combine absolute and relative pose losses")
flags.DEFINE_bool("mem_save_grad", True, "Use memory saving gradients.")
FLAGS = flags.FLAGS

MEAN = [0.5001157, 0.4440133, 0.43354344]
STD = [0.22229797, 0.23737237, 0.2185392]

TRAIN_STEPS = int(FLAGS.train_size / FLAGS.batch_size)
EVAL_STEPS = int(FLAGS.eval_size / FLAGS.batch_size)
print("Train steps: ", str(TRAIN_STEPS*FLAGS.epochs))
print("Eval steps: ", str(EVAL_STEPS))

if FLAGS.mem_save_grad:
    print ("Using OpenAI gradient checkpointing!")
    tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory


def model_fn(features, labels, mode, params):
    tf.summary.image("images", features)

    is_training = mode == ModeKeys.TRAIN
    global_step = tf.train.get_global_step()

    num_outputs = 6
    if FLAGS.abs_and_rel:
        num_outputs = 12

    predictions = architectures.build_resnet34(features, num_output=num_outputs, is_training=is_training)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    eval_metric_ops = {"rmse": tf.metrics.root_mean_squared_error(tf.cast(labels, tf.float64),
                                                                  tf.cast(predictions, tf.float64))}
    loss = None
    train_op = None
    train_hooks = None

    if mode != ModeKeys.INFER:
        sx = tf.Variable(0, dtype=tf.float32, name='optimal_trans_weight')
        sq = tf.Variable(-3, dtype=tf.float32, name='optimal_rot_weight')

        with tf.variable_scope('label_pred_sclicing'):
            predictions_rel, predictions_abs = tf.split(predictions, 2, axis=1)
            pred_abs_trans = tf.slice(predictions_abs, [0, 0], [-1, 3])
            pred_abs_quat = tf.slice(predictions_abs, [0, 3], [-1, 4])
            pred_rel_trans = tf.slice(predictions_rel, [0, 0], [-1, 3])
            pred_rel_quat = tf.slice(predictions_rel, [0, 3], [-1, 4])

            labels_rel, labels_abs = tf.split(labels, 2, axis=1)
            label_abs_trans = tf.slice(labels_abs, [0, 0], [-1, 3])
            label_abs_quat = tf.slice(labels_abs, [0, 3], [-1, 4])
            label_rel_trans = tf.slice(labels_rel, [0, 0], [-1, 3])
            label_rel_quat = tf.slice(labels_rel, [0, 3], [-1, 4])

        with tf.variable_scope('loss_abs'):
            if FLAGS.loss == 'l1':
                loss_abs_trans = tf.reduce_sum(tf.losses.absolute_difference(label_abs_trans, pred_abs_trans))
                loss_abs_quat = tf.reduce_sum(tf.losses.absolute_difference(label_abs_quat, pred_abs_quat))
            elif FLAGS.loss == 'l2':
                loss_abs_trans = tf.losses.mean_squared_error(label_abs_trans, pred_abs_trans)
                loss_abs_quat = tf.losses.mean_squared_error(label_abs_quat, pred_abs_quat)

            mse_abs = tf.add(loss_abs_trans, loss_abs_quat)

            trans_loss_abs = tf.add(tf.scalar_mul(loss_abs_trans, tf.exp(-sx)), sx)
            rot_loss_abs = tf.add(tf.scalar_mul(loss_abs_quat, tf.exp(-sq)), sq)

            loss = tf.add(trans_loss_abs, rot_loss_abs, name='loss_abs')

        if FLAGS.abs_and_rel:
            with tf.variable_scope('loss_rel'):
                loss_rel_trans = tf.reduce_sum(tf.losses.absolute_difference(label_rel_trans, pred_rel_trans))
                loss_rel_quat = tf.reduce_sum(tf.losses.absolute_difference(label_rel_quat, pred_rel_quat))
                loss_rel = tf.add(loss_rel_quat, loss_rel_trans)
            loss = tf.add(loss, loss_rel, name='loss_absrel')
            tf.summary.scalar('loss_rel', loss_rel)

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
        tf.summary.scalar('mse_abs', mse_abs)
        tf.summary.scalar('lr', lr)
        tf.summary.scalar('optimal_trans_weight', sx)
        tf.summary.scalar('optimal_rot_weight', sq)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)

        tensors_to_log = {'loss': loss, 'mse_abs': mse_abs}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
        train_hooks = [logging_hook]

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss,
                                      train_op=train_op, training_hooks=train_hooks,
                                      eval_metric_ops=eval_metric_ops)


def input_fn(data_dir, subset, batch_size, use_distortion_for_training=True):
    with tf.device('/cpu:0'):
        dataset = ScenesDataset(data_dir, subset=subset,
                                abs_and_rel=FLAGS.abs_and_rel,
                                target_height=FLAGS.target_height,
                                target_width=FLAGS.target_width,
                                mean=MEAN,
                                std=STD)
        image_batch, label_batch = dataset.make_batch(batch_size)
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
    # Define model parameters
    if FLAGS.abs_and_rel:
        num_classes = 12
    else:
        num_classes = 6

    params = tf.contrib.training.HParams(
        learning_rate=FLAGS.learning_rate,
        n_classes=num_classes,
        train_steps=TRAIN_STEPS * FLAGS.epochs,
        min_eval_frequency=TRAIN_STEPS * 10  # every 10 epochs evaluate
    )

    # Session config
    config = tf.ConfigProto()
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
