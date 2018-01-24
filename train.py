import functools

import tensorflow as tf

import architectures
import memory_saving_gradients

# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory

from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn import learn_runner
from kings_dataset import KingsDataset

keras = tf.keras

optimizer = tf.train.AdagradOptimizer

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "/media/ryan/E4DE46CCDE4696A8/KingsCollege/", "Dataset directory")
flags.DEFINE_string("optimizer", "adam", "Optimizer type")
flags.DEFINE_string("loss", "l1", "type of norm loss")
flags.DEFINE_float("grad_clip", None, "Gradient Norm Clipping")
flags.DEFINE_float("learning_rate", 0.001, "learing rate of optimizer")
flags.DEFINE_float("gpu_usage", 1.0, "% usage of gpu")
flags.DEFINE_integer("batch_size", 8, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 224, "Image height")
flags.DEFINE_integer("img_width", 224, "Image width")
flags.DEFINE_integer("train_size", 1219, "Size of training set")
flags.DEFINE_integer("eval_size", 342, "Size of eval set")
flags.DEFINE_integer("epochs", 300, "Number of runs through dataset")
flags.DEFINE_bool("abs_and_rel", True, "Combine absolute and relative pose losses")
FLAGS = flags.FLAGS

TRAIN_STEPS = int(FLAGS.train_size / FLAGS.batch_size)
EVAL_STEPS = int(FLAGS.eval_size / FLAGS.batch_size)
print ("Train steps: ", str(TRAIN_STEPS))
print ("Eval steps: ", str(EVAL_STEPS))


def model_fn(features, labels, mode, params):
    tf.summary.image("images", features)

    is_training = mode == ModeKeys.TRAIN
    global_step = tf.train.get_global_step()

    predictions = architectures.build_resnet34(features, is_training=is_training)

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

        if FLAGS.loss == 'l1':
            loss_abs_trans = tf.reduce_mean(tf.losses.absolute_difference(label_abs_trans, pred_abs_trans))
            loss_abs_quat = tf.reduce_mean(tf.losses.absolute_difference(label_abs_quat, pred_abs_quat))
        elif FLAGS.loss == 'l2':
            loss_abs_trans = tf.losses.mean_squared_error(label_abs_trans, pred_abs_trans)
            loss_abs_quat = tf.losses.mean_squared_error(label_abs_quat, pred_abs_quat)

        mse_abs = tf.add(loss_abs_trans, loss_abs_quat)

        trans_loss_abs = tf.add(tf.scalar_mul(loss_abs_trans, tf.exp(-sx)), sx)
        rot_loss_abs = tf.add(tf.scalar_mul(loss_abs_quat, tf.exp(-sq)), sq)

        loss = tf.add(trans_loss_abs, rot_loss_abs, name='loss')

        if FLAGS.abs_and_rel:
            loss_rel_trans = tf.reduce_mean(tf.losses.absolute_difference(label_rel_trans, pred_rel_trans))
            loss_rel_quat = tf.reduce_mean(tf.losses.absolute_difference(label_rel_quat, pred_rel_quat))
            loss = tf.add(loss, tf.add(loss_rel_trans, loss_rel_quat))

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
        use_distortion = subset == 'train' and use_distortion_for_training
        dataset = KingsDataset(data_dir, subset, use_distortion)
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
    params = tf.contrib.training.HParams(
        learning_rate=FLAGS.learning_rate,
        n_classes=7,
        train_steps=TRAIN_STEPS * FLAGS.epochs,
        min_eval_frequency=500
    )

    # Set the run_config and the directory to save the model and stats
    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(model_dir='./resnet34')

    learn_runner.run(
        experiment_fn=get_experiment_fn(FLAGS.dataset_dir),  # First-class function
        run_config=run_config,  # RunConfig
        schedule="train_and_evaluate",  # What to run
        hparams=params  # HParams
    )


if __name__ == '__main__':
    tf.app.run()
