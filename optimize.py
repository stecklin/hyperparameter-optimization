import os
from datetime import datetime

import dlib
import tensorflow as tf

from datasets import cifar100
from models.mobilenet import MobileNet

# Data and logging
tf.app.flags.DEFINE_string('data_dir', None, 'Directory containing the training and validation data')
tf.app.flags.DEFINE_string('out_dir', None, 'Directory used to save summaries and checkpoints')
tf.app.flags.DEFINE_integer('log_steps', 5000, 'Number of steps after which summaries and checkpoints are saved')

# Optimization
tf.app.flags.DEFINE_integer('num_train_runs', 30, 'Number of times the optimization algorithms can train a network')

# Training
tf.app.flags.DEFINE_integer('max_steps', 100000, 'Number of steps performed per training run')
tf.app.flags.DEFINE_integer('batch_size', 96, 'Batch size used for training and validation dataset')
tf.app.flags.DEFINE_float('learning_rate', 0.045, 'Initial learning rate')
tf.app.flags.DEFINE_integer('decay_steps', 500, 'Number of steps after which the learning rate is decayed')
tf.app.flags.DEFINE_float('decay_rate', 0.98, 'Factor by which the learning rate is decayed')

# Model regularization
tf.app.flags.DEFINE_float('depth_multiplier', 1.0, 'Float multiplier for the number of channels in convolutions')
tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'Amount of weight decay used for regularization')
tf.app.flags.DEFINE_bool('regularize_depthwise', False, 'Whether or not apply regularization on depthwise layers')
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.999, 'Probability to keep neurons during training with dropout')

# Optimization goal
tf.app.flags.DEFINE_bool('optimize_learning_rate', True, 'If True, optimize learning rate, else model regularization')

FLAGS = tf.app.flags.FLAGS


def train(log_dir):
    """Train a network and return the best validation accuracy observed."""
    with tf.Graph().as_default():
        # Data loading
        train_dataset = cifar100.get_data(FLAGS.data_dir, is_training=True, batch_size=FLAGS.batch_size)
        val_dataset = cifar100.get_data(FLAGS.data_dir, is_training=False, batch_size=FLAGS.batch_size)
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        images, labels = iterator.get_next()
        train_iterator = train_dataset.make_one_shot_iterator()
        val_iterator = val_dataset.make_initializable_iterator()

        # Build model
        is_training = tf.placeholder(tf.bool, name='is_training')
        model = MobileNet(images, 100, is_training, labels)

        saver = tf.train.Saver()
        top_accuracy = 0

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            train_handle, val_handle = sess.run([train_iterator.string_handle(), val_iterator.string_handle()])

            train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)
            val_writer = tf.summary.FileWriter(os.path.join(log_dir, 'val'))

            # Training
            sess.run(model.accuracy_reset)
            for step in range(FLAGS.max_steps):
                if step % FLAGS.log_steps == 0:
                    _, _, summary = sess.run([model.train_op, model.accuracy_update, model.train_summary_op],
                                             feed_dict={is_training: True, handle: train_handle})
                    # Log train summaries
                    train_writer.add_summary(summary, global_step=step)
                    metrics_summary = sess.run(model.metrics_summary_op)
                    train_writer.add_summary(metrics_summary, global_step=step)
                    # Evaluate model
                    sess.run([val_iterator.initializer, model.accuracy_reset])
                    while True:
                        try:
                            sess.run(model.accuracy_update, feed_dict={is_training: False, handle: val_handle})
                        except tf.errors.OutOfRangeError:
                            # Finished pass over validation set
                            break
                    # Update top accuracy
                    current_accuracy = sess.run(model.accuracy)
                    top_accuracy = max(top_accuracy, current_accuracy)
                    # Log validation summaries
                    metrics_summary = sess.run(model.metrics_summary_op)
                    val_writer.add_summary(metrics_summary, global_step=step)

                    saver.save(sess, os.path.join(FLAGS.out_dir, 'model'), global_step=step)
                    sess.run(model.accuracy_reset)
                else:
                    sess.run([model.train_op, model.accuracy_update],
                             feed_dict={is_training: True, handle: train_handle})

            if step % FLAGS.log_steps != 0:
                # Final evaluation
                sess.run([val_iterator.initializer, model.accuracy_reset])
                while True:
                    try:
                        sess.run(model.accuracy_update, feed_dict={is_training: False, handle: val_handle})
                    except tf.errors.OutOfRangeError:
                        # Finished pass over validation set
                        break
                # Update top accuracy
                current_accuracy = sess.run(model.accuracy)
                top_accuracy = max(top_accuracy, current_accuracy)
                # Log validation summaries
                metrics_summary = sess.run(model.metrics_summary_op)
                val_writer.add_summary(metrics_summary, global_step=step)

                saver.save(sess, os.path.join(FLAGS.out_dir, 'model'), global_step=step)

            return top_accuracy


def hyperparameter_score(depth_multiplier, weight_decay, dropout_keep_prob):
    """Modify flags for hyperparameters to optimize, invoke training and return the score.
    Function called by the optimization algorithm."""
    # Create log dir
    sub_dir = '%.3f_%.5f_%.3f' % (depth_multiplier, weight_decay, dropout_keep_prob)
    log_dir = os.path.join(FLAGS.out_dir, sub_dir)
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)
    # Set flags
    FLAGS.depth_multiplier = depth_multiplier
    FLAGS.weight_decay = weight_decay
    FLAGS.dropout_keep_prob = dropout_keep_prob
    # Get score
    accuracy = train(log_dir)
    print('[%s] depth_multiplier = %.3f, weight_decay = %.5f, dropout_keep_prob = %.3f, accuracy = %.2f%%' %
          (datetime.now().strftime('%H:%M'), depth_multiplier, weight_decay, dropout_keep_prob, accuracy * 100))
    return float(accuracy)


def optimize():
    """Run optimization for depth multiplier, weight decay and dropout keep probability."""
    lower_bounds = [0.25, 0.00004, 0.5]
    upper_bounds = [1.0, 0.001, 1.0]
    result = dlib.find_max_global(hyperparameter_score, lower_bounds, upper_bounds, FLAGS.num_train_runs)
    print('Finished optimization. Best hyperparameters found: \n'
          'depth_multiplier = %.3f, weight_decay = %.5f, dropout_keep_prob = %.3f, accuracy = %.2f%%' %
          (result[0][0], result[0][1], result[0][2], result[1] * 100))


def main(_):
    optimize()


if __name__ == '__main__':
    tf.flags.mark_flags_as_required(['data_dir', 'out_dir'])
    tf.app.run()
