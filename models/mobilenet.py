import tensorflow as tf
from tensorflow.contrib import slim

from nets.mobilenet_v1 import mobilenet_v1, mobilenet_v1_arg_scope

FLAGS = tf.app.flags.FLAGS


class MobileNet:
    def __init__(self, inputs, num_classes, is_training, labels):
        self.create_model(inputs, num_classes, is_training)
        self.create_metrics(labels)
        self.create_loss(labels)
        self.create_train_op()
        self.create_summaries()

    def create_model(self, inputs, num_classes, is_training):
        with slim.arg_scope(mobilenet_v1_arg_scope(is_training)):
            self.logits, self.end_points = mobilenet_v1(inputs, num_classes, FLAGS.dropout_keep_prob, is_training,
                                                        depth_multiplier=FLAGS.depth_multiplier)

        for var in tf.model_variables():
            if 'weights' in var.op.name:
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, var)

    def create_metrics(self, labels):
        predictions = tf.argmax(self.logits, 1)
        self.accuracy, self.accuracy_update = tf.metrics.accuracy(labels, predictions, name='accuracy')
        vars_to_reset = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='accuracy')
        self.accuracy_reset = tf.variables_initializer(vars_to_reset, 'accuracy_reset')

    def create_loss(self, labels):
        self.cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(labels, self.logits,
                                                                         scope='cross_entropy_loss')
        self.total_loss = tf.losses.get_total_loss()

    def create_train_op(self):
        global_step = tf.train.get_or_create_global_step()
        self.learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_steps,
                                                        FLAGS.decay_rate, name='learning_rate')
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.gradients = optimizer.compute_gradients(self.total_loss)
            self.train_op = optimizer.apply_gradients(self.gradients, global_step)

    def create_summaries(self):
        tf.summary.scalar('metrics/accuracy', self.accuracy, collections=['metrics'])
        tf.summary.scalar('losses/cross_entropy_loss', self.cross_entropy_loss, collections=['train'])
        tf.summary.scalar('losses/total_loss', self.total_loss, collections=['train'])
        tf.summary.scalar('training/learning_rate', self.learning_rate, collections=['train'])
        for weight in tf.get_collection(tf.GraphKeys.WEIGHTS):
            tf.summary.histogram('weights/' + weight.op.name, weight, collections=['train'])
        for name, end_point in self.end_points.items():
            tf.summary.histogram('activations/' + name, end_point, collections=['train'])
        for grad, var in self.gradients:
            if grad is not None:
                tf.summary.histogram('gradients/' + var.op.name, grad, collections=['train'])

        self.train_summary_op = tf.summary.merge_all(key='train')
        self.metrics_summary_op = tf.summary.merge_all(key='metrics')
