import os
import multiprocessing as mp

import tensorflow as tf


HEIGHT = 32
WIDTH = 32
DEPTH = 3
IMAGE_BYTES = HEIGHT * WIDTH * DEPTH
LABEL_BYTES = 2
TRAIN_SIZE = 50000
VAL_SIZE = 10000
NUM_CLASSES = 100


def get_data(data_dir, is_training, batch_size, buffer_size=5000):
    """Create a dataset of preprocessed and batched CIFAR-100 images and labels."""
    file_names = get_filenames(data_dir, is_training)
    dataset = tf.data.FixedLengthRecordDataset(file_names, IMAGE_BYTES + LABEL_BYTES)

    dataset = dataset.map(parse_record, num_parallel_calls=mp.cpu_count())
    dataset = dataset.prefetch(4 * batch_size)

    if is_training:
        dataset = dataset.map(preprocess_for_train, num_parallel_calls=mp.cpu_count())
    dataset = dataset.prefetch(4 * batch_size)

    dataset = dataset.map(lambda image, label: (tf.image.per_image_standardization(image), label),
                          num_parallel_calls=mp.cpu_count())
    dataset = dataset.prefetch(4 * batch_size)

    if is_training:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)

    return dataset


def get_filenames(data_dir, is_training):
    """Return the filename for the selected subset."""
    if is_training:
        return os.path.join(data_dir, 'train.bin')
    else:
        return os.path.join(data_dir, 'test.bin')


def parse_record(value):
    """Parse a CIFAR-100 record from value to an (image, label) pair."""
    raw_record = tf.decode_raw(value, tf.uint8)

    # Get only fine label and ignore coarse label
    label = tf.cast(raw_record[1], tf.int64)
    image_depth_height_width = tf.reshape(raw_record[LABEL_BYTES:IMAGE_BYTES + LABEL_BYTES], [DEPTH, HEIGHT, WIDTH])
    image = tf.cast(tf.transpose(image_depth_height_width, [1, 2, 0]), tf.float32)

    return image, label


def preprocess_for_train(image, label):
    """Preprocess a single training image of layout [height, width, depth]."""
    image = tf.image.resize_image_with_crop_or_pad(image, HEIGHT + 4, WIDTH + 4)
    image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])

    image = tf.image.random_flip_left_right(image)

    return image, label
