from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf


# Default parameter values
FEATURE_DIM = 2048
MAX_FRAMES = 250
NUM_READERS = 4
NUM_PREPROCESS_THREADS = 4
INPUT_QUEUE_MEMORY_FACTOR = 4



def parse_sequential_example_proto(example_serialized, feature_dim=None):
    """
    Parses a SequentialExample proto containing a sequence of CNN features.
    Args:
      example_serialized: scalar Tensor tf.string containing a serialized Example protocol buffer.
      feature_dim: dimension of the feature maps
    Returns:
      video_buffer: tensor with shape [n_frames, feature_dim]
      labels: list of tf.int64 containing the label for each frame
      sequence_length: tf.int64, number of non zero-padded frames
      filename: name of the video
    """
    if feature_dim is None:
        feature_dim = FEATURE_DIM

    # Context features in SequentialExample proto.
    context_features = {
        "sequence_length": tf.FixedLenFeature([], dtype=tf.int64),
        "filename": tf.FixedLenFeature([], dtype=tf.string),
        "text": tf.FixedLenFeature([], dtype=tf.string),
        "fps": tf.FixedLenFeature([], dtype=tf.float32)
    }

    sequence_features = {
        "features": tf.FixedLenSequenceFeature([feature_dim], dtype=tf.float32),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    # Parse the example
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=example_serialized,
        context_features=context_features,
        sequence_features=sequence_features
    )

    return sequence_parsed['features'], sequence_parsed['labels'], \
           context_parsed['sequence_length'], context_parsed['filename']


def batch_inputs(tf_record_pattern, batch_size, train, num_preprocess_threads=None, num_readers=None):
    """
    Creates the data loading pipeline.
    Args:
      tf_record_pattern: pattern for the input TFRecord data
      batch_size: integer
      train: boolean
      num_preprocess_threads: integer, total number of preprocessing threads
      num_readers: integer, number of parallel readers
    Returns:
      features: 4-D float Tensor of [batch_size, max_frames, feature_dim] with the CNN features
      labels: 2-D integer Tensor of [batch_size, max_frames].
      sequence_length: 1-D integer Tensor of [batch_size].
      filename: 1-D string Tensor of [batch_size].
    Raises:
      ValueError: if data is not found
    """

    with tf.name_scope('batch_processing'):
        data_files = tf.gfile.Glob(tf_record_pattern)
        if data_files is None:
            raise ValueError('No data files found for this dataset')

        # Create filename_queue
        if train:
            filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=16)
        else:
            filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=1, num_epochs=1)

        if num_preprocess_threads is None:
            num_preprocess_threads = NUM_PREPROCESS_THREADS

        if num_readers is None:
            num_readers = NUM_READERS

        if num_readers < 1:
            raise ValueError('Please make num_readers at least 1')

        # Approximate number of examples per shard.
        examples_per_shard = 191
        min_queue_examples = examples_per_shard * INPUT_QUEUE_MEMORY_FACTOR
        if train:
            examples_queue = tf.RandomShuffleQueue(
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples,
                dtypes=[tf.string])
        else:
            examples_queue = tf.FIFOQueue(
                capacity=examples_per_shard + 3 * batch_size,
                dtypes=[tf.string])

        # Create multiple readers to populate the queue of examples.
        if num_readers > 1:
            enqueue_ops = []
            for _ in range(num_readers):
                reader = tf.TFRecordReader()
                _, value = reader.read(filename_queue)
                enqueue_ops.append(examples_queue.enqueue([value]))

            tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            example_serialized = examples_queue.dequeue()
        else:
            reader = tf.TFRecordReader()
            _, example_serialized = reader.read(filename_queue)

        parsed_examples = []
        for thread_id in range(num_preprocess_threads):
            # Parse a serialized Example proto to extract the features and metadata.
            features, label, sequence_length, filename = parse_sequential_example_proto(example_serialized)
            features.set_shape([MAX_FRAMES, FEATURE_DIM])
            label.set_shape([MAX_FRAMES])
            sequence_length.set_shape([])
            sequence_length = tf.cast(sequence_length, tf.int32)
            parsed_examples.append([features, label, sequence_length, filename])

        batch_tensors = tf.train.batch_join(
            parsed_examples,
            batch_size=batch_size,
            capacity=2 * num_preprocess_threads * batch_size,
            allow_smaller_final_batch=True)

        videos = tf.reshape(tf.cast(batch_tensors[0], tf.float32), shape=[batch_size, MAX_FRAMES, FEATURE_DIM])
        labels = tf.reshape(batch_tensors[1], [batch_size, MAX_FRAMES])
        sequence_length = batch_tensors[2]
        filenames = batch_tensors[3]
        return videos, labels, sequence_length, filenames


def generate_batch(tf_record_pattern, batch_size, train=False, num_preprocess_threads=None):
    """
    Generate batches of data. Use this function as the inputs for a network.
    Args:
      tf_record_pattern: pattern for the input TFRecord data
      batch_size: integer, number of examples in batch
      train: boolean. Toggles data augmentation.
      num_preprocess_threads: integer, total number of preprocessing threads but
        None defaults to NUM_PREPROCESS_THREADS.
    Returns:
      features: 4-D float Tensor of [batch_size, max_frames, feature_dim] with the CNN features
      labels: 2-D integer Tensor of [batch_size, max_frames].
      sequence_length: 1-D integer Tensor of [batch_size].
      filename: 1-D string Tensor of [batch_size].
    """
    # Force all input processing onto CPU in order to reserve the GPU for the forward inference and back-propagation.
    with tf.device('/cpu:0'):
        batch_tensors = batch_inputs(
            tf_record_pattern, batch_size, train=train,
            num_preprocess_threads=num_preprocess_threads,
            num_readers=NUM_READERS)
    return batch_tensors
