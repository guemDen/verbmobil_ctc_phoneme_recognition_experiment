import tensorflow as tf
import numpy as np
import time


def model(num_layers, num_units):

    cells = []
    for _ in range(num_layers):
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        cells.append(cell)
    stack = tf.nn.rnn_cell.MultiRNNCell(cells)

    return stack


def inference(x, seq_len, W, b, stack, num_hidden, num_classes):

    outputs, _ = tf.nn.dynamic_rnn(stack, x, seq_len, dtype=tf.float32)

    shape = tf.shape(x)
    batch_size, max_timesteps = shape[0], shape[1]

    outputs = tf.reshape(outputs, [-1, num_hidden])

    logits = tf.matmul(outputs, W) + b
    logits = tf.reshape(logits, [batch_size, -1, num_classes])
    logits = tf.transpose(logits, (1, 0, 2), name=logits)

    return logits


def loss(y, logits, seq_len):

    return tf.nn.ctc_loss(y, logits, seq_len)


def cost(loss):

    return tf.reduce_mean(loss)


def optimize(learning_rate, momentum, cost):

    return tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cost)


def decode(logits, seq_len):

    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

    return decoded, log_prob


def label_error_rate(decoded, y):

    return tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), y))


def run_model(x_train, y_train, num_features, num_examples, num_epochs, batch_size,
    num_batches_per_epoch, learning_rate, momentum, num_layers, num_hidden, num_classes):

    x_train = tf.placeholder(tf.float32, [None, None, num_features])
    y_train = tf.sparse_placeholder(tf.int32)
    seq_len = tf.placeholder(tf.int32, [None])
    W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[num_classes]))

    stack = model(num_layers, num_hidden)

    logits = inference(x_train, seq_len, W, b, stack, num_hidden, num_classes)

    loss_ = loss(y_train, logits, seq_len)

    cost_ = cost(loss_)

    optimizer = optimize(learning_rate, momentum, cost_)

    decoded, log_prob = decode(logits, seq_len)

    ler = label_error_rate(decoded, y_train)

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        shuffled_indexes = np.random.permutation(num_examples)
        x_train = x_train[shuffled_indexes]
        y_train = y_train[shuffled_indexes]

        for curr_epoch in range(num_epochs):
            train_cost = train_ler = 0
            start = time.time()

            for batch in range(num_batches_per_epoch):

                indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]

                batch_x_train = x_train[indexes]

                batch_y_train = y_train[indexes]

                feed = {x_train: batch_x_train,
                        y_train: batch_y_train,
                        seq_len: None}

                batch_cost, _ = session.run([cost_, optimizer], feed)
                train_cost += batch_cost*batch_size
                train_ler += session.run(ler, feed_dict=feed)*batch_size

        train_cost /= num_examples
        train_ler /= num_examples

        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler, time.time() - start))