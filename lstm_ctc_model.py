import tensorflow as tf
import numpy as np
import time

import utils
from vocab import vocab

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
    logits = tf.transpose(logits, (1, 0, 2), name=vocab.logits)

    return logits


def loss(y, logits, seq_len):

    return tf.nn.ctc_loss(y, logits, seq_len)


def cost(loss):

    return tf.reduce_mean(loss, name=vocab.cost)


def optimize(learning_rate, momentum, cost):

    return tf.train.MomentumOptimizer(learning_rate, momentum, name=vocab.optimizer).minimize(cost)


def decode(logits, seq_len):

    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

    return decoded, log_prob


def label_error_rate(decoded, y):

    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded, tf.int32), y), name=vocab.label_error_rate)

    return ler


def run_model(x_train, y_train, x_val, y_val, num_features, num_train_examples, num_val_examples, num_epochs, batch_size,
    num_batches_per_epoch, learning_rate, momentum, num_layers, num_hidden, num_classes):

    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, [None, None, num_features], name=vocab.x)
        y = tf.sparse_placeholder(tf.int32, name=vocab.y)
        seq_len = tf.placeholder(tf.int32, [None], name=vocab.seq_len)

        W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[num_classes]))

        stack = model(num_layers, num_hidden)

        logits = inference(x, seq_len, W, b, stack, num_hidden, num_classes)

        loss_ = loss(y, logits, seq_len)

        cost_ = cost(loss_)

        optimizer = optimize(learning_rate, momentum, cost_)

        decoded, log_prob = decode(logits, seq_len)

        ler = label_error_rate(decoded=decoded[0], y=y)

        #summaries
        #cost, ler for train set
        tf.summary.scalar("training_cost", cost_)
        tf.summary.scalar("training_label_error_rate", ler)
        summary_ops_train = tf.summary.merge_all()

        #cost, ler for val set
        tf.summary.scalar("validation_cost", cost_)
        tf.summary.scalar("validation_label_error_rate", ler)
        summary_ops_validation = tf.summary.merge_all()

    with tf.Session(graph=graph) as session:
        init = tf.global_variables_initializer()
        session.run(init)

        # path soll automatisch fuer jeweilige hyperparameter config erzeugt werden
        path_model_hyperparams = "model-num_layers=%d-num_hidden=%d-num_epochs=%d-batch_size=%d-learning_rate=%s" \
                                % (num_layers, num_hidden, num_epochs, batch_size, str(learning_rate))

        writer_train = tf.summary.FileWriter('./tensorboard_graphs/' + path_model_hyperparams + '/train', session.graph)
        writer_validation = tf.summary.FileWriter('./tensorboard_graphs/' + path_model_hyperparams + '/validation')

        shuffled_indexes = np.random.permutation(num_train_examples)
        x_train = x_train[shuffled_indexes]
        y_train = y_train[shuffled_indexes]

        for curr_epoch in range(num_epochs):
            train_cost = train_ler = 0
            start = time.time()

            for batch in range(num_batches_per_epoch):

                indexes = [i % num_train_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]

                batch_x_train = x_train[indexes]
                batch_x_train, batch_x_train_seq_len = utils.pad_sequences(batch_x_train)

                batch_y_train = utils.sparse_tuple_from(y_train[indexes])

                feed = {x: batch_x_train,
                        y: batch_y_train,
                        seq_len: batch_x_train_seq_len}

                batch_cost, _ = session.run([cost_, optimizer], feed)
                train_cost += batch_cost*batch_size
                train_ler += session.run(ler, feed_dict=feed)*batch_size
                summary_train = session.run(summary_ops_train, feed_dict=feed)

            train_cost /= num_train_examples
            train_ler /= num_train_examples

            writer_train.add_summary(summary_train, global_step=curr_epoch)

            val_indexes = [i for i in range(num_val_examples)]
            x_validation, x_val_seq_len = utils.pad_sequences(x_val[val_indexes])
            y_validation = utils.sparse_tuple_from(y_val[val_indexes])

            val_feed = {x: x_validation, y: y_validation, seq_len: x_val_seq_len}

            val_cost, val_ler = session.run([cost_, ler], feed_dict=val_feed)
            summary_validation = session.run(summary_ops_validation, feed_dict=val_feed)
            writer_validation.add_summary(summary_validation, global_step=curr_epoch)

            log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
            print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler, val_cost, val_ler, time.time() - start))