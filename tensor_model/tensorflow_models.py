import os
import tensorflow as tf
from django.conf import settings


def MDS_train_load(test_RSSI):
    def xavier_init(n_inputs, n_outputs, uniform=True):

        if uniform:
            init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
            return tf.random_uniform_initializer(-init_range, init_range)
        else:
            stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
            return tf.truncated_normal_initializer(stddev=stddev)

    X = tf.placeholder("float", [None, 40])
    Y = tf.placeholder("float", [None, 1250])

    W1 = tf.get_variable("W1", shape=[40, 400], initializer=xavier_init(40, 400))
    W2 = tf.get_variable("W2", shape=[400, 900], initializer=xavier_init(400, 900))
    W3 = tf.get_variable("W3", shape=[900, 1600], initializer=xavier_init(900, 1600))
    W4 = tf.get_variable("W4", shape=[1600, 1250], initializer=xavier_init(1600, 1250))

    b1 = tf.Variable(tf.random_normal([400]), name="b1")
    b2 = tf.Variable(tf.random_normal([900]), name="b2")
    b3 = tf.Variable(tf.random_normal([1600]), name="b3")
    b4 = tf.Variable(tf.random_normal([1250]), name="b4")

    dropout_rate = tf.placeholder("float")
    _L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
    L1 = tf.nn.dropout(_L1, dropout_rate)
    _L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))
    L2 = tf.nn.dropout(_L2, dropout_rate)
    _L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), b3))
    L3 = tf.nn.dropout(_L3, dropout_rate)

    hypothesis = tf.add(tf.matmul(L3, W4), b4)

    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, os.path.realpath(os.path.join(settings.BASE_DIR, 'tensor_model', '20170104', 'model1.pd')))
    #print("Model restored.")
    sample = sess.run(hypothesis, feed_dict={X: test_RSSI, dropout_rate: 1})
    #print(sample, sess.run(tf.argmax(sample, 1)))
    result = sess.run(tf.argmax(sample, 1))
    return result
