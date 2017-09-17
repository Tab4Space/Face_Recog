import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
KEEP_PROB = 0.7
LEARNING_RATE = 1e-3
TRAIN_EPOCH = 15
BATCH_SIZE = 10
NUM_THREADS = 4
CAPACITY = 5000
MIN_AFTER_DEQUEUE = 100
NUM_CLASSES = 3
FILTER_SIZE = 2
POOLING_SIZE = 2
MODEL_NAME = './tmp/model-{}-{}-{}'.format(TRAIN_EPOCH, LEARNING_RATE, BATCH_SIZE)

def generateModel():
    csv_file = tf.train.string_input_producer(['/home/bhappy/Face_Recog/label.csv'], name='filenamequeue', shuffle=True)
    csv_reader = tf.TextLineReader()
    _, line = csv_reader.read(csv_file)

    imagefile, label_decoded = tf.decode_csv(line, record_defaults=[[""], [""]])
    image_decoded = tf.image.decode_png(tf.read_file(imagefile), channels=1)

    image_cast = tf.cast(image_decoded, tf.float32)
    image = tf.reshape(image_cast, [IMAGE_WIDTH, IMAGE_HEIGHT, 1])


    # similary tf.placeholder
    # Training batch set
    image_batch, label_batch = tf.train.shuffle_batch([image, label_decoded], batch_size=BATCH_SIZE, num_threads=NUM_THREADS, capacity=CAPACITY, min_after_dequeue=MIN_AFTER_DEQUEUE)

    X = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
    Y = tf.placeholder(tf.int32, [BATCH_SIZE, 1])
    Y_one_hot = tf.one_hot(Y, NUM_CLASSES)
    Y_one_hot = tf.reshape(Y_one_hot, [-1, NUM_CLASSES])


    ### Graph part
    filter1 = tf.get_variable('filter1', shape=[FILTER_SIZE, FILTER_SIZE, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
    L1 = tf.nn.conv2d(X, filter1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')


    filter2= tf.get_variable('filter2', shape=[FILTER_SIZE, FILTER_SIZE, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
    L2 = tf.nn.conv2d(L1, filter2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')


    filter3 = tf.get_variable('filter3', shape=[FILTER_SIZE, FILTER_SIZE, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    L3 = tf.nn.conv2d(L2, filter3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')


    filter4 = tf.get_variable('filter4', shape=[FILTER_SIZE, FILTER_SIZE, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
    L4 = tf.nn.conv2d(L3, filter4, strides=[1, 1, 1, 1], padding='SAME')
    L4 = tf.nn.relu(L4)
    L4 = tf.nn.max_pool(L4, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')


    filter5 = tf.get_variable('filter5', shape=[FILTER_SIZE, FILTER_SIZE, 256, 512], initializer=tf.contrib.layers.xavier_initializer())
    L5 = tf.nn.conv2d(L4, filter5, strides=[1, 1, 1, 1], padding='SAME')
    L5 = tf.nn.relu(L5)
    L5 = tf.nn.max_pool(L5, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')


    filter6 = tf.get_variable('filter6', shape=[FILTER_SIZE, FILTER_SIZE, 512, 1024], initializer=tf.contrib.layers.xavier_initializer())
    L6 = tf.nn.conv2d(L5, filter6, strides=[1, 1, 1, 1], padding='SAME')
    L6 = tf.nn.relu(L6)
    L6 = tf.nn.max_pool(L6, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')
    L6 = tf.reshape(L6, [-1, 1*1*1024])


    flat_W1 = tf.get_variable("flat_W", shape=[1*1*1024, 3], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([3]))
    logits = tf.matmul(L6, flat_W1) + b1


    saver = tf.train.Saver()


    print("=========================================================================================")
    print("logits: ", logits)
    print("Y one hot: ", Y_one_hot)
    print("=========================================================================================")


    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)


    with tf.Session() as sess:
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())

        for epoch in range(TRAIN_EPOCH):
            avg_cost = 0
            total_batch = int(1500/BATCH_SIZE)

            for i in range(total_batch):
                batch_x, batch_y = sess.run([image_batch, label_batch])
                
                batch_y = batch_y.reshape(BATCH_SIZE, 1)
                
                cost_value, _ = sess.run([cost, optimizer], feed_dict={X: batch_x, Y: batch_y})
                avg_cost += cost_value / total_batch

                saver.save(sess, MODEL_NAME)

            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

        print('Learning Finished!')

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # print('Accuracy!!!!!!: ', sess.run(accuracy))