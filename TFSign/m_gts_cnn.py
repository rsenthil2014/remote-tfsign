import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn.cross_validation import train_test_split
import numpy as np
import math
import os
import time
import pickle
#matplotlib inline

#parameters
IMG_SIZE = 32 # IMG_SIZE x IMG_SIZE
GRAYSCALE = False # convert image to gray scale?
NUM_CHANNELS = 1 if GRAYSCALE else 3
NUM_CLASSES = 43

#Model parameters
LR = 5e-3 # Learning rate
KEEP_PROB = 0.5 #drop out for training
OPT = tf.train.GradientDescentOptimizer(learning_rate = LR)

#Training process
RESTORE = False
RESUME = True
NUM_EPOCH =40
BATCH_SIZE = 128
BATCH_SIZE_INF = 2048 # For calculating accuracy
VALIDATION_SIZE = 0.2 # fraction to be used as validation set
SAVE_MODEL = True # To save trained model to disk
MODEL_SAVE_PATH = 'c:/tmp/GTS/model.ckpt'

###########################################
# Helper functions
###########################################

# Load pickled data
training_file = 'c:/tmp/GTS/train.p'
testing_file = 'c:/tmp/GTS/test.p'

#y_ = tf.placeholder(tf.float32, [None , NUM_CLASSES])

'''
def load_data():
    # Load pickled data
    training_file = 'c:/tmp/GTS/train.p'
    testing_file = 'c:/tmp/GTS/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
    
    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']

    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=VALIDATION_SIZE)
'''
def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    

def rgb2gray(rgb):    
    # Convert RGB images to Grayscale
    return np.dot(rgb[...,:3],[0.299,0.587,0.114])

    
def preprocess_data(X,y):
    # preprocess data 
    if GRAYSCALE:
        X = rgb2gray(X)

    # Make all image values fall with in range of -1 to 1
    X= X.astype('float32')
    X = (X-128.)/128.

    #convert the labels from numeric to one hot encoded
    y_onehot = np.zeros((y.shape[0], NUM_CLASSES))
    for i, onehot_label in enumerate(y_onehot):
        onehot_label[y[i]]=1.
    y = y_onehot

    return X,y

def next_batch(X, y, batch_size, augment_data):

    #provide data batch wise
    start_idx = 0
    while start_idx < X.shape[0]:
        images = X[start_idx : start_idx + batch_size]
        labels = y[start_idx : start_idx + batch_size]
        yield(np.array(images), np.array(labels))
        # Yield will make sure the continuty of the batch elements 
        start_idx += batch_size

def calculate_accuracy(data_gen, data_size, batch_size, accuracy, x,y, keep_prob, sess):

    num_batches = math.ceil(data_size/ batch_size)
    last_batch_size = data_size % batch_size

    accs = [] # accuracy for each batch
    for _ in range(num_batches):
        images,labels = next(data_gen)

        #Keep probability to 1 as it is inference
        acc = sess.run(accuracy,feed_dict = {x:images, y:labels, keep_prob:1.})
        accs.append(acc)
    # average of all full batches, except last batch
    acc_full = np.mean(accs[:-1])

    acc = (acc_full *(data_size - last_batch_size)+accs[-1] * last_batch_size)/data_size
    return acc

###########################################
# Conv Neural Network functions
###########################################


###########################################
# Conv Neural Network functions
###########################################
  
def neural_network():
    ''' Define the CNN network'''

    #Tensor representing input images and labels
    x = tf.placeholder(tf.float32, [None,IMG_SIZE,IMG_SIZE,NUM_CHANNELS])
    keep_prob = tf.placeholder(tf.float32)
    y_ = tf.placeholder(tf.float32, [None , NUM_CLASSES])

    # Reshape input picture
    #x = tf.reshape(x, shape=[-1,IMG_SIZE,IMG_SIZE,NUM_CHANNELS])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1 = weight_variable([5, 5, NUM_CHANNELS, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 16 feature to 32 feature.
    W_conv2 = weight_variable([5, 5, 16, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)

    # Third convolutional layer -- maps 32 feature to 64 feature.
    W_conv3 = weight_variable([5, 5, 32, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    # Third pooling layer.
    h_pool3 = max_pool_2x2(h_conv3)

    # Fully connected layer 1 - 4x4x64 feature maps -- maps this to 1024 features.
    W_fc1 = weight_variable([4 * 4 * 64, 1024])
    b_fc1 = bias_variable([1024])

    #x =           tf.reshape(x, shape=[-1,IMG_SIZE,IMG_SIZE,NUM_CHANNELS])
    h_pool3_flat = tf.reshape(h_pool3, shape=[-1, 4*4*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of features.
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # Map the 1024 features to 43 classes, one for each class
    W_fc2 = weight_variable([1024, NUM_CLASSES])
    b_fc2 = bias_variable([NUM_CLASSES])    

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    #loss = tf.reduce_mean(
    #  tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))

    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y_))

    logits = y_conv
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

    prediction = tf.argmax(y_conv,1)

    correct_prediction = tf.equal(prediction, tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return x,y_,keep_prob, logits,optimizer,prediction,accuracy    

###########################################
# Train Conv Neural Network 
###########################################

def train_network():


    #load_data()
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
    
    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']

    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=VALIDATION_SIZE)

    # Launching Graph
    with tf.Graph().as_default(), tf.Session() as sess:
        x,y, keep_prob,logits, optimizer, prediction, accuracy = neural_network()

        # Begin Training
        saver = tf.train.Saver()
        
        # Dump summary for Tensor Board
        train_writer = tf.summary.FileWriter('tf_summary/train', sess.graph)
        
        if RESUME or RESTORE:
            print('Restoring previously trained model at %s'% MODEL_SAVE_PATH)
            #Restore previously trained model
            saver.restore(sess, MODEL_SAVE_PATH)

            #Restore previous accuracy history
            with open('accuracy_history.p','rb')as f:
                accuracy_history = pickle.load(f)

            if RESTORE:
                return accuracy_history
        else:
            print('Training model from scratch')
            #init = tf.initialize_all_variables()
            init = tf.global_variables_initializer()
            sess.run(init)

            #keep track of training and validation accuracy over EPOCH 
            accuracy_history = []

        #Record time elapsed for performance check
        last_time = time.time()
        train_start_time = time.time()

        #Run NUM_EPOCH epochs of training
        for epoch in range(NUM_EPOCH):

            train_gen = next_batch(X_train, y_train, BATCH_SIZE, True)

            num_batches_train = math.ceil(X_train.shape[0]/BATCH_SIZE)

            # Run training on each batch
            for _ in range(num_batches_train):

                images, labels = next(train_gen)

                #perform gradient update in current batch
                sess.run(optimizer, feed_dict={x:images,y:labels, keep_prob:KEEP_PROB})

            #Training set
            train_gen = next_batch(X_train, y_train, BATCH_SIZE_INF, True)
            train_size = X_train.shape[0]
            train_acc = calculate_accuracy(train_gen, train_size, BATCH_SIZE_INF, accuracy, x, y, keep_prob, sess)

            # Validation set
            valid_gen = next_batch(X_valid, y_valid, BATCH_SIZE_INF, True)
            valid_size = X_valid.shape[0]
            valid_acc = calculate_accuracy(valid_gen, valid_size, BATCH_SIZE_INF, accuracy, x, y, keep_prob, sess)

            #record accuracy for report
            accuracy_history.append((train_acc, valid_acc))

            #Print accuracy every 10 epochs
            if(epoch+1)%10 == 0 or epoch ==0 or (epoch+1) == NUM_EPOCH:
                print('Epoch %d -- Train acc.:%.4f, Validation acc.:%.4f, Elapsed time: %.2f sec' %\
                     (epoch+1, train_acc, valid_acc, time.time() - last_time))
                last_time = time.time()

                if SAVE_MODEL:
                    # Save model to disk- check point for every 10 EPOCH
                    save_path = saver.save(sess, MODEL_SAVE_PATH)
                    print('Trained model saved at: %s' % save_path)

                    # Also save accuracy history
                    print('Accuracy history saved at accuracy_history.p')
                    with open('accuracy_history.p', 'wb') as f:
                        pickle.dump(accuracy_history, f)
            

        total_time = time.time() - train_start_time
        print('Total elapsed time: %.2f sec (%.2f min)' % (total_time, total_time/60))          

        # After training is complete, evaluate accuracy on test set
        print('Calculating test accuracy...')
        test_gen = next_batch(X_test, y_test, BATCH_SIZE_INF, False)
        test_size = X_test.shape[0]
        test_acc = calculate_accuracy(test_gen, test_size, BATCH_SIZE_INF, accuracy, x, y, keep_prob, sess)
        print('Test acc.: %.4f' % (test_acc,))

        if SAVE_MODEL:
            # Save model to disk
            save_path = saver.save(sess, MODEL_SAVE_PATH)
            print('Trained model saved at: %s' % save_path)

            # Also save accuracy history
            print('Accuracy history saved at accuracy_history.p')
            with open('accuracy_history.p', 'wb') as f:
                pickle.dump(accuracy_history, f)
    return accuracy_history
                
accuracy_history = train_network()

            
        
        
    
    
    











