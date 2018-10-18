#@Time      :2018/10/11 16:14
#@Author    :zhounan
# @FileName: LEAD.py

import numpy as np
import tensorflow as tf
from sklearn import svm
from BayesianNetwork import build_structure
from sklearn.externals import joblib

class LEAD(object):

    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.num = self.train_y.shape[0]  # number of dataset
        self.feature_num = train_x.shape[1]
        self.label_num = train_y.shape[1]
        self.errors = np.zeros(train_y.shape)
        self.DAG = np.zeros(self.label_num, self.label_num)
        self.clf_list = []

    def train(self):
        self.curve_fitting()  #
        self.build_DAG()
        self.construct_classifier()
        self.load_model()

    # Implementing step 1 as shown in Subsection 2.2.2 in the paper
    #use tesorflow to construct 3 layers neural networks to implement nonlinear regression
    def curve_fitting(self):
        batch_size = 64
        dataset_size = self.num

        x = tf.placeholder(tf.float32, [None, self.feature_num], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 1], name='y-input')

        w1 = tf.Variable(tf.random_normal([self.feature_num, 3], stddev=1, seed=1))
        w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

        bias1 = tf.Variable(tf.random_normal([3], stddev=1, seed=1))
        bias2 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))

        a = tf.nn.sigmoid(tf.matmul(x, w1) + bias1)
        y = tf.nn.sigmoid(tf.matmul(a, w2) + bias2)

        cross_entropy = tf.reduce_mean(-y_ * tf.log(tf.clip_by_value(y,1e-8,1.0)) - (1-y_) * tf.log(1 - tf.clip_by_value(y,1e-8,1.0)))
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            for i in range(self.label_num):
                #迭代次数
                steps = 5000
                for j in range(steps):
                    start = (j * batch_size) % dataset_size
                    end = min(start + batch_size, dataset_size)

                    sess.run(train_step, feed_dict={x: self.train_x[start:end], y_: self.train_y[start:end, i].reshape(-1,1)})
                    #print("begin to train")

                    if i % 1000 == 0:
                        total_cross_entropy = sess.run(cross_entropy, feed_dict={x: train_x, y_:self.train_y[:, i].reshape(-1,1)})
                        #print(sess.run(y_, feed_dict={y_: self.train_y[:, i].reshape(-1, 1)}))
                        #print("entropy   " , total_cross_entropy)

                error = y_ - tf.round(y)
                self.errors[:, i] = sess.run(error, feed_dict={x: train_x, y_: train_y[:, i].reshape(-1,1)}).flatten()
                #print(self.errors[:,i])
                #end for

            saver = tf.train.Saver()
            saver.save(sess, "./tf_model/model")
            self.errors = self.errors.astype(np.int)
            np.save('prepare_data/errors.npy', self.errors)
            #end for

    # Implementing step 2 as shown in Subsection 2.2.2 in the paper
    #use pgmpy package to build bayesian network structure
    def build_DAG(self):
        DAG = build_structure(self.errors)

        np.save('prepare_data/DAG.npy', DAG)
        return 0

    # Implementing step 3 as shown in Subsection 2.2.2 in the paper
    #use DAG and svm to build classifier,then save it
    def construct_classifier(self):

        for i in range(self.label_num):
            cols =  np.nonzero(self.DAG[:,i] == 1)

            for col in cols:
                tempX = np.c_[self.train_x, self.train_y[:, col]]
                tempY = np.delete(self.train_y, col, axis=1)
            #end for

            clf = svm.SVC(kernel='rbf')
            clf.fit(tempX, tempY)
            self.clf_list[i] = clf
        #end for

        joblib.dump(self.clf_list, 'sk-model/clf_list.pkl')

     # Implementing step 4 as shown in Subsection 2.2.2 in the paper
    def predict(self, test_x):
        return 0

    def load_model(self):
        self.DAG = np.load('prepare_data/DAG.npy')
        self.clf_list = joblib.load('save/clf.pkl')

def load_data():
    train_x = np.load('prepare_data/train_x.npy')
    train_y = np.load('prepare_data/train_y.npy')
    test_x = np.load('prepare_data/test_x.npy')
    test_y = np.load('prepare_data/test_y.npy')

    return train_x, train_y

if __name__=='__main__':
    train_x, train_y = load_data()
    lead = LEAD(train_x, train_y)
    #lead.train()                   #if you first run this pro ,you should run this function



