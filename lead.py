#@Time      :2018/10/11 16:14
#@Author    :zhounan
# @FileName: LEAD.py

import numpy as np
import tensorflow as tf
from sklearn import svm
from bayesian_network import build_structure
from sklearn.externals import joblib
from Learning_error import hamming_loss

class LEAD(object):

    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.num = self.train_y.shape[0]  # number of dataset
        self.feature_num = train_x.shape[1]
        self.label_num = train_y.shape[1]
        self.errors = np.zeros(train_y.shape)
        self.DAG = np.zeros((self.label_num, self.label_num))
        self.clf_list = []
        self.output = np.zeros(self.label_num)
        self.is_caculate = np.zeros(self.label_num, np.bool)

    def train(self):
        #self.curve_fitting()
        #self.build_DAG()
        self.construct_classifier()


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
            np.save('dataset/errors.npy', self.errors)
            #end for

    # Implementing step 2 as shown in Subsection 2.2.2 in the paper
    #use pgmpy package to build bayesian network structure
    def build_DAG(self):
        edges, DAG = build_structure(self.errors)
        np.save('dataset/DAG.npy', DAG)
        return 0

    # Implementing step 3 as shown in Subsection 2.2.2 in the paper
    #use DAG and svm to build classifier,then save it
    def construct_classifier(self):

        for i in range(self.label_num):
            cols =  np.nonzero(self.DAG[:,i] == 1)
            tempX = self.train_x

            for col in cols:
                tempX = np.c_[tempX, self.train_y[:, col]]
            #end for

            tempY = self.train_y[:, i].reshape(-1,1)
            clf = svm.SVC(kernel='rbf', probability=True)
            clf.fit(tempX, tempY)
            self.clf_list.append(clf)
        #end for

        joblib.dump(self.clf_list, 'sk-model/clf_list.pkl')

     # Implementing step 4 as shown in Subsection 2.2.2 in the paper
    def predict(self, test_x):
        label_parent = []

        for i in range(self.label_num):
            label_parent.append(np.nonzero(self.DAG[:, i] == 1))

        for i in range(self.label_num):
            if self.is_caculate[i] == 0:
                self.output[i], _ = self.predict_single(test_x, label_parent, i)

        return self.output

    def predict_single(self,test_x, label_parent, idx):
        #if idx-th label is independent
        if np.size(label_parent[idx]) == 0:
            print('独立节点计算',idx)
            pro_pos = self.clf_list[idx].predict_proba(test_x)[0][0]
            pro_neg = 1 - pro_pos

            self.is_caculate[idx] = 1

            return pro_pos, pro_neg
        # if idx-th label has parents
        else:
            pa_size = np.size(label_parent[idx])
            print(idx,'有父节点计算,大小', pa_size)
            pa_pro_pos = np.zeros(pa_size)
            pa_pro_neg = np.zeros(pa_size)
            temp_pa_value = np.zeros((2**pa_size, pa_size), np.int64)

            for i in range(2 ** pa_size):
                temp = np.zeros(pa_size, np.int64)
                for j in range(pa_size):
                    temp[j] = ((i >> j) & 1)
                temp_pa_value[i, :] = temp[::-1]

            idx_pa_list = label_parent[idx][0].tolist()
            for i in range (len(idx_pa_list)):
                if self.is_caculate[idx_pa_list[i]] == 0:
                    pa_pro_pos[i], pa_pro_neg[i] =  self.predict_single(test_x, label_parent, idx_pa_list[i])
                else:
                    pa_pro_pos[i] = self.output[idx_pa_list[i]]
                    pa_pro_neg[i] = 1 - pa_pro_pos[i]
                #end if
            #end for

            pro_pos = 0
            for i in range(2 ** pa_size):
                tempX = np.c_[test_x, temp_pa_value[i,:].reshape(1, -1)]
                temp_pro = self.clf_list[idx].predict_proba(tempX)[0][0]

                for j in range(pa_size):
                    if temp_pa_value[i,j] == 1:
                        temp_pro = temp_pro * pa_pro_pos[j]
                    else:
                        temp_pro = temp_pro * pa_pro_neg[j]
                    #end if
                #end for

                pro_pos = pro_pos + temp_pro
            #end for

            pro_neg = 1 - pro_pos
            self.is_caculate[idx] = 1
        #end if
        return pro_pos, pro_neg

    def load_model(self):
        self.DAG = np.load('dataset/DAG.npy')
        self.clf_list = joblib.load('sk-model/clf_list.pkl')

def load_data():
    train_x = np.load('dataset/train_x.npy')
    train_y = np.load('dataset/train_y.npy')
    test_x = np.load('dataset/test_x.npy')
    test_y = np.load('dataset/test_y.npy')

    return train_x, train_y, test_x, test_y

if __name__=='__main__':
    train_x, train_y, test_x, test_y = load_data()
    lead = LEAD(train_x, train_y)
    lead.load_model()
    #lead.train()                   #if you first run this pro ,you should run this function

    output = np.zeros(test_y.shape)
    for i in range(test_x.shape[0]):
        output[i,:] = lead.predict(test_x[i, :].reshape(1, -1))
    predict = np.rint(output).astype(np.int64)
    print(hamming_loss(test_y, predict))
