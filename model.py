###########################################################
##  LCNN is defined in this file                          #
###########################################################
##  Copyright (c) 2018, National Institute of Informatics #
##  Author:      Fuming Fang                              #
##  Affiliation: National Institute of Informatics        #
##  Email:       fang@nii.ac.jp                           #
###########################################################

from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
import sys
import struct
from dataio import dataio

dim = 864
length = 400

class lcnn(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        
        self.batch_size = args.batch_size
        self.beta1 = args.beta1
        self.w_weight = args.w_weight
        
        self.build_model()
        self.saver = tf.train.Saver()


    def linear(self, data_in, out_size, name=None, stddev=0.02, bs=0.0):
        with tf.variable_scope(name):
            weight = tf.get_variable("weight", [data_in.get_shape()[-1], out_size], tf.float32,
                                     tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable("bias", [out_size],
                                   initializer=tf.constant_initializer(bs))
        
            return tf.matmul(data_in, weight) + bias


    def conv2d(self, data_in, filters, kernel_size, strides, padding='SAME', name=None):
        conv = tf.layers.conv2d(data_in, filters, kernel_size, strides,
                                use_bias=True, padding=padding, name=name,
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        return conv

    def mfm(self, data_in, out_size, name=None):
        t1 = data_in[:,:,:,:out_size]
        t2 = data_in[:,:,:,out_size:]
        return tf.maximum(t1, t2)

    def max_pool(self, data_in, ksize, strides, padding='SAME', name=None):
        ksize = [1, ksize, ksize, 1]
        strides = [1, strides, strides, 1]
        return tf.nn.max_pool(data_in, ksize=ksize, strides=strides,
                              padding=padding, name=name)

    def define_network(self):
        self.data = tf.placeholder(tf.float32, [None, dim, length, 1], name='input_data')
        self.keep_prob = tf.placeholder(tf.float32, None, name='keep_prob')

        conv1 = self.conv2d(self.data, 32, 5, 1, name='conv1')
        mfm1 = self.mfm(conv1, 16, name='mfm1')

        maxpool1 = self.max_pool(mfm1, 2, 2, name='maxpool1')

        conv2a = self.conv2d(maxpool1, 32, 1, 1, name='conv2a')
        mfm2a = self.mfm(conv2a, 16, name='mfm2a')
        conv2b = self.conv2d(mfm2a, 48, 3, 1, name='conv2b')
        mfm2b = self.mfm(conv2b, 24, name='mfm2b')

        maxpool2 = self.max_pool(mfm2b, 2, 2, name='maxpool2')

        conv3a = self.conv2d(maxpool2, 48, 1, 1, name='conv3a')
        mfm3a = self.mfm(conv3a, 24, name='mfm3a')
        conv3b = self.conv2d(mfm3a, 64, 3, 1, name='conv3b')
        mfm3b = self.mfm(conv3b, 32, name='mfm3b')

        maxpool3 = self.max_pool(mfm3b, 2, 2, name='maxpool3')

        conv4a = self.conv2d(maxpool3, 64, 1, 1, name='conv4a')
        mfm4a = self.mfm(conv4a, 32, name='mfm4a')
        conv4b = self.conv2d(mfm4a, 32, 3, 1, name='conv4b')
        mfm4b = self.mfm(conv4b, 16, name='mfm4b')

        maxpool4 = self.max_pool(mfm4b, 2, 2, name='maxpool4')

        conv5a = self.conv2d(maxpool4, 32, 1, 1, name='conv5a')
        mfm5a = self.mfm(conv5a, 16, name='mfm5a')
        conv5b = self.conv2d(mfm5a, 32, 3, 1, name='conv5b')
        mfm5b = self.mfm(conv5b, 16, name='mfm5b')

        maxpool5 = self.max_pool(mfm5b, 2, 2, padding='VALID', name='maxpool5')

        flatten = tf.contrib.slim.flatten(maxpool5, scope='flatten_maxpool5')
        fc6 = self.linear(flatten, 64, name='fc6')
        mfm6 = tf.maximum(fc6[:,:32], fc6[:,32:])
        self.fc6 = tf.nn.dropout(mfm6, self.keep_prob, name='dropout_fc6')
        
        fc7 = self.linear(self.fc6, 2, name='fc7')
        
        return fc7

    def build_model(self):
        out = self.define_network()
        
        self.y = tf.placeholder(tf.float32, [None, 2],name='label')
        self.prediction = tf.nn.softmax(out)
        correct_prediction = tf.equal(tf.argmax(self.prediction,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.test_acc = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=out, labels=self.y))
        
        self.define_optm()

        
    def define_optm(self):
        t_vars = tf.trainable_variables()
        for var in t_vars:
            print(var.name)

        self.lr = tf.placeholder(tf.float32, None, name='lr')
        self.optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1) \
                             .minimize(self.cross_entropy)
        
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        
    def train(self, args):
        if args.retrain and self.load(args.checkpoint_dir):
            print("successfully loaded")
        else:
            print("failed to load")
        sys.stdout.flush()
        
        counter = 0
        start_time = time.time()

        self.dataio = dataio(args.train_genuine, args.train_spoof,
                             args.dev_genuine, args.dev_spoof,
                             batch_size=self.batch_size)
        
        n_epoch = args.epoch
        batch_idxs = self.dataio.frames // self.batch_size
        lr = args.lr

        previous_acc = self.dev_acc(0)
        early_stop = 0
        
        for epoch in range(n_epoch):
            self.dataio.shuffle()

            for idx in range(0, batch_idxs):
                batch_x, batch_y = self.dataio.batch()
                feed_dict = {self.data: batch_x, self.y: batch_y,
                             self.lr: lr, self.keep_prob: args.keep_prob}
                _ = self.sess.run(self.optim, feed_dict=feed_dict)

                if counter % args.print_freq == 0:
                    feed_dict[self.keep_prob] = 1.0
                    loss, acc = self.sess.run([self.cross_entropy, self.accuracy],
                                              feed_dict=feed_dict)
                    print("Epoch: [%2d %4d/%4d], loss: [%.6f], acc: [%.2f%%]" %
                          (epoch+1, idx, batch_idxs, loss, acc*100.0))
                    sys.stdout.flush()

                if np.mod(counter, args.save_freq) == 2:
                    self.save(args.checkpoint_dir, counter)

                counter += 1

            acc = self.dev_acc(epoch+1)
            if previous_acc > acc:
                early_stop += 1
                lr = lr * args.dlr
                print('set learning rate: %.12f' % lr)
                sys.stdout.flush()
            else:
                early_stop = 0
                
            previous_acc = acc

    def dev_acc(self, epoch):
        acc = 0.0
        for idx in range(self.dataio.dev_iterations):
            x, y = self.dataio.dev_batch()
            feed_dict = {self.data: x, self.y: y, self.keep_prob: 1.0}
            acc += self.sess.run(self.test_acc, feed_dict=feed_dict)

        acc /= self.dataio.dev_frames
        print('Epoch: %2d, acc: [%.2f%%]' % (epoch, acc*100.0))
        sys.stdout.flush()

        return acc
            

    def save(self, checkpoint_dir, step):
        model_name = "lcnn.model"
        checkpoint_dir = self.checkpoint_dir
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print("loading model parameters ...")

        checkpoint_dir = self.checkpoint_dir

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
        
    def test(self, args):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.dataio = dataio(None, None, None, None, args.test_data)
       
        if self.load(args.checkpoint_dir):
            print("successfully loaded")
        else:
            print("failed to load")

        if args.out_type == 'class':
            network = self.prediction
        elif args.out_type == 'feature':
            network = self.fc6
        else:
            raise NotImplementedError
            
        for idx in range(self.dataio.test_frames):
            print(self.dataio.test_names[idx])
            sys.stdout.flush()

            data = self.dataio.test_data[idx]
            feed_dict = {self.data: data, self.keep_prob: 1.0}
            output = self.sess.run(network, feed_dict=feed_dict)
            if args.out_type == 'class':
                output = np.log(output + np.exp(-100))

            name = os.path.basename(self.dataio.test_names[idx])
            name = os.path.join(args.test_dir, name)
            self.dataio.save_data(name, output)
