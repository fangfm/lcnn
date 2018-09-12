#! /usr/bin/env python

import argparse
import os
import tensorflow as tf
import numpy as np
import sys
from model import lcnn

parser = argparse.ArgumentParser(description='')
parser.add_argument('--train_genuine', type=str, default=None)
parser.add_argument('--train_spoof', type=str, default=None)
parser.add_argument('--dev_genuine', type=str, default=None)
parser.add_argument('--dev_spoof', type=str, default=None)
parser.add_argument('--test_data', type=str, default=None)

parser.add_argument('--epoch', dest='epoch', type=int, default=9)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1)

parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--dlr', type=float, default=0.9)
parser.add_argument('--keep_prob', type=float, default=1.0)
parser.add_argument('--w_weight', type=float, default=0.001)

parser.add_argument('--out_type', type=str, default='class', help='class/feature')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=50)
parser.add_argument('--print_freq', dest='print_freq', type=int, default=10)
parser.add_argument('--retrain', dest='retrain', type=bool, default=True)
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint')
parser.add_argument('--test_dir', dest='test_dir', default='./test')

def main(args):    
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if args.phase == 'test' and not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = lcnn(sess, args)
        model.train(args) if args.phase == 'train' \
            else model.test(args)

if __name__ == '__main__':
    args = parser.parse_args()

    rseed = np.arange(100)
    np.random.shuffle(rseed)
    tf.set_random_seed(rseed[0])    
    print("random seed: %d" % rseed[0])
    sys.stdout.flush()
    
    main(args)
