import os
import time

from multiprocessing import Process
import gluonnlp as nlp
import numpy as np
from gluonnlp.data import SQuAD
from mxnet import nd,gluon
import mxnet as mx
from mxnet.gluon import nn

class Transform(object):
    def __init__(self):
        pass

    def __call__(self, record_index, question_id, question, context, answer_list,
                 answer_start_list):
        return np.ones((100,1)),np.ones((100,3))

def train():
    print(f"PID: {os.getpid()}")
    #input("Press Enter")
    #time.sleep(30)
    print(f"Awake!")
    try:
        print("Start training...")
        train_data = SQuAD('train')
        dataloader = gluon.data.DataLoader(train_data.transform(Transform()),batch_size=128, shuffle=True, num_workers=4)
        net = nn.HybridSequential()
        net.add(nn.Dense(10))
        #net.initialize(mx.init.Xavier(), ctx=mx.gpu(0))
        net.initialize(mx.init.Xavier()) # CPU
        print(net)
        print("Done training")
    except:
        print("Exception training")

a = mx.nd.zeros([1,2,3])

p = Process(target=train)
print('Forking...')
p.start()
print('Forked.')
p.join()
