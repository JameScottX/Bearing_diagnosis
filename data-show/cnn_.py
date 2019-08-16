import tensorflow as tf
import numpy as np

USED =['conv1' , "pool1","conv2","pool2","conv3","conv4","conv5","pool5"]
LEARNING_RATE = 0.0005



"""alexNet model"""
"""test by Jun WenCui in 2019 7/20"""
class CNN_(object):
    
    def __init__(self, keepPro, sess,skip = USED ,modelPath = "bvlc_alexnet.npy"):
       
        self.KEEPPRO = keepPro
        self.used = skip
        self.modepath = modelPath
        self.out_num =4
        self.batch = 96

        self.x_image = tf.placeholder(tf.float32,[self.batch ,227,227,3])
        self.y_  = tf.placeholder(tf.float32,[None,self.out_num])
       
        prediction = self._cnn_bulid()
  
        with tf.variable_scope('my_loss'):
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(prediction)))   #这里是分类问题编码后的交叉熵损失函数
        with tf.variable_scope('my_train'):
            self.train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.loadModel()

    def _cnn_bulid(self):

        conv1 = convLayer(self.x_image, 11, 11, 4, 4, 96, "conv1", "VALID")
        lrn1 = LRN(conv1, 2, 2e-05, 0.75, "norm1")
        pool1 = maxPoolLayer(lrn1, 3, 3, 2, 2, "pool1", "VALID")

        conv2 = convLayer(pool1, 5, 5, 1, 1, 256, "conv2", groups = 2)
        lrn2 = LRN(conv2, 2, 2e-05, 0.75, "lrn2")
        pool2 = maxPoolLayer(lrn2, 3, 3, 2, 2, "pool2", "VALID")

        conv3 = convLayer(pool2, 3, 3, 1, 1, 384, "conv3")

        conv4 = convLayer(conv3, 3, 3, 1, 1, 384, "conv4", groups = 2)

        conv5 = convLayer(conv4, 3, 3, 1, 1, 256, "conv5", groups = 2)
        pool5 = maxPoolLayer(conv5, 3, 3, 2, 2, "pool5", "VALID")
  
        flatten =tf.reshape(pool5,shape= [-1,256 * 6 * 6])
        #以下为迁移学习部分
        #定义训练层1
        my_layer1 = myLayer(flatten,256 * 6 * 6,4096,tf.nn.relu,"my_layer1")

        dropout1  = tf.nn.dropout(my_layer1, self.KEEPPRO)  #随机失活部分神经元
        #定义训练层1
        my_layer2 = myLayer(dropout1,4096,4096,tf.nn.relu,"my_layer2")

        dropout2  = tf.nn.dropout(my_layer2, self.KEEPPRO)  #随机失活部分神经元
        
        self.net_out = myLayer(dropout2,4096,self.out_num,tf.nn.softmax,"my_layer3")

        return self.net_out     


    def train_(self,x,y):

        loss_ , _ = self.sess.run([self.loss,self.train],{self.x_image: x  ,  self.y_: y})

        return loss_

    def save_(self,path = 'model/'):
        saver = tf.train.Saver()
        saver.save(self.sess,path, write_meta_graph=False)

    def prediction_(self,x,y):
        pre_ = self.sess.run(self.net_out,{self.x_image : x})
        num_correct = tf.equal(tf.argmax(pre_,1),tf.argmax(y,1))
    
        accuary = tf.reduce_mean(tf.cast(num_correct,tf.float32))
        result  = self.sess.run(accuary,{self.x_image: x})

        return result

    def loadModel(self):
        try :
            w_dict  = np.load(self.modepath,encoding = "bytes").item()
        except:
            print("weight-bias data get is wrong")

        for name in w_dict :
            if name  in self.used:
                
                with tf.variable_scope(name,reuse = True):
                    for p in w_dict[name]:
                        if len(p.shape) !=1:
                            self.sess.run(tf.get_variable('w',trainable = False).assign(p))
                        else :
                            self.sess.run(tf.get_variable('b',trainable = False).assign(p))


def convLayer(x, kHeight, kWidth, strideX, strideY,
              featureNum, name, padding = "SAME", groups = 1):
    """convolution"""
    channel = int(x.get_shape()[-1])
    print('@@@',channel)
    conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1, strideY, strideX, 1], padding = padding)
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [kHeight, kWidth, channel/groups, featureNum])
        b = tf.get_variable("b", shape = [featureNum])

        xNew = tf.split(value = x, num_or_size_splits = groups, axis = 3)
        wNew = tf.split(value = w, num_or_size_splits = groups, axis = 3)

        featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]
        mergeFeatureMap = tf.concat(axis = 3, values = featureMap)
        
        out = tf.nn.bias_add(mergeFeatureMap, b)
        return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()), name = scope.name)


def myLayer(x,inputD,outputD,func,name):#训练参数配置

    with tf.variable_scope(name) as scope:
        w = tf.Variable(tf.truncated_normal([inputD,outputD], stddev = 0.01,dtype =tf.float32))
        b = tf.Variable(tf.constant(0.1,shape = [outputD]))
        return func(tf.nn.xw_plus_b(x,w,b))


def fcLayer(x, inputD, outputD, func, name):#使用文件中的权重

    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")
        b = tf.get_variable("b", [outputD], dtype = "float")
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)

        if func == 'Relu':
            return tf.nn.relu(out)
        else:
            return out

def LRN(x, R, alpha, beta, name = None, bias = 1.0):
    return tf.nn.local_response_normalization(x, depth_radius = R, alpha = alpha,
                                            beta = beta, bias = bias, name = name)

def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
                        strides = [1, strideX, strideY, 1], padding = padding, name = name)