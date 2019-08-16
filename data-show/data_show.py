import tensorflow as tf
from  cnn_ import CNN_
from data_get import *
import cv2

alexnet = CNN_(1,0)

#func_save_image(300)   #数据集创建

x_s,y_s = image_get(300)   #数据集读取
print(x_s.shape,y_s.shape) 


#xs_ = x_s[batch_id]
#print(xs_[0])
#print (y_s[batch_id[0]])

#cv2.imshow('', xs_[0])
#cv2.waitKey(0)

def net_predict():

    batch_id = np.random.randint(0,len(x_s),alexnet.batch)
    pre_ = alexnet.prediction_(x_s[batch_id],y_s[batch_id])
    return pre_
 
for i in range(20):

    batch_id = np.random.randint(0,len(x_s),alexnet.batch)
    loss = alexnet.train_(x_s[batch_id],y_s[batch_id])
    
    if loss < 0.05:
        alexnet.save_()
        print('*** train is over ! *** model saved as model/XXX')
        break
    print(" loss: ",loss," || itra: ", i , '|| accuary : ',net_predict() )












