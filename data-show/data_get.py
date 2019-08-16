import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pywt
import os
import cv2

fc = pywt.central_frequency('cgau8')   #定义小波变换
plt.axis('off')

SAMPLE_TIME = 12000

#数据预期采集
def data_get(addr,kind,name):

    data = sio.loadmat(addr)
    fliter_   = filter(lambda x: kind in x, data.keys())
    fliter_list = [item for item in fliter_]
    fliter_key = fliter_list[0]
    out_data = data[fliter_key][:]
    return out_data
 

def data__draw_choose(data):

    def data_batch(data,range):
        basic = np.random.randint(0,120000-range) #这里120000为数据长度
        return np.squeeze(data[basic:basic+range]),basic

    data_,_rand = data_batch(data,4800)   #随机获取长度得数据
    cwtmatr, frequencies = pywt.cwt(data_, np.arange(1,30), 'cgau8',1/SAMPLE_TIME)
    t  = [i for i in range(0,len(data_))]
    plt.contourf(t,frequencies, abs(cwtmatr))
    return _rand

_00_0_97_nomal = data_get('data/00-0-97-nomal.mat', 'DE_time','00-0-97-nomal')
_07_0_97_ball = data_get('data/07-0-97-ball.mat', 'DE_time','07-0-97-ball')
_07_0_97_inner = data_get('data/07-0-97-inner.mat', 'DE_time','07-0-97-inner')
_07_0_97_ext1 = data_get('data/07-0-97-ext1.mat', 'DE_time','07-0-97-ext1')


def func_save_image(size):

    name_= ''  
    #os.mkdir('./image/_00_0_97_nomal')
    #os.mkdir('./image/_07_0_97_ball')
    #os.mkdir('./image/_07_0_97_inner')
    #os.mkdir('./image/_07_0_97_ext1')

    def save_func(path,n):
        name_ = path + '_'+ str(n)+'.tiff'
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.savefig('image/'+path+'/'+name_,bbox_inches='tight',pad_inches=0.0)
        plt.clf()   #清除内存
        print('image/'+name_)

    for n in range(size):
        for i in range(4):       
            if i ==0:
                rand_ =data__draw_choose(_00_0_97_nomal)
                save_func('_00_0_97_nomal',n)              
            elif i ==1:
                rand_ =data__draw_choose(_07_0_97_ball)               
                save_func('_07_0_97_ball',n)
            elif i ==2:
                rand_ =data__draw_choose(_07_0_97_inner)             
                save_func('_07_0_97_inner',n)
            elif i ==3 :
                rand_ =data__draw_choose(_07_0_97_ext1)              
                save_func('_07_0_97_ext1',n)           
            

def image_resize(path):

    img = cv2.imread(path)/255 
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #cv2.imshow("1",img)
    #cv2.waitKey(0)
    #print(a)
    #img = skimage.io.imread(path)/255
    #resized_img = skimage.transform.resize(img, (227, 227))[ :, :, :]   
    #io.imshow(resized_img)
    #io.show()
    return np.asarray( cv2.resize(img, (227, 227)))


def image_get(size):
    
    image_feature = {
        '_00_0_97_nomal':[],
        '_07_0_97_ball':[],
        '_07_0_97_inner':[],
        '_07_0_97_ext1':[]
        }

    image_y = [[]for i in range(4)]

    for k in image_feature.keys():
        dir = './image/' + k
        for file in os.listdir(dir):    
            
            if file.lower().endswith('iff'):

                img_= image_resize(os.path.join(dir, file))

                if len(image_feature[k]) >= size:      
                    break
                image_feature[k].append(img_)#添加校正后得图片
        print(dir, len(image_feature[k]))

    image_y[0] = [[1,0,0,0]for i in range(len(image_feature['_00_0_97_nomal']))]
    image_y[1] = [[0,1,0,0]for i in range(len(image_feature['_07_0_97_ball']))]
    image_y[2] = [[0,0,1,0]for i in range(len(image_feature['_07_0_97_inner']))]
    image_y[3] = [[0,0,0,1]for i in range(len(image_feature['_07_0_97_ext1']))]
    #数据混合

    x_sum = np.concatenate((image_feature['_00_0_97_nomal'],
                            image_feature['_07_0_97_ball'],
                            image_feature['_07_0_97_inner'],
                            image_feature['_07_0_97_ext1']) ,axis =0)

    y_sum = np.concatenate((image_y[0],
                            image_y[1],
                            image_y[2],
                            image_y[3]),axis =0)

    #清除避免占用内存
    image_feature.clear()
    image_y.clear()

    return x_sum,y_sum
    


