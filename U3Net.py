from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from imutils import paths
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input, MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Add
from tensorflow.keras import regularizers 
from tensorflow.keras.layers import BatchNormalization as bn
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import model_from_json
from tensorflow.keras.regularizers import l2

import tensorflow.keras.backend as K

import numpy as np
import random
import os
import cv2
from pathlib import Path
from sklearn import preprocessing
import tensorflow as tf
print(tf.test.gpu_device_name())

dropout_rate = 0.2
l2_lambda = 0.0002

'''
nb_conv is the number of the convolution
          
rate: is the dilate rate of the convolution
'''
def conv_unit(input_tensor, stage, nb_filter, kernel_size=3, nb_conv = 1, rate = 1):

    act = 'relu'
    
    x = Conv2D(nb_filter, (kernel_size, kernel_size), dilation_rate=(rate, rate), activation=act, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = Dropout(dropout_rate, name='drop'+stage+'_1')(x)
    if (nb_conv==2):
        x = Conv2D(nb_filter, (kernel_size, kernel_size),dilation_rate=(rate, rate), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(dropout_rate, name='drop'+stage+'_2')(x)
    x = bn(name = 'bn'+stage)(x)

    return x

'''
    We defined 2 type of node: basic node and dense node
    The basic node (b_node) is not dense structure
    The dense_node (d_node) is dense structure
    
'''


def first_b_node(input_tensor, dilate_mode = False, nb_conv = 1):
    ### create the dilate rate,
    if dilate_mode:
        dilate_rate = [1,2,3]
    else: dilate_rate = [1,1,1]
   
    x1 = conv_unit(input_tensor,stage = '00',nb_filter = 32, nb_conv = nb_conv, rate = dilate_rate[0])
    x2 = conv_unit(x1,stage = '01',nb_filter = 32, nb_conv = nb_conv, rate = dilate_rate[1])
    x3 = conv_unit(x2,stage = '02',nb_filter = 32, nb_conv = nb_conv, rate = dilate_rate[2])
    return x1,x2,x3

def first_d_node(input_tensor, dilate_mode = False, nb_conv = 1):
    ### create the dilate rate
    if dilate_mode:
        dilate_rate = [1,2,3]
    else: dilate_rate = [1,1,1]
    x1 = conv_unit(input_tensor,stage = '00',nb_filter = 32, nb_conv = nb_conv, rate = dilate_rate[0])
    x2 = conv_unit(x1,stage = '01',nb_filter = 32, nb_conv = nb_conv, rate = dilate_rate[1])
    x3 = concatenate([x1, x2], name='concat01', axis=bn_axis)
    x3 = conv_unit(x3,stage = '02',nb_filter = 32, nb_conv = nb_conv, rate = dilate_rate[2])
    
    return x1,x2,x3

def b_node(x1,x2,x3, dilate_mode = False, nb_conv = 1, node_nb = 2):
    ### create the dilate rate
    nb_filter = [32,64,128,256,512]
    if dilate_mode:
        dilate_rate = [1,2,3]
    else: dilate_rate = [1,1,1]
    y1 = conv_unit(x1,stage = str(node_nb)+'0',nb_filter = nb_filter[node_nb-1], nb_conv = nb_conv, rate = dilate_rate[0])
    y2 = concatenate([x2, y1], name='concat'+str(node_nb)+'0', axis=bn_axis)
    y2 = conv_unit(y2,stage = str(node_nb)+'1',nb_filter = nb_filter[node_nb-1], nb_conv = nb_conv, rate = dilate_rate[1])
    y3 = concatenate([x3, y2], name='concat'+str(node_nb)+'1', axis=bn_axis)
    y3 = conv_unit(y3,stage = str(node_nb)+'2',nb_filter = nb_filter[node_nb-1], nb_conv = nb_conv, rate = dilate_rate[2])
    
    return y1,y2,y3

def d_node(x1,x2,x3, dilate_mode = False, nb_conv = 1, node_nb = 2):
    ### create the dilate rate
    nb_filter = [32,64,128,256,512]
    if dilate_mode:
        dilate_rate = [1,2,3]
    else: dilate_rate = [1,1,1]
    y1 = conv_unit(x1,stage = str(node_nb)+'0',nb_filter = nb_filter[node_nb-1], nb_conv = nb_conv, rate = dilate_rate[0])
    y2 = concatenate([x2, y1], name='concat'+str(node_nb)+'0', axis=bn_axis)
    y2 = conv_unit(y1,stage = str(node_nb)+'1',nb_filter = nb_filter[node_nb-1], nb_conv = nb_conv, rate = dilate_rate[1])
    y3 = concatenate([x3, y1, y2], name='concat'+str(node_nb)+'1', axis=bn_axis)
    y3 = conv_unit(y3,stage = str(node_nb)+'2',nb_filter = nb_filter[node_nb-1], nb_conv = nb_conv, rate = dilate_rate[2])
        
    return y1,y2,y3

def pooling(x1,x2,x3,layer):
    y1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool'+str(layer)+'0')(x1)
    y2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool'+str(layer)+'1')(x2)
    y3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool'+str(layer)+'2')(x3)
    
    return y1,y2,y3

def upconv(x1,x2,x3,layer):
    nb_filter = [256,128,64,32]
    y1 = Conv2DTranspose(nb_filter[layer], (2, 2), strides=(2, 2), name='upconv'+str(layer)+'0', padding='same')(x1)
    y2 = Conv2DTranspose(nb_filter[layer], (2, 2), strides=(2, 2), name='upconv'+str(layer)+'1', padding='same')(x2)
    y3 = Conv2DTranspose(nb_filter[layer], (2, 2), strides=(2, 2), name='upconv'+str(layer)+'2', padding='same')(x3)
    
    return y1,y2,y3

### basic node in encoder
def b_node_u(s1,s2,s3,x1,x2,x3, dilate_mode = False, nb_conv = 1, node_nb = 6):
    ### create the dilate rate
    nb_filter = [32,64,128,256,512]
    if dilate_mode:
        dilate_rate = [1,2,3]
    else: dilate_rate = [1,1,1]
    y1 = concatenate([x1, s1], name='concat'+str(node_nb)+'0', axis=bn_axis)
    y1 = conv_unit(y1,stage = str(node_nb)+'0',nb_filter = nb_filter[9-node_nb], nb_conv = nb_conv, rate = dilate_rate[0])
    y2 = concatenate([x2, s2, y1], name='concat'+str(node_nb)+'1', axis=bn_axis)
    y2 = conv_unit(y2,stage = str(node_nb)+'1',nb_filter = nb_filter[9-node_nb], nb_conv = nb_conv, rate = dilate_rate[1])
    y3 = concatenate([x3, s3, y2], name='concat'+str(node_nb)+'2', axis=bn_axis)
    y3 = conv_unit(y3,stage = str(node_nb)+'2',nb_filter = nb_filter[9-node_nb], nb_conv = nb_conv, rate = dilate_rate[2])
        
    return y1,y2,y3

def d_node_u(s1,s2,s3,x1,x2,x3, dilate_mode = False, nb_conv = 1, node_nb = 6):
    ### create the dilate rate
    nb_filter = [32,64,128,256,512]
    if dilate_mode:
        dilate_rate = [1,2,3]
    else: dilate_rate = [1,1,1]
    y1 = concatenate([x1, s1], name='concat'+str(node_nb)+'0', axis=bn_axis)
    y1 = conv_unit(y1,stage = str(node_nb)+'0',nb_filter = nb_filter[9-node_nb], nb_conv = nb_conv, rate = dilate_rate[0])
    y2 = concatenate([x2, s2, y1], name='concat'+str(node_nb)+'1', axis=bn_axis)
    y2 = conv_unit(y2,stage = str(node_nb)+'1',nb_filter = nb_filter[9-node_nb], nb_conv = nb_conv, rate = dilate_rate[1])
    y3 = concatenate([x3, s3, y1, y2], name='concat'+str(node_nb)+'2', axis=bn_axis)
    y3 = conv_unit(y3,stage = str(node_nb)+'2',nb_filter = nb_filter[9-node_nb], nb_conv = nb_conv, rate = dilate_rate[2])
        
    return y1,y2,y3
    
    
#####3333333333333333333333333333333333333333333333333333333
##### Dense Multi Unet using the pre-define function  333333
#####3333333333333333333333333333333333333333333333333333333

def M3U_Netpp(img_rows, img_cols, channel=3, num_class=3,deep_supervision=False,dilate_mode = False,nb_conv = 1):
    
    # Handle Dimension Ordering for different backends
    global bn_axis
    
    bn_axis = 3
    img_input = Input(shape=(img_rows, img_cols, channel), name='main_input')
    
    
    ### encoder
    c01,c02,c03 = first_d_node(img_input, dilate_mode, nb_conv)
    p01,p02,p03 = pooling(c01,c02,c03, layer = 0)
    
    c11,c12,c13 = d_node(p01,p02,p03, dilate_mode, nb_conv, node_nb = 2)
    p11,p12,p13 = pooling(c11,c12,c13, layer = 1)
    
    c21,c22,c23 = d_node(p11,p12,p13, dilate_mode, nb_conv, node_nb = 3)
    p21,p22,p23 = pooling(c21,c22,c23, layer = 2)
    
    c31,c32,c33 = d_node(p21,p22,p23, dilate_mode, nb_conv, node_nb = 4)
    p31,p32,p33 = pooling(c31,c32,c33, layer = 3)
    
    c41,c42,c43 = d_node(p31,p32,p33, dilate_mode, nb_conv, node_nb = 5)
    
    ### decoder
    u01,u02,u03 = upconv(c41,c42,c43,layer=0)
    c51,c52,c53 = d_node_u(c31,c32,c33,u01,u02,u03, dilate_mode, nb_conv, node_nb = 6)
    
    u11,u12,u13 = upconv(c51,c52,c53,layer=1)
    c61,c62,c63 = d_node_u(c21,c22,c23,u11,u12,u13, dilate_mode, nb_conv, node_nb = 7)
    
    u21,u22,u23 = upconv(c61,c62,c63,layer=2)
    c71,c72,c73 = d_node_u(c11,c12,c13,u21,u22,u23, dilate_mode, nb_conv, node_nb = 8)
    
    u31,u32,u33 = upconv(c71,c72,c73,layer=3)
    c81,c82,c83 = d_node_u(c01,c02,c03,u31,u32,u33, dilate_mode, nb_conv, node_nb = 9)
    
    ####Output
    output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(c81)
    output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(c82)
    output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(c83)
    
    
    deep_out = concatenate([output_1,output_2,output_3], name = 'concat_out', axis=bn_axis)
    deep_out = Conv2D(num_class, (1, 1), activation='sigmoid', name='deep_out', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(deep_out)
    
    if deep_supervision:
        model = Model(img_input, [deep_out])
    else:
        model = Model(img_input, [output_3])

    return model