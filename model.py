from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Add
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import BatchNormalization 
from tensorflow.python.keras.layers import UpSampling2D
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers import ZeroPadding2D

from tensorflow.python.keras.layers.merge import add, concatenate

from tensorflow.python.keras.models import Model

def res_block(input_x, filters, sizes, layer_id):
    
    f1, f2 = filters
    s1, s2 = sizes
    
    #1
    x = Conv2D(f1, s1, padding='same', use_bias=False, name='conv_' + str(layer_id))(input_x)
    x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(layer_id))(x)
    x = LeakyReLU(alpha=0.1, name='leaky_' + str(layer_id))(x)
    
    #2
    x = Conv2D(f2, s2, padding='same', use_bias=False, name='conv_' + str(layer_id + 1))(x)
    x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(layer_id + 1))(x)
    x = LeakyReLU(alpha=0.1, name='leaky_' + str(layer_id + 1))(x)
        
    x = add([input_x, x])
    
    return x

def conv_stride_2_norm_relu(input_x, f, layer_id):
    
    x = ZeroPadding2D(((1,0),(1,0)))(input_x)
    x = Conv2D(f, 3, padding='valid', use_bias=False, strides=2, name='conv_' + str(layer_id))(x)
    x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(layer_id))(x)
    x = LeakyReLU(alpha=0.1, name='leaky_' + str(layer_id))(x)
    
    return x

def conv_norm_relu(input_x, f, layer_id, size=3):
    
    x = Conv2D(f, size, padding='same', use_bias=False, name='conv_' + str(layer_id))(input_x)
    x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(layer_id))(x)
    x = LeakyReLU(alpha=0.1, name='leaky_' + str(layer_id))(x)    

    return x
    
def compile_model():
    
    input_x = Input(shape=(None, None, 3)) 
    
    #1, 416
    x = conv_norm_relu(input_x,32,0) 

    #2, 208
    x = conv_stride_2_norm_relu(x,64,1)    
    
    #3,4, 208
    x = res_block(x,(32,64),(1,3),2)
    
    #5, 104
    x = conv_stride_2_norm_relu(x,128,4)    
    
    #6-9, 104
    for i in range(2):
        x = res_block(x,(64,128),(1,3),i*2 + 5)
    
    #10, 52
    x = conv_stride_2_norm_relu(x,256,9)    
    
    #11-26, 52
    for i in range(8):
        x = res_block(x,(128,256),(1,3), i*2+10)
    
    #saving for future conc for scale 1, 52x52x256
    scale_1_conc_1 = x 
    
    #27, 26
    x = conv_stride_2_norm_relu(x,512,26)    
        
    #28-44, 26
    for i in range(8):
        x = res_block(x,(256,512),(1,3), i*2+27)
       
    #saving for future conc for scale 2, 26x26x512
    scale_2_conc_1 = x 

    #45, 13
    x = conv_stride_2_norm_relu(x,1024,43)    
    
    #46-53, 13
    for i in range(4):
        x = res_block(x,(512,1024),(1,3), i*2 + 44)
    
    #Darknet-53 ended here
    
    #scale 3 detection, 13
    for i in range(2):
        x = conv_norm_relu(x, 512, i*2 + 52, size=1)
        x = conv_norm_relu(x, 1024, i*2 + 53)
    
    x = conv_norm_relu(x, 512, 56, size=1)
    z = conv_norm_relu(x, 1024, 57)
    
    #13x13x255
    scale_3_det = Conv2D(75, 1, padding='same', name='conv_58', use_bias=True)(z)
    
    #saving for future conc for scale 2
    #conv + upsample for scale 2
    #13
    x = conv_norm_relu(x, 256, 59, size=1)
    #26
    x = UpSampling2D(2)(x)
    #26x26x(256+512) 
    x = concatenate([x, scale_2_conc_1])
    
    for i in range(2):
        x = conv_norm_relu(x, 256, i*2 + 60, size=1)
        x = conv_norm_relu(x, 512, i*2 + 61)
        
    x = conv_norm_relu(x, 256, 64, size=1)
    z = conv_norm_relu(x, 512, 65)
    
    #saving for future conc for scale 1
    #conv + upsample for scale 1
    
    #scale 2 detection
    scale_2_det = Conv2D(75, 1, padding='same', name='conv_66', use_bias=True)(z)
    
    #26
    x = conv_norm_relu(x, 128, 67, size=1)
    #52
    x = UpSampling2D(2)(x)    
    #52x52x(128 + 256)
    x = concatenate([x, scale_1_conc_1])

    for i in range(3):
        x = conv_norm_relu(x, 128, i*2 + 68, size=1)
        x = conv_norm_relu(x, 256, i*2 + 69)
    
    #scale 1 detection 
    scale_1_det = Conv2D(75,1, padding='same', name='conv_74', use_bias=True)(x)

    model = Model(input_x, [scale_3_det, scale_2_det, scale_1_det])
    
    return model


import struct
import numpy as np

class WeightDecoder:
    def __init__(self, weight_file):
        with open(weight_file, 'rb') as w_f:
            major, = struct.unpack('i', w_f.read(4))
            minor, = struct.unpack('i', w_f.read(4))
            revision, = struct.unpack('i', w_f.read(4))
            if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
                w_f.read(8)
            else:
                w_f.read(4)
            transpose = (major > 1000) or (minor > 1000)
            binary = w_f.read()
        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype='float32')
 
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
 
    def load_weights(self, model):
        for i in range(75):
            conv_layer = model.get_layer('conv_' + str(i))
            print("loading weights of convolution #" + str(i))
                
            if i not in [58, 66, 74]:
                norm_layer = model.get_layer('bnorm_' + str(i))
                size = np.prod(norm_layer.get_weights()[0].shape)
                beta  = self.read_bytes(size) # bias
                gamma = self.read_bytes(size) # scale
                mean  = self.read_bytes(size) # mean
                var   = self.read_bytes(size) # variance
                weights = norm_layer.set_weights([gamma, beta, mean, var])
             
            if len(conv_layer.get_weights()) > 1:
#                 print(self.offset)
#                 bias   = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
#                 print('filter weights = ', conv_layer.get_weights()[0].shape)
#                 kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
#                 kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
#                 kernel = kernel.transpose([2,3,1,0])
#                 conv_layer.set_weights([kernel, bias])
#                 print(self.offset)
                
                s = 0
                
                if i == 58:
                    s += 1024*255 + 255
                elif i == 66:
                    s += 512*255 + 255
                elif i == 74:
                    s += 256*255 + 255
                    
                self.offset += s

            else:
                kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2,3,1,0])
                conv_layer.set_weights([kernel])
            print(self.offset, len(self.all_weights))
             
                
    def reset(self):
        self.offset = 0

