from torch.nn import Conv2d
from torch.nn import BatchNorm2d
from torch.nn import ZeroPad2d

from torch.nn import Module

import torch.nn.functional as F

class ResBlock(Module):
    
    def __init__(self, in_f, filters):
        super(ResBlock, self).__init__()
        
        f1, f2 = filters
        self.conv_1 = Conv2d(in_f, f1, 1, padding=1, bias=False)
        self.bnorm_1 = BatchNorm2d(f1, eps=1e-03)
        
        self.conv_2 = Conv2d(f1, f2, 3, bias=False)
        self.bnorm_2 = BatchNorm2d(f2, eps=1e-03)
    
        
    def forward(self, input_x):

        x = self.conv_1(input_x)
        x = self.bnorm_1(x)
        x = F.leaky_relu(x, negative_slope=0.1)

        x = self.conv_2(x)
        x = self.bnorm_2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        
        x = input_x + x
        
        return x 
    
class ConvStride2NormRelu(Module):
    
    def __init__(self, in_f, f):
        super(ConvStride2NormRelu, self).__init__()
        
        self.zero = ZeroPad2d((1, 0, 1, 0))
        self.conv = Conv2d(in_f, f, 3, stride=2, bias=False)
        self.bnorm = BatchNorm2d(f, eps=1e-03)
        
    def forward(self, input_x):
        
        x = self.zero(input_x)
        x = self.conv(x)   
        x = self.bnorm(x)
       
        x = F.leaky_relu(x, negative_slope=0.1)
        
        return x

class ConvNormRelu(Module):
    
    def __init__(self, in_f, f, k=3):
        super(ConvNormRelu, self).__init__()
        
        self.conv = Conv2d(in_f, f, k,padding=1, bias=False)
        self.bnorm = BatchNorm2d(f, eps=1e-03)
        
    def forward(self, input_x):
        
        x = self.conv(input_x)   
        x = self.bnorm(x)
       
        x = F.leaky_relu(x, negative_slope=0.1)
        
        return x

class YOLO(Module):
   
    def __init__(self):
        super(YOLO, self).__init__()
        
        self.add_module('conv_100', ConvNormRelu(3, 32))
        self.conv_1 = ConvStride2NormRelu(32, 64)
        
        self.res_2_3 = ResBlock(64,(32,64))
        self.conv_4 = ConvStride2NormRelu(64, 128)

        self.res_5_6 = ResBlock(128,(64,128))
        self.res_7_8 = ResBlock(128,(64,128))
        self.conv_9 = ConvStride2NormRelu(128, 256)

        self.res_10_11 = ResBlock(256,(128,256))
        self.res_12_13 = ResBlock(256,(128,256))
        self.res_14_15 = ResBlock(256,(128,256))
        self.res_16_17 = ResBlock(256,(128,256))
        self.res_18_19 = ResBlock(256,(128,256))
        self.res_20_21 = ResBlock(256,(128,256))
        self.res_22_23 = ResBlock(256,(128,256))
        self.res_24_25 = ResBlock(256,(128,256))

        self.conv_26 = ConvStride2NormRelu(256, 512)
        
    def forward(self, input_x):
        
        x = input_x
        
        for i, m in enumerate(self.named_children()):    
            name, module = m
            print(name)
#             x = m(x)
            
        return x
    
#     def load_weights():
        
#         self.wd = WeightDecoder(weight_file)
#         self.wd.load_weights(self)
        
    
    
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
    scale_3_det = Conv2D(255,1, padding='same', name='conv_58', use_bias=True)(z)
    
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
    scale_2_det = Conv2D(255,1, padding='same', name='conv_66', use_bias=True)(z)
    
    #26
    x = conv_norm_relu(x, 128, 67, size=1)
    #52
    x = UpSampling2D(2)(x)    
    #52x52x(128 + 256)
    x = concatenate([x, scale_1_conc_1])

    for i in range(3):
        x = conv_norm_relu(x, 128, i*2 + 68, size=1)
        x = conv_norm_relu(x, 256, i*2 + 69)
    
    scale_1_det = Conv2D(255,1, padding='same', name='conv_74', use_bias=True)(x)

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
        for i, layer in enumerate(model.children()):

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
                bias   = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                print('filter weights = ', conv_layer.get_weights()[0].shape)
                kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2,3,1,0])
                conv_layer.set_weights([kernel, bias])
            else:
                kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2,3,1,0])
                conv_layer.set_weights([kernel])
            print(self.offset, len(self.all_weights))
             
                
    def reset(self):
        self.offset = 0

