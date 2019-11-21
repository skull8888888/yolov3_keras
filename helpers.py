from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
import os

from PIL import Image
import matplotlib.pyplot as plot
import matplotlib.patches as patches
import numpy as np

from bs4 import BeautifulSoup as bs

import math
import tensorflow.keras.backend as K

def calculate_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    return iou


def get_bboxes(image_filename): 
    
    file = open('VOCdevkit/VOC2012/Annotations/' + image_filename + '.xml').read()
    annot = bs(file, features="lxml")

    bboxes = []
    
    for object in annot.findAll('object'):
        try:
            class_name = object.find('name').text
            xmin = int(float(object.xmin.text))
            xmax = int(float(object.xmax.text))
            ymin = int(float(object.ymin.text))
            ymax = int(float(object.ymax.text))

            bboxes.append([xmin,ymin,xmax,ymax,class_name])
        except:
            print('Broken ', image_filename)
    return bboxes

def resize_image(image_name):

    size = 416, 416
    
    im = Image.open('VOCdevkit/VOC2012/JPEGImages/' + image_name + '.jpg')
    
    scale = 416 / max(im.size)
    
    im.thumbnail(size, Image.ANTIALIAS)
    
    x_offset = (size[0] - im.size[0])//2
    y_offset = (size[1] - im.size[1])//2
    
         
    final = Image.new(mode='RGB',size=size,color=(0,0,0))
    final.paste(im, (x_offset, y_offset))
                
    bboxes = get_bboxes(image_name)
    scaled_bboxes = []
    
    for bbox in bboxes:
        
        xmin = int(bbox[0] * scale) + x_offset
        ymin = int(bbox[1] * scale) + y_offset
        xmax = int(bbox[2] * scale) + x_offset
        ymax = int(bbox[3] * scale) + y_offset

        scaled_bbox = [xmin,ymin,xmax,ymax, bbox[4]]
        scaled_bboxes.append(scaled_bbox)
                
    return np.array(final), scaled_bboxes

def resize_transform_image(image_name):

    size = 416, 416
    
    im = Image.open(image_name + '.jpg')
    
    scale = C.IMG_WIDTH / max(im.size)
    
    im.thumbnail(size, Image.ANTIALIAS)
    
    x_offset = (size[0] - im.size[0])//2
    y_offset = (size[1] - im.size[1])//2
    
         
    final = Image.new(mode='RGB',size=size,color=(0,0,0))
    final.paste(im, (x_offset, y_offset))
                             
    return np.array(final)

def flip_horizontally(C, img, scaled_bboxes):

    flipped_img = np.fliplr(img)
         
    for i, bbox in enumerate(scaled_bboxes):
        
        x1 = C.IMG_WIDTH - bbox[0]
        x2 = C.IMG_WIDTH - bbox[2]
        
        bbox[0] = min(x1,x2)
        bbox[2] = max(x1,x2)
        
        scaled_bboxes[i] = bbox
        
    return flipped_img, scaled_bboxes

def shift_intensity(img):
   
    params = {
        'channel_shift_intensity':10
    }
    new_img = IDG().apply_transform(img,params)
    
    return new_img

def shift_brightness(img):
    
    params = {
        'brightness':0.01, 
    }
    new_img = IDG().apply_transform(img,params)
    
    return new_img

def get_img_bboxes(image_name):
    
#     need_shift_brightness = np.random.randint(2,size=1)[0]
#     need_shift_incensity = np.random.randint(2,size=1)[0]
#     need_flip = np.random.randint(2,size=1)[0]
    
    img, bboxes = resize_image(image_name)
   
#     if need_flip:
#         img, bboxes = flip_horizontally(C, img, bboxes)
#     if need_shift_incensity:
#         img = shift_intensity(img)
#     if need_shift_brightness:
#         img = shift_brightness(img)
        
    return img, bboxes

def get_full_ground_truth_data(bboxes, classes, scales, anchors):
    
    #bboxes is an array of arrays [[xmin,ymin,xmax,ymax,class_name]]

    img_width, img_height = 416, 416
    
    cells_full_data = []
    
    for scale in scales:
                
        scale_data = np.zeros((scale, scale, 75), dtype='float32') 
        cells_full_data.append(scale_data)
    
    for bbox_index, bbox in enumerate(bboxes):
        
        bbox_center_x = (bbox[2] + bbox[0]) // 2
        bbox_center_y = (bbox[3] + bbox[1]) // 2
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
          
        max_iou = 0
        best_scale = 0
        best_anchor_index = 0
    
        for scale_id, scale in enumerate(scales):
            
            fm_stride_x = 416 // scale
            fm_stride_y = 416 // scale

            x_fm = bbox_center_x // fm_stride_x
            y_fm = bbox_center_y // fm_stride_y

            xmin = x_fm*fm_stride_x
            xmax = (x_fm + 1)*fm_stride_x

            ymin = y_fm*fm_stride_y
            ymax = (y_fm + 1)*fm_stride_y 

            cell_coord = [xmin, ymin, xmax, ymax]    

            cell_center_x = (xmin + xmax) / 2
            cell_center_y = (ymin + ymax) / 2

            for anchor_index, anchor_size in enumerate(anchors[scale_id]):

                anchor_width = anchor_size[0]
                anchor_height = anchor_size[1]

                anchor_xmin = cell_center_x - anchor_width // 2
                anchor_xmax = cell_center_x + anchor_width // 2

                anchor_ymin = cell_center_y - anchor_height // 2 
                anchor_ymax = cell_center_y + anchor_height // 2

                anchor_center_x = (anchor_xmin + anchor_xmax) / 2
                anchor_center_y = (anchor_ymin + anchor_ymax) / 2

                anchor_coord = [anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax]

                iou = calculate_iou(anchor_coord, bbox[:4])

                if iou > max_iou:
                    max_iou = iou
                    best_scale = scale_id
                    best_anchor_index = anchor_index
        
        
        scale = scales[best_scale]
        
        fm_stride_x = 416 // scale
        fm_stride_y = 416 // scale
        
        x_fm = bbox_center_x // fm_stride_x
        y_fm = bbox_center_y // fm_stride_y
                
        x = (bbox_center_x - xmin) / fm_stride_x
        if x == 0: x = 0.0001
        delta_x = math.log(x/(1-x))

        y = (bbox_center_y - ymin) / fm_stride_y
        if y == 0: y = 0.0001
        delta_y = math.log(y/(1-y))

        best_anchor = anchors[best_scale][best_anchor_index]

        delta_width = math.log(bbox_width / (best_anchor[0])) 
        delta_height = math.log(bbox_height / (best_anchor[1]))

        #creating hot encoded class
        hot_encoded_class = np.zeros((20), dtype='float32')

        class_name = bbox[4]
        class_index = classes[class_name]

        hot_encoded_class[class_index] = 1
       
        try:
            cells_full_data[best_scale][x_fm, y_fm, best_anchor_index * 25 + 5:best_anchor_index * 25 + 5 + 20] = hot_encoded_class
            cells_full_data[best_scale][x_fm, y_fm, best_anchor_index * 25: best_anchor_index * 25 + 5] = [delta_x, delta_y, delta_width, delta_height, 1]
        except:
            print(NameError)
                
                
    return cells_full_data


def show(img, bboxes, data, anchors):
    
    fig,ax = plot.subplots(1)
    ax.imshow(img)
    
    for bbox in bboxes:
        
        x = bbox[0]
        y = bbox[1]
        
        width = bbox[2] - x
        height = bbox[3] - y
        
        
        bbox_center_x = (bbox[2] + bbox[0]) // 2
        bbox_center_y = (bbox[3] + bbox[1]) // 2
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        
        print(bbox_center_x)
        print(bbox_center_y)
        print(bbox_width)
        print(bbox_height)
        
        rect = patches.Rectangle((x,y), width, height, linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        
    for scale_id, scale in enumerate([13,26,52]):
    
        fm_stride_x = 416 // scale
        fm_stride_y = 416 // scale
        
        for x in range(scale):

            for y in range(scale):
                
                detectors = data[scale_id][x,y,:]

                for anchor_index, objectness in enumerate(detectors[4::25]):
                    
                    if objectness == 1:
                        print('best_scale ', scale_id)
                        print('best_anchor ', anchor_index)
                        scaled_anchors = anchors[scale_id]
                        
                        reshaped = detectors.reshape(3,25)
                        detector = reshaped[anchor_index]

                        delta_x = detector[0]
                        delta_y = detector[1]

                        delta_w = detector[2]
                        delta_h = detector[3]
                        
                        width = scaled_anchors[anchor_index][0] * math.exp(delta_w)
                        height = scaled_anchors[anchor_index][1] * math.exp(delta_h)

                        x_c = x * fm_stride_x + fm_stride_x // 2 - width // 2
                        y_c = y * fm_stride_x + fm_stride_x // 2 - height // 2

                        rect = patches.Rectangle((x_c,y_c), width, height, linewidth=1,edgecolor='g',facecolor='none')
                        ax.add_patch(rect)

                        print("x = ", x * fm_stride_x + (1 / (1 + math.exp(-delta_x))) * fm_stride_x)
                        print("y = ", y * fm_stride_y + (1 / (1 + math.exp(-delta_y))) * fm_stride_y)

                        print('width = ', scaled_anchors[anchor_index][0] * math.exp(delta_w))
                        print('height = ', scaled_anchors[anchor_index][1] * math.exp(delta_h))
                    
                    
def image_data_generator(images, batch_size, classes, strides, anchors):
    
    while True:
        
        random_files = np.random.choice(images, batch_size)
        
        input_batch = np.zeros((batch_size, 416, 416, 3), dtype='int32')
        

        output_batch_1 = np.zeros((batch_size, 52, 52, 75))
        output_batch_2 = np.zeros((batch_size, 26, 26, 75))        
        output_batch_3 = np.zeros((batch_size, 13, 13, 75))
        
        for file_index, file in enumerate(random_files):
            
            filename, file_extension = os.path.splitext(file)
    
            img, scaled_bboxes = get_img_bboxes(filename)  
            gt_data = get_full_ground_truth_data(scaled_bboxes, classes, strides, anchors)

            input_batch[file_index] = img
            
            output_batch_1[file_index] = gt_data[2]
            output_batch_2[file_index] = gt_data[1]
            output_batch_3[file_index] = gt_data[0]

        output_batch = [output_batch_3,output_batch_2,output_batch_1]

        yield (input_batch, output_batch)
        

# import tensorflow as tf

def iou(reshaped_true, reshaped_pred, anchors):
            
    y_true = reshaped_true[reshaped_true[:,:,:,:,4] > 0]
    y_pred = reshaped_pred[reshaped_true[:,:,:,:,4] > 0]
        
    true_y_center = 1 / (1 + K.exp(-y_true[:,:,:,:,1])
    pred_y_center = 1 / (1 + K.exp(-y_pred[:,:,:,:,1])
    
    true_w = anchors[:,0] * K.exp(y_true[:,:,:,:,2])
    pred_w = anchors[:,0] * K.exp(y_pred[:,:,:,:,2])
    
    true_h = anchors[0,:] * K.exp(y_true[:,:,:,:,3])
    pred_h = anchors[0,:] * K.exp(y_pred[:,:,:,:,3])
    
                         
#     with tf.Session() as sess:
#         print(sess.run())
       
                         
#     true_min_x = true_x_center - true_w / 2
#     true_max_x = true_x_center + true_w / 2
#     true_min_y = true_y_center - true_h / 2
#     true_max_y = true_y_center + true_h / 2
    
#     pred_min_x = pred_x_center - pred_w / 2
#     pred_max_x = pred_x_center + pred_w / 2
#     pred_min_y = pred_y_center - pred_h / 2
#     pred_max_y = pred_y_center + pred_h / 2
                         
#     for i in range()
    
#     #bboxes is an array of arrays [[xmin,ymin,xmax,ymax,class_name]]
        
#     xA = K.maximum(true_min_x, pred_min_x)
#     yA = K.maximum(true_min_y, pred_min_y)
    
#     xB = K.minimum(true_max_x, pred_max_x)
#     yB = K.minimum(true_max_y, pred_max_y)
 
#     interArea = K.maximum(0, xB - xA + 1) * K.maximum(0, yB - yA + 1)
 
#     boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
#     boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
#     iou = interArea / float(boxAArea + boxBArea - interArea)
 
#     return iou
    
        
def loss(batch_size, s, anchors):
    
    def real_loss(y_true, y_pred):
        
        coord_lambda = 5
        no_object_lambda = 0.5 
        
        loss = 0
        
        true_reshaped = K.reshape(y_true, (batch_size, s, s, 3, 25))
        pred_reshaped = K.reshape(y_pred, (batch_size, s, s, 3, 25))
        
        iou(true_reshaped, pred_reshaped, anchors)
                        
        mask = true_reshaped[:,:,:,:,4]
        
        long_mask = mask[:,:,:,:,None]
        
        #coord loss
        xy_square = long_mask*K.square(true_reshaped[:,:,:,:,:2] - pred_reshaped[:,:,:,:,:2])
        xy_loss = K.sum(xy_square)
        
        wh_square = long_mask*K.square(K.sqrt(true_reshaped[:,:,:,:,2:4]) - K.sqrt(pred_reshaped[:,:,:,:,2:4]))
        wh_loss = K.sum(wh_square)
        
        coord_loss = coord_lambda * (xy_loss + wh_loss) 
        
        #objectness loss 
        object_square = mask*K.square(mask - K.sigmoid(pred_reshaped[:,:,:,:,4]))
        object_loss = K.sum(object_square)

        no_object_mask = 1 - mask
        no_object_square = no_object_mask*K.square(0 - K.sigmoid(pred_reshaped[:,:,:,:,4]))
        no_object_loss = no_object_lambda*K.sum(no_object_square)
        
        objectness_loss = (no_object_loss + object_loss)
                
        #class loss
        class_xe = long_mask*K.binary_crossentropy(true_reshaped[:,:,:,:,5:], K.sigmoid(pred_reshaped[:,:,:,:,5:]))
        class_loss = K.sum(class_xe)
        
        with tf.Session() as sess:
            
            print (sess.run(mask[mask > 0]))
            print (sess.run(coord_loss))
            print (sess.run(object_loss))
            print (sess.run(no_object_loss))
            print (sess.run(class_loss))
        
        loss += (coord_loss + objectness_loss + class_loss)
        
        return loss
        
    return real_loss
