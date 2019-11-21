import numpy as np
import matplotlib.pyplot as plot
import matplotlib.patches as patches

SCALES = 3

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def process_output(output, anchors, objectness_thresh, img_w, img_h):

    """
        requirements for inputs:

         output should be a 3d tensor of shape
         (grid height X grid width X [SCALES * {4 + 1 + number of classes}])
         where SCALES is the number of scales at which we work,
         4 is the number of box coordinates, 1 is objectness

         anchors should be a 2d tensor of shape (SCALES, 2),
         because there are two anchors for bounding boxes:
         height and width
    """

    assert len(output.shape) == 3, "output is not a 3d tensor"
    assert anchors.shape == (3,2), "anchors have wrong shape"

    """
        get grid height and width
    """

    grid_h, grid_w = output.shape[:2]

    """
        reshape output so that third dimension is the scale
         and fourth is coords + objectness + classes
    """

    scaled_output = output.reshape((grid_h, grid_w, SCALES, -1))

    """
        unpack coordinates, objectness and class probabilities
    """

    tx = scaled_output[:,:,:,0]
    ty = scaled_output[:,:,:,1]
    tw = scaled_output[:,:,:,2]
    th = scaled_output[:,:,:,3]

    obj = sigmoid(scaled_output[:,:,:,4])

    classes = scaled_output[:,:,:,5:]

    """
        initialize cx and cy offsets of a grid like:

                    grid_w
        +-----------------------------
        | cx = 0       | cx = 1
        | cy = 0       | cy = 0
 grid_h +--------------+--------------
        | cx = 0       | ...
        | cy = 1       |

    """

    cx_vector = np.arange(grid_w)
    cy_vector = np.arange(grid_h)

    cx = np.tile(cx_vector, [grid_h, 1])
    cy = np.tile(cy_vector, [grid_w, 1]).T

    """
        calculate relative centered coordinates of all boxes
    """

    bx = (sigmoid(tx) + cx[:,:,None]) / grid_w
    by = (sigmoid(ty) + cy[:,:,None]) / grid_h
    bw = anchors[None, None, :, 0] * np.exp(tw) / img_w
    bh = anchors[None, None, :, 1] * np.exp(th) / img_h

    """
        make bounding boxes
    """

    f = obj > objectness_thresh
    boxes = np.concatenate((bx[f][:,None], by[f][:,None], bw[f][:,None], bh[f][:,None], classes[f]), axis=1)
#     print(obj)
    return boxes

def show(img, boxes, class_names):

    """
        requirements:

         img is a numpy array hight X width X channels

         boxes is an output from process_output

         class_names is a list of class names in their correct
         order
    """

    """
        show image on plot
    """

    fig,ax = plot.subplots(1)
    ax.imshow(img)

    img_h, img_w = img.shape[:2]

    for box in boxes:
        """
            unpack and convert centered relative coordinates
             into absolute
        """

        center_x_rel, center_y_rel, width_rel, height_rel = box[:4]
        classes = box[4:]

        width = width_rel * img_w
        height = height_rel * img_h

        x = center_x_rel * img_w - width / 2
        y = center_y_rel * img_h - height / 2

        """
            draw bounding boxes and class names
        """

        rect = patches.Rectangle((x,y), width, height, linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        plot.text(x,y,class_names[classes.argmax()], color='m')
