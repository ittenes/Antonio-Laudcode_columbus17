"""
This file is designed for prediction of bounding boxes for a single image.

Predictions could be made in two ways: command line style or service style. Command line style denotes that one can
run this script from the command line and configure all options right in the command line. Service style allows
to call :func:`initialize` function once and call :func:`hot_predict` function as many times as it needed to.

"""

import tensorflow as tf
import os, json, subprocess
from optparse import OptionParser

from scipy.misc import imread, imresize
import numpy as np
from PIL import Image, ImageDraw

from train import build_forward
from utils.annolist import AnnotationLib as al
from utils.train_utils import add_rectangles, rescale_boxes
from utils.data_utils import Rotate90

import Tkinter
from PIL import Image, ImageTk
import cv2
import subprocess

class Displayer:
    ison = True

    def __init__(self,ison):
        self.ison = ison
        if self.ison:
            # A root window for displaying objects
            self.root = Tkinter.Tk()
                # Convert the Image object into a TkPhoto object
            self.canvas = Tkinter.Canvas(self.root, height=2000, width=2500, bd=0, highlightthickness=0, relief='ridge')
            self.canvas.pack()

    def showifison(self, frame):
        if self.ison:
            b,g,r = cv2.split(frame)
            img2 = cv2.merge((r,g,b))
            im = Image.fromarray(img2)
            converted = ImageTk.PhotoImage(image=im)
            self.canvas.create_image(0, 0, image=converted, anchor=Tkinter.NW)

    def drawrectanglesifison(self,rects):
        if self.ison:
            for r in rects:
                self.canvas.create_rectangle(int(r.left()), int(r.top()), int(r.right()), int(r.bottom()), fill="blue")

    def enddrawingifison(self):
        if self.ison:
            self.root.update()


def initialize(weights_path, hypes_path, options=None):
    """Initialize prediction process.

    All long running operations like TensorFlow session start and weights loading are made here.

    Args:
        weights_path (string): The path to the model weights file.
        hypes_path (string): The path to the hyperparameters file.
        options (dict): The options dictionary with parameters for the initialization process.

    Returns (dict):
        The dict object which contains `sess` - TensorFlow session, `pred_boxes` - predicted boxes Tensor,
          `pred_confidences` - predicted confidences Tensor, `x_in` - input image Tensor,
          `hypes` - hyperparametets dictionary.
    """

    H = prepare_options(hypes_path, options)

    tf.reset_default_graph()
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
    if H['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas \
            = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        pred_confidences = tf.reshape(
            tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], H['num_classes']])),
            [grid_area, H['rnn_len'], H['num_classes']])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)

    saver = tf.train.Saver()
    sess = tf.Session()
    #sess.run(tf.initialize_all_variables())
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, weights_path)
    return {'sess': sess, 'pred_boxes': pred_boxes, 'pred_confidences': pred_confidences, 'x_in': x_in, 'hypes': H}


def hot_predict(image_path, parameters, to_json=True):
    """Makes predictions when all long running preparation operations are made.

    Args:
        image_path (string): The path to the source image.
        parameters (dict): The parameters produced by :func:`initialize`.

    Returns (Annotation):
        The annotation for the source image.
    """

    H = parameters['hypes']
    # The default options for prediction of bounding boxes.
    options = H['evaluate']
    if 'pred_options' in parameters:
        # The new options for prediction of bounding boxes
        for key, val in parameters['pred_options'].items():
            options[key] = val

    # predict
    orig_img = imread(image_path)[:, :, :3]
    img = Rotate90.do(orig_img) if 'rotate90' in H['data'] and H['data']['rotate90'] else orig_img
    img = imresize(img, (H['image_height'], H['image_width']), interp='cubic')
    np_pred_boxes, np_pred_confidences = parameters['sess']. \
        run([parameters['pred_boxes'], parameters['pred_confidences']], feed_dict={parameters['x_in']: img})

    image_info = {'path': image_path, 'original': orig_img, 'transformed': img}
    pred_anno = postprocess(image_info, np_pred_boxes, np_pred_confidences, H, options)
    result = [r.writeJSON() for r in pred_anno] if to_json else pred_anno
    return result

def hot_predict_img(image, parameters, to_json=True):
    """Makes predictions when all long running preparation operations are made.

    Args:
        image_path (string): The path to the source image.
        parameters (dict): The parameters produced by :func:`initialize`.

    Returns (Annotation):
        The annotation for the source image.
    """

    H = parameters['hypes']
    # The default options for prediction of bounding boxes.
    options = H['evaluate']
    if 'pred_options' in parameters:
        # The new options for prediction of bounding boxes
        for key, val in parameters['pred_options'].items():
            options[key] = val

    # predict
    orig_img = (image)[:, :, :3]
    img = Rotate90.do(orig_img) if 'rotate90' in H['data'] and H['data']['rotate90'] else orig_img
    img = imresize(img, (512, 512), interp='cubic')
    np_pred_boxes, np_pred_confidences = parameters['sess']. \
        run([parameters['pred_boxes'], parameters['pred_confidences']], feed_dict={parameters['x_in']: img})

    image_info = { 'original': orig_img, 'transformed': img}
    pred_anno = postprocess(image_info, np_pred_boxes, np_pred_confidences, H, options)
    result = [r.writeJSON() for r in pred_anno] if to_json else pred_anno
    return result

def postprocess(image_info, np_pred_boxes, np_pred_confidences, H, options):
    pred_anno = al.Annotation()
    #pred_anno.imageName = image_info['path']
    #pred_anno.imagePath = os.path.abspath(image_info['path'])
    _, rects = add_rectangles(H, [image_info['transformed']], np_pred_confidences, np_pred_boxes, use_stitching=True,
                              rnn_len=H['rnn_len'], min_conf=options['min_conf'], tau=options['tau'],
                              show_suppressed=False)

    rects = [r for r in rects if r.x1 < r.x2 and r.y1 < r.y2 and r.score > options['min_conf']]
    h, w = image_info['original'].shape[:2]
    if 'rotate90' in H['data'] and H['data']['rotate90']:
        # original image height is a width for roatated one
        rects = Rotate90.invert(h, rects)
    pred_anno.rects = rects
    pred_anno = rescale_boxes((H['image_height'], H['image_width']), pred_anno, h, w)
    return pred_anno


def prepare_options(hypes_path='hypes.json', options=None):
    """Sets parameters of the prediction process. If evaluate options provided partially, it'll merge them.
    The priority is given to options argument to overwrite the same obtained from the hyperparameters file.

    Args:
        hypes_path (string): The path to model hyperparameters file.
        options (dict): The command line options to set before start predictions.

    Returns (dict):
        The model hyperparameters dictionary.
    """

    with open(hypes_path, 'r') as f:
        H = json.load(f)

    # set default options values if they were not provided
    if options is None:
        if 'evaluate' in H:
            options = H['evaluate']
        else:
            print ('Evaluate parameters were not found! You can provide them through hyperparameters json file '
                   'or hot_predict options parameter.')
            return None
    else:
        if 'evaluate' not in H:
            H['evaluate'] = {}
        # merge options argument into evaluate options from hyperparameters file
        for key, val in options.items():
            H['evaluate'][key] = val

    os.environ['CUDA_VISIBLE_DEVICES'] = str(H['evaluate']['gpu'])
    return H


def save_results(image_path, anno):
    """Saves results of the prediction.

    Args:
        image_path (string): The path to source image to predict bounding boxes.
        anno (Annotation): The predicted annotations for source image.

    Returns:
        Nothing.
    """

    # draw
    new_img = Image.open(image_path)
    d = ImageDraw.Draw(new_img)
    rects = anno['rects'] if type(anno) is dict else anno.rects
    for r in rects:
        d.rectangle([r.left(), r.top(), r.right(), r.bottom()], outline=(255, 0, 0))

    # save
    fpath = os.path.join(os.path.dirname(image_path), 'result.png')
    new_img.save(fpath)
    subprocess.call(['chmod', '777', fpath])

    fpath = os.path.join(os.path.dirname(image_path), 'result.json')
    if type(anno) is dict:
        with open(fpath, 'w') as f:
            json.dump(anno, f)
    else:
        al.saveJSON(fpath, anno)
    subprocess.call(['chmod', '777', fpath])


def save_results_but_not_image(image_path, anno):
    """Saves results of the prediction.

    Args:
        image_path (string): The path to source image to predict bounding boxes.
        anno (Annotation): The predicted annotations for source image.

    Returns:
        Nothing.
    """

    fpath = os.path.join(os.path.dirname(image_path), 'result.json')
    if type(anno) is dict:
        with open(fpath, 'w') as f:
            json.dump(anno, f)
    else:
        al.saveJSON(fpath, anno)
    subprocess.call(['chmod', '777', fpath])

#SERVER IMG
def recvall(s, count):
    buf = b''
    while count:
        newbuf = s.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def main():
    parser = OptionParser(usage='usage: %prog [options] <image> <weights> <hypes>')
    parser.add_option('--gpu', action='store', type='int', default=0)
    parser.add_option('--tau', action='store', type='float', default=0.25)
    parser.add_option('--min_conf', action='store', type='float', default=0.2)

    (options, args) = parser.parse_args()
    if len(args) < 3:
        print ('Provide image, weights and hypes paths')
        return

    init_params = initialize(args[1], args[2], options.__dict__)

    displayer = Displayer(True)

    # webcam
    # cap = cv2.VideoCapture(0)
    # if cap.isOpened() == 0:
    #    cap.open(0)

    #SERVER
    import socket
    import numpy

    def recvall(sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    TCP_IP = '192.168.1.36'
    TCP_PORT = 5001

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(True)
    conn, addr = s.accept()
    print 'Connected by', addr



    a=0
    while(True):
        # webcam
        #ret, frame = cap.read()
        #displayer.showifison(frame)

        #SERVER img
        length = recvall(conn,16)
        if length != None:
            stringData = recvall(conn, int(length))
            data = numpy.fromstring(stringData, dtype='uint8')

            decimg=cv2.imdecode(data,1)

            cv2.imwrite("./data/00.jpg", decimg)
            #conn.send(str(a))
            a = a+1

            # IA
            pred_anno = hot_predict_img(decimg, init_params, False)
            # OUTPUT IA
            print("---");
            rects = pred_anno['rects'] if type(pred_anno) is dict else pred_anno.rects
            for r in rects:
                print(r.left(), r.top(), r.right(), r.bottom())
                conn.send(  str(int(r.left())) +"_"+
                            str(int(r.top())) +"_"+
                            str(int(r.right())) +"_"+
                            str(int(r.bottom())) + ":")



        #displayer.drawrectanglesifison(rects)
        #displayer.enddrawingifison()

        # SERVER


    s.close()




if __name__ == '__main__':
    main()
