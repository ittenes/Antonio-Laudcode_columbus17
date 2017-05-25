import cv2
import numpy as np
import copy
import annolist.AnnotationLib as al
from imgaug import augmenters as iaa
import imgaug as ia
from scipy.ndimage.interpolation import rotate as imrotate


def annotation_to_h5(H, a):
    cell_width = H['grid_width']
    cell_height = H['grid_height']
    max_len = H['rnn_len']
    region_size = H['region_size']
    assert H['region_size'] == H['image_height'] / H['grid_height']
    assert H['region_size'] == H['image_width'] / H['grid_width']
    cell_regions = get_cell_grid(cell_width, cell_height, region_size)

    cells_per_image = len(cell_regions)

    box_list = [[] for idx in range(cells_per_image)]
            
    for cidx, c in enumerate(cell_regions):
        box_list[cidx] = [r for r in a.rects if all(r.intersection(c))]

    boxes = np.zeros((cells_per_image, max_len, 4), dtype = np.float)
    box_flags = np.zeros((cells_per_image, max_len), dtype = np.float)

    for cidx in xrange(cells_per_image):
        #assert(cur_num_boxes <= max_len)

        cell_ox = 0.5 * (cell_regions[cidx].x1 + cell_regions[cidx].x2)
        cell_oy = 0.5 * (cell_regions[cidx].y1 + cell_regions[cidx].y2)

        unsorted_boxes = []
        for bidx in xrange(min(len(box_list[cidx]), max_len)):

            # relative box position with respect to cell
            ox = 0.5 * (box_list[cidx][bidx].x1 + box_list[cidx][bidx].x2) - cell_ox
            oy = 0.5 * (box_list[cidx][bidx].y1 + box_list[cidx][bidx].y2) - cell_oy

            width = abs(box_list[cidx][bidx].x2 - box_list[cidx][bidx].x1)
            height= abs(box_list[cidx][bidx].y2 - box_list[cidx][bidx].y1)
            
            if (abs(ox) < H['focus_size'] * region_size and abs(oy) < H['focus_size'] * region_size and
                    width < H['biggest_box_px'] and height < H['biggest_box_px']):
                unsorted_boxes.append(np.array([ox, oy, width, height], dtype=np.float))

        for bidx, box in enumerate(sorted(unsorted_boxes, key=lambda x: x[0]**2 + x[1]**2)):
            boxes[cidx, bidx,  :] = box
            if H['num_classes'] <= 2:
                box_flags[cidx, bidx] = max(box_list[cidx][bidx].silhouetteID, 1)
            else: # multiclass detection
                # Note: class 0 reserved for empty boxes
                box_flags[cidx, bidx] = box_list[cidx][bidx].classID 

    return boxes, box_flags


def get_cell_grid(cell_width, cell_height, region_size):

    cell_regions = []
    for iy in xrange(cell_height):
        for ix in xrange(cell_width):
            cidx = iy * cell_width + ix
            ox = (ix + 0.5) * region_size
            oy = (iy + 0.5) * region_size

            r = al.AnnoRect(ox - 0.5 * region_size, oy - 0.5 * region_size,
                            ox + 0.5 * region_size, oy + 0.5 * region_size)
            r.track_id = cidx

            cell_regions.append(r)


    return cell_regions


def annotation_jitter(I, a_in, min_box_width=20, jitter_scale_min=0.9, jitter_scale_max=1.1, jitter_offset=16, target_width=640, target_height=480):
    a = copy.deepcopy(a_in)

    # MA: sanity check
    new_rects = []
    for i in range(len(a.rects)):
        r = a.rects[i]
        try:
            assert(r.x1 < r.x2 and r.y1 < r.y2)
            new_rects.append(r)
        except:
            print('bad rectangle')
    a.rects = new_rects


    if a.rects:
        cur_min_box_width = min([r.width() for r in a.rects])
    else:
        cur_min_box_width = min_box_width / jitter_scale_min

    # don't downscale below min_box_width 
    jitter_scale_min = max(jitter_scale_min, float(min_box_width) / cur_min_box_width)

    # it's always ok to upscale 
    jitter_scale_min = min(jitter_scale_min, 1.0)

    jitter_scale_max = jitter_scale_max

    jitter_scale = np.random.uniform(jitter_scale_min, jitter_scale_max)

    jitter_flip = np.random.random_integers(0, 1)

    if jitter_flip == 1:
        I = np.fliplr(I)

        for r in a:
            r.x1 = I.shape[1] - r.x1
            r.x2 = I.shape[1] - r.x2
            r.x1, r.x2 = r.x2, r.x1

            for p in r.point:
                p.x = I.shape[1] - p.x

    I1 = cv2.resize(I, None, fx=jitter_scale, fy=jitter_scale, interpolation = cv2.INTER_CUBIC)

    jitter_offset_x = np.random.random_integers(-jitter_offset, jitter_offset)
    jitter_offset_y = np.random.random_integers(-jitter_offset, jitter_offset)



    rescaled_width = I1.shape[1]
    rescaled_height = I1.shape[0]

    px = round(0.5*(target_width)) - round(0.5*(rescaled_width)) + jitter_offset_x
    py = round(0.5*(target_height)) - round(0.5*(rescaled_height)) + jitter_offset_y

    I2 = np.zeros((target_height, target_width, 3), dtype=I1.dtype)

    x1 = max(0, px)
    y1 = max(0, py)
    x2 = min(rescaled_width, target_width - x1)
    y2 = min(rescaled_height, target_height - y1)

    I2[0:(y2 - y1), 0:(x2 - x1), :] = I1[y1:y2, x1:x2, :]

    ox1 = round(0.5*rescaled_width) + jitter_offset_x
    oy1 = round(0.5*rescaled_height) + jitter_offset_y

    ox2 = round(0.5*target_width)
    oy2 = round(0.5*target_height)

    for r in a:
        r.x1 = round(jitter_scale*r.x1 - x1)
        r.x2 = round(jitter_scale*r.x2 - x1)

        r.y1 = round(jitter_scale*r.y1 - y1)
        r.y2 = round(jitter_scale*r.y2 - y1)

        if r.x1 < 0:
            r.x1 = 0

        if r.y1 < 0:
            r.y1 = 0

        if r.x2 >= I2.shape[1]:
            r.x2 = I2.shape[1] - 1

        if r.y2 >= I2.shape[0]:
            r.y2 = I2.shape[0] - 1

        for p in r.point:
            p.x = round(jitter_scale*p.x - x1)
            p.y = round(jitter_scale*p.y - y1)

        # MA: make sure all points are inside the image
        r.point = [p for p in r.point if p.x >=0 and p.y >=0 and p.x < I2.shape[1] and p.y < I2.shape[0]]

    new_rects = []
    for r in a.rects:
        if r.x1 <= r.x2 and r.y1 <= r.y2:
            new_rects.append(r)
        else:
            pass

    a.rects = new_rects

    return I2, a


class Augmentations(object):
    """
    The class is intended to organise augmentation processes.
    """
    def __init__(self, hypes):
        """Constructs instance of augmentations pipeline.
        Args:
            hypes (dict): Defines which augmentations to use.
            process_type (string): Defines the process type we wish to apply augmentations.
             Could be one of the following: train, predict_pre, predict_post.
        """
        # transforms factory
        transforms = {
            'rotate': iaa.Affine(rotate=(-5, 5)),
            'flip_lr': iaa.Fliplr(0.5),
            'blur': iaa.GaussianBlur(sigma=(0, 3.0))
        }

        # build pipeline using chosen transforms
        self.pipeline = []
        for hype, val in hypes.items():
            if hype in transforms and transforms[hype]:
                self.pipeline.append(transforms[hype])

    def process(self, image, rects):
        """Applies augmentation pipeline to images and bounding boxes.
        Args:
            image (object): The target image to rotate.
            rects (list): List of bounding boxes.
        Returns (tuple):
            Augmented images and bounding boxes
        """
        # rects -> keypoints
        keypoints = np.array(rects).reshape((-1, 2))
        keypoints = [ia.Keypoint(kp[0], kp[1]) for kp in keypoints]

        # transform images and keypoints
        seq = iaa.Sequential(self.pipeline)
        seq_det = seq.to_deterministic()
        images_aug = seq_det.augment_images([image])
        keypoints_aug = np.array(seq_det.augment_keypoints(keypoints)).reshape((-1, 2))

        # keypoints -> rects
        rects = [[min(k_pair[0].x, k_pair[1].x), min(k_pair[0].y, k_pair[1].y),
                  max(k_pair[0].x, k_pair[1].x), max(k_pair[0].y, k_pair[1].y)] for k_pair in keypoints_aug]
        return images_aug, rects


class Rotate90(object):
    @staticmethod
    def do(image, anno=None):
        """
        Does the rotation for image and rectangles for 90 degrees counterclockwise.
        Args:
            image (Image): The target image to rotate.
            anno (Annotation): The annotations to be rotated with the image.
        Returns (tuple):
            Rotated image and annotations for it.
        """
        w = image.shape[1]
        new_image = imrotate(image, 90, reshape=True)
        if anno is not None:
            anno.rects = [al.AnnoRect(r.y1, w - r.x2, r.y2, w - r.x1) for r in anno.rects]
        return new_image, anno

    @staticmethod
    def invert(width, rects):
        """Inverts the rotation for 90 degrees.
        Args:
            width (int): width of rotated image.
            rects (list): The list of rectangles on rotated image.
        Returns (list):
            The list of annotations for original image.
        """
        return [al.AnnoRect(width - r.y2, r.x1, width - r.y1, r.x2) for r in rects]
