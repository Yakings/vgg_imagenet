import sys
import tensorflow as tf
import os
import random
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET

from segmentation.utils import dataset_util, paser_pascolvoc
from segmentation.utils.paser_pascolvoc import paser_pascolvoc_instance_segmentation, paser_pascolvoc_instance_segmentation_np_version

print(os.getcwd())

'''
###############################
 write data to tfrecord files
##############################
'''

DIRECTORY_ANNOTATIONS = 'Annotations/'
RANDOM_SEED = 4240
SAMPLES_PER_FILES = 1000
DIRECTORY_IMAGES = 'JPEGImages/'
DIRECTORY_SEGMENTATIONCLASS = 'SegmentationClass/'
DIRECTORY_SEGMENTATIONOBJECT = 'SegmentationObject/'

IMAGE_TYPE = 'jpg'
SEGMENTATION_TYPE = 'png'

ANNOTATION_TYPE = 'xml'

LABELS ={
    'none':(0, 'Background')
}

# write one piece data to tfrecord
#       using tfrecord_writer, from dataset_dir+name
def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):

    # conbine to path
    image_filename = os.path.join(dataset_dir, DIRECTORY_IMAGES, name + '.' + IMAGE_TYPE)
    segmentationClass_filename = os.path.join(dataset_dir,DIRECTORY_SEGMENTATIONCLASS,name + '.' + SEGMENTATION_TYPE)
    # annotation_filename = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS, name + '.' + ANNOTATION_TYPE)
    segmentationObject_filename = os.path.join(dataset_dir, DIRECTORY_SEGMENTATIONOBJECT, name + '.' + SEGMENTATION_TYPE)

    if os.path.exists(image_filename) and os.path.exists(segmentationObject_filename) and os.path.exists(segmentationClass_filename):

        instance_tensor, instance_labels = paser_pascolvoc_instance_segmentation(segmentationObject_filename, segmentationClass_filename)

        instance_tensor,instance_labels = tf.Session().run([instance_tensor,instance_labels])

        # change numpy to bytes, so that use tf.train.BytesList([]) to save to tfrecord
        instance_tensor_row = instance_tensor.tobytes()
        instance_labels_row = instance_labels.tobytes()

        img_pil = Image.open(image_filename)

        pil_w, pil_h = img_pil.size
        img_height = int(pil_h)
        img_width = int(pil_w)
        shape = [img_height, img_width]
        # img_np = np.array(img_pil,dtype=np.uint8)
        # image_data = img_np.tobytes()

        # Read image
        image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
        # image_data.tobytes() #if image_data is Image or Numpy object

        #####################################
        # # Read XML annotation file
        # tree = ET.parse((annotation_filename))
        # root = tree.getroot()
        # img_height = int(root.find('./size/height').text)
        # img_width = int(root.find('./size/height').text)
        #
        # # Image shape
        # shape = [img_height, img_width]
        #
        # # find annotations
        # bboxes = []
        # labels = []
        # labels_text = []
        # difficult = []
        # truncated = []
        # xmins = []
        # ymins = []
        # xmaxs = []
        # ymasx = []
        #
        # # read all objects from
        # # xml files
        # for obj in root.findall('object'):
        #     class_label = obj.find('name').text
        #     label = class_label
        #     labels.append(int(LABELS[label][0]))
        #     labels_text.append(label.encode('utf8'))
        #
        #     difficult.append(0)
        #     truncated.append(0)
        #
        #     bbox = obj.find('bndbox')
        #
        #     ymin = float(bbox.find('ymin').text)/shape[0]
        #     xmin = float(bbox.find('xmin').text)/shape[1]
        #     if xmin < 0.0:
        #         xmin = 0
        #     if ymin < 0.0:
        #         ymin = 0
        #     ymax = float(bbox.find('ymax').text) / shape[0]
        #     xmax = float(bbox.find('xmax').text) / shape[1]
        #     if xmin > 1:
        #         xmin = 1
        #     if ymin >1:
        #         ymin = 1
        ###################################

        filename = bytes(image_filename, encoding='utf8')
        image_format = IMAGE_TYPE.encode('utf8')

        example = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    # 'image/height':tf.train.Feature(int64_list=tf.train.Int64List(values=[shape[0]])),
                    'image/height': dataset_util.int64_feature(shape[0]),
                    'image/width':dataset_util.int64_feature(shape[1]),

                    'image/filename':dataset_util.bytes_feature(filename),
                    'image_raw':dataset_util.bytes_feature(image_data),

                    # 'instance_segmentation':dataset_util.int64_list_feature(instance_tensor),
                    # 'instance_class':dataset_util.int64_list_feature(instance_labels)

                    'instance_segmentation': dataset_util.bytes_feature(instance_tensor_row),
                    'instance_class': dataset_util.bytes_feature(instance_labels_row)

                })
        )
        tfrecord_writer.write(example.SerializeToString())
    pass


def _add_to_tfrecord_np(dataset_dir, name, tfrecord_writer):

    # conbine to path
    image_filename = os.path.join(dataset_dir, DIRECTORY_IMAGES, name + '.' + IMAGE_TYPE)
    segmentationClass_filename = os.path.join(dataset_dir,DIRECTORY_SEGMENTATIONCLASS,name + '.' + SEGMENTATION_TYPE)
    # annotation_filename = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS, name + '.' + ANNOTATION_TYPE)
    segmentationObject_filename = os.path.join(dataset_dir, DIRECTORY_SEGMENTATIONOBJECT, name + '.' + SEGMENTATION_TYPE)

    if os.path.exists(image_filename) and os.path.exists(segmentationObject_filename) and os.path.exists(segmentationClass_filename):

        instance_np_array, instance_labels = paser_pascolvoc_instance_segmentation_np_version(segmentationObject_filename, segmentationClass_filename)

        # change numpy to bytes, so that use tf.train.BytesList([]) to save to tfrecord
        instance_np_row = instance_np_array.tobytes()
        instance_labels_row = instance_labels.tobytes()

        img_pil = Image.open(image_filename)

        pil_w, pil_h = img_pil.size
        img_height = int(pil_h)
        img_width = int(pil_w)
        shape = [img_height, img_width]
        # print(shape)
        # img_np = np.array(img_pil,dtype=np.uint8)
        # image_data = img_np.tobytes()

        # Read image
        image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
        # image_data.tobytes() #if image_data is Image or Numpy object


        filename = bytes(image_filename, encoding='utf8')
        image_format = IMAGE_TYPE.encode('utf8')

        example = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    # 'image/height':tf.train.Feature(int64_list=tf.train.Int64List(values=[shape[0]])),
                    'image/height': dataset_util.int64_feature(shape[0]),
                    'image/width':dataset_util.int64_feature(shape[1]),

                    'image/filename':dataset_util.bytes_feature(filename),
                    'image_raw':dataset_util.bytes_feature(image_data),

                    # 'instance_segmentation':dataset_util.int64_list_feature(instance_tensor),
                    # 'instance_class':dataset_util.int64_list_feature(instance_labels)

                    'instance_segmentation': dataset_util.bytes_feature(instance_np_row),
                    'instance_class': dataset_util.bytes_feature(instance_labels_row)

                })
        )
        tfrecord_writer.write(example.SerializeToString())
    pass










def _get_output_filename(output_dir, name, idx):
    print(output_dir,name,idx)
    return '%s/%s%03d.tfrecord' % (output_dir, name, idx)

def run(dataset_dir, output_dir, file_prefix = 'train', shuffling = True):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    path = os.path.join(dataset_dir,DIRECTORY_IMAGES)
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    i = 0
    fidx = 0
    while i < len(filenames):

        tf_filename = _get_output_filename(output_dir, file_prefix, fidx)
        print(tf_filename)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i <len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> converting image %d/%d'%(i+1,len(filenames)))
                sys.stdout.flush()
                print('\n')

                filename = filenames[i]
                img_name = filename[:-4]
                _add_to_tfrecord_np(dataset_dir, img_name, tfrecord_writer)
                i += 1
                j += 1
            fidx +=1

        print('\nDataset converted.')

def test():
    dataset_dir2 = r'./sample'
    # dataset_dir = r'/home/store-1-img/VOC2007/VOCdevkit/VOC2012/'
    dataset_dir = r'/Volumes/TOSHIBA/Deep_Learning/All_Data/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/'
    run(dataset_dir=dataset_dir, output_dir=dataset_dir2, file_prefix='train', shuffling=False)

if __name__ =='__main__':
    if len(sys.argv) == 3:
        dataset_dir, output_dir = sys.argv[1], sys.argv[2]
        run(dataset_dir, output_dir)
    elif len(sys.argv) == 4:
        dataset_dir, output_dir, file_prefix = sys.argv[1], sys.argv[2], sys.argv[3]
        run(dataset_dir, output_dir, file_prefix)
    else:
        print("wrong number of arguments")
