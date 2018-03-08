import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution

LABEL_IN_COLOR = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                    [0, 64, 128]]



def paser_pascolvoc_instance_segmentation(segmentationObject_filename, segmentationClass_filename, NUM_CLASS = 21, NUM_OBJECT = 21 ):
    segmentationClass_data = tf.gfile.FastGFile(segmentationClass_filename,'rb').read()
    segmentationObject_data = tf.gfile.FastGFile(segmentationObject_filename,'rb').read()

    segmentationClass_img = tf.image.decode_png(segmentationClass_data,channels=3,dtype=tf.uint8)
    segmentationObject_img = tf.image.decode_png(segmentationObject_data,channels=3,dtype=tf.uint8)
    # print(segmentationObject_img.mode)


    # init instance vector
    # img_shape = tf.shape(segmentationObject_img)
    # num_class_tensor = tf.constant(NUM_CLASS)
    # img_shape_two_dim = tf.slice(img_shape,begin=0,size=2)
    # instance_shape = tf.concat([img_shape_two_dim, num_class_tensor], axis=0)

    # instance_vector = tf.constant(0,dtype=tf.uint8,shape=instance_shape)
    instance_vector = []
    instance_class = []

    instance_labels = tf.constant([],tf.int64)
    for i in range(NUM_OBJECT):
        # print(i, '/', NUM_OBJECT)
        compare_with_dim = tf.equal(segmentationObject_img,LABEL_IN_COLOR[i])
        mask_boolen = tf.reduce_all(input_tensor=compare_with_dim,axis=-1)
        mask_int = tf.cast(mask_boolen,dtype=tf.uint8)
        instance_vector.append(mask_int)

        instance_class_mask = tf.multiply(segmentationClass_img, mask_int)
        for j in range(NUM_CLASS):
            compare_label_dim = tf.equal(instance_class_mask,LABEL_IN_COLOR[j])
            compare_label  = tf.reduce_all(compare_with_dim, axis=-1)
            compare_label_final = tf.reduce_any(compare_label)

            instance_labels = tf.cond(compare_label_final, lambda:tf.concat([instance_labels,[j]],axis=0), lambda:instance_labels)

    instance_tensor = tf.convert_to_tensor(instance_vector)

    return instance_tensor, instance_labels




def paser_pascolvoc_instance_segmentation_np_version(segmentationObject_filename, segmentationClass_filename, NUM_CLASS = 21, NUM_OBJECT = 21 ):

    segmentationClass_img = Image.open(segmentationClass_filename)
    segmentationObject_img = Image.open(segmentationObject_filename)

    # print(segmentationClass_img.mode) #P模式，需转化为RGB模式

    segmentationClass_img = segmentationClass_img.convert('RGB')
    segmentationObject_img = segmentationObject_img.convert('RGB')

    segmentationClass_img = np.array(segmentationClass_img,dtype=np.uint8)
    segmentationObject_img = np.array(segmentationObject_img,dtype=np.uint8)

    instance_vector = []
    instance_labels = []

    for i in range(0,NUM_OBJECT):
        # print(i, '/', NUM_OBJECT)
        compare_with_dim = np.equal(segmentationObject_img,LABEL_IN_COLOR[i])
        mask_boolen = np.all(compare_with_dim,axis=-1)
        mask_int = mask_boolen.astype(np.uint8)
        mask_int_dim_expand = np.expand_dims(mask_int,axis=2)
        instance_vector.append(mask_int_dim_expand)

        mask_int = np.expand_dims(mask_int,axis=2)
        instance_class_mask = np.multiply(segmentationClass_img, mask_int)
        for j in range(1,NUM_CLASS):
            compare_label_dim = np.equal(instance_class_mask,LABEL_IN_COLOR[j])
            compare_label  = np.all(compare_label_dim, axis=-1)
            compare_label_final = np.any(compare_label)

            if compare_label_final:
                instance_labels.append((i,j))
    # im=Image.fromarray(instance_vector[2]*255)
    # im.save('./fig.png')

    # instance_vector = np.expand_dims(instance_vector,axis=3)
    instance_np_array = np.concatenate(instance_vector,axis=2)

    return instance_np_array, np.array(instance_labels)  # instance_tensor 中0是背景




def test(dataset_dir, name= '000032'):
    import os
    DIRECTORY_ANNOTATIONS = 'Annotations/'
    RANDOM_SEED = 4240
    SAMPLES_PER_FILES = 20000
    DIRECTORY_IMAGES = 'JPEGImages/'
    DIRECTORY_SEGMENTATIONCLASS = 'SegmentationClass/'
    DIRECTORY_SEGMENTATIONOBJECT = 'SegmentationObject/'

    IMAGE_TYPE = 'jpg'
    SEGMENTATION_TYPE = 'png'

    image_filename = os.path.join(dataset_dir, DIRECTORY_IMAGES, name + '.' + IMAGE_TYPE)
    segmentationClass_filename = os.path.join(dataset_dir, DIRECTORY_SEGMENTATIONCLASS, name + '.' + SEGMENTATION_TYPE)
    segmentationObject_filename = os.path.join(dataset_dir, DIRECTORY_SEGMENTATIONOBJECT,
                                               name + '.' + SEGMENTATION_TYPE)

    if os.path.exists(image_filename) and os.path.exists(segmentationObject_filename) and os.path.exists(
            segmentationClass_filename):
        instance_tensor, instance_labels = paser_pascolvoc_instance_segmentation_np_version(segmentationObject_filename,
                                                                                 segmentationClass_filename)


if __name__ == '__main__':
    dataset_dir = '/Volumes/TOSHIBA/Deep_Learning/All_Data/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/'
    test(dataset_dir)
    pass
