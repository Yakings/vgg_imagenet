import os
import tensorflow as tf
# tf.data.Dataset.from_tensor_slices()
def read_filename_label_to_tuple(data_dir):
    '''
    read all files path and label
    :param data_dir: train dataset root path
    :return:  [ (filepath,label), (...,...), ...]
    '''
    label_names_bad = os.listdir(data_dir)

    label_names = []
    for label in label_names_bad:
        full_label = os.path.join(data_dir, label)
        if os.path.isdir(full_label):
            label_names.append(label)
    name_labels = []
    counter = 0
    # number_to_label = {0:'none'}
    number_to_label = {}
    for label in label_names[:30]:
        full_label = os.path.join(data_dir,label)
        if os.path.isdir(full_label):
            file_names = os.listdir(full_label)
            for name in file_names:
                full_name = os.path.join(full_label,name)
                name_labels.append((full_name,counter))
            number_to_label[counter] = label
            counter += 1
    return name_labels, number_to_label
    pass


def _read_function(image_path,label):
    '''
    read data from path  (will be used in get_input_tensor_tf_dataset() )
    :param image_path, labela;
    :return:
    '''
    # image_path = data[0]
    # label = data[1]
    # print(image_path,label)
    # image_string  = tf.gfile.FastGFile(image_path,'rb').read()
    image_string  = tf.read_file(image_path)
    image_tensor = tf.image.decode_jpeg(image_string,channels=3,name='image_tensor')
    image_tensor = tf.image.resize_images(image_tensor,[224,224])
    # print(label)
    # label_tensor = tf.constant(label,name='label_tensor')
    label_tensor = tf.cast(label,dtype=tf.int32)
    # print(label_tensor)

    return image_tensor,label_tensor # , debug
    # return parsed_features["image"], parsed_features["label"]


def get_input_tensor_tf_dataset(data_list, batch_size=72, NUM_EPOCHS = 5000):
    '''

    :param data_list:
    :param batch_size:
    :param NUM_EPOCHS:
    :return:
    '''
    dataset = tf.data.Dataset.from_tensor_slices(data_list)

    dataset = dataset.map(map_func = _read_function,num_parallel_calls=16)

    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # dataset = dataset.repeat(count=NUM_EPOCHS)
    # dataset = dataset.repeat()
    # dataset = dataset.shuffle(buffer_size=256)

    # batched_dataset = dataset.batch(batch_size)
    # batched_dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=([None,None,3],[]), padding_values=None)

    batched_dataset = dataset

##########################################################################
    # without eager mode
    iterator = batched_dataset.make_one_shot_iterator()
    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    image_batch_tensor, label_tensor = iterator.get_next()
#########################################################################
    # eager mode

    # iterator = tfe.Iterator(batched_dataset)
    # image_batch_tensor, \
    # instance_batch_segmentation_tensor, \
    # instance_batch_class_tensor = iterator.next()

#######################################################################

    return image_batch_tensor, \
           label_tensor

if __name__ == '__main__':
    import numpy as np
    data_dir = '/Volumes/TOSHIBA/Deep_Learning/All_Data/ImageNet/ILSVRC2012/ImageNet2012Train/'
    name_labels, number_to_label = read_filename_label_to_tuple(data_dir=data_dir)
    # print(name_labels)
    # print(number_to_label)
    # image_tensor, label_tensor = _read_function(name_labels[0][0],name_labels[0][1])
    name_labels_np = np.array(name_labels)
    name_np = name_labels_np[:,0]
    label_np = name_labels_np[:,1]
    label_np = np.array(label_np,dtype=np.int32)
    print(label_np)
    image_batch_tensor, label_tensor = get_input_tensor_tf_dataset(data_list= (name_np,label_np))
    pass