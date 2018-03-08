import tensorflow as tf
# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution

def get_tfrecord_tensor(tfrecord_list):
    filename_queue = tf.train.string_input_producer(string_tensor=tfrecord_list,
                                                    num_epochs=None,
                                                    shuffle=True,
                                                    seed=None,
                                                    capacity=2,
                                                    shared_name=None,
                                                    name=None,
                                                    cancel_op=None
                                                    )
    tf_reader = tf.TFRecordReader()

    _, serialized_exampal = tf_reader.read(queue=filename_queue)
    features = tf.parse_single_example(serialized=serialized_exampal,
                                       features={
                                           'image/height':
                                               tf.FixedLenFeature([], tf.int64, 1),
                                           'image/width':
                                               tf.FixedLenFeature([], tf.int64, 1),
                                           'image_raw':
                                               tf.FixedLenFeature([],tf.string),
                                           'instance_segmentation':
                                               tf.FixedLenFeature([],tf.string),
                                           'instance_class':
                                               tf.FixedLenFeature([],tf.string)
                                       })

    image_height = tf.cast(features['image/height'], tf.int64)
    image_width = tf.cast(features['image/width'], tf.int64)
    image_shape = tf.stack([image_height,image_width, tf.constant(3,dtype=tf.int64)],axis=0)
    # [281, 500]

    #if use   tf.gfile.FastGFile(image_filename, 'rb').read() to read picture and save to tfrecord, use this
    image = tf.image.decode_jpeg(contents=features['image_raw'],channels=3)
    image_tensor = image

    # if use numpy to read and save picture, use this
    # image = tf.decode_raw(bytes=features['image_raw'],out_type=tf.uint8)
    # image_tensor = tf.reshape(image,shape=image_shape)

    # shpe: [channel,height,width]
    instance_shape = tf.stack([image_height,image_width,tf.constant(21,dtype=tf.int64),],axis=0)
    instance_segmentation = tf.decode_raw(bytes=features['instance_segmentation'],out_type=tf.uint8)
    instance_segmentation_tensor = tf.reshape(instance_segmentation,shape=instance_shape)


    instance_class = tf.decode_raw(bytes=features['instance_class'],out_type=tf.int64)
    # instance_class_tensor = tf.reshape(instance_class,shape=[4,2])
    instance_class_tensor = instance_class

    # do something to image
    # debug = image

    return image_tensor,instance_segmentation_tensor, instance_class_tensor # , debug


def get_tfrecord_batch_tensor(tfrecord_list,batch_size):
    image_tensor, instance_segmentation_tensor, instance_class_tensor = get_tfrecord_tensor(tfrecord_list=tfrecord_list)
    image_batch_tensor, instance_batch_segmentation_tensor, instance_batch_class_tensor = tf.train.shuffle_batch(tensors = [image_tensor, instance_segmentation_tensor, instance_class_tensor], batch_size= batch_size,capacity=10, min_after_dequeue=3)

    return image_batch_tensor, instance_batch_segmentation_tensor, instance_batch_class_tensor




# Transforms a scalar string `example_proto` into a pair of a scalar string and
# a scalar integer, representing an image and its label, respectively.
def _parse_function(example_proto):
    features={    'image/height':
                   tf.FixedLenFeature([], tf.int64, 1),
               'image/width':
                   tf.FixedLenFeature([], tf.int64, 1),
               'image_raw':
                   tf.FixedLenFeature([],tf.string),
               'instance_segmentation':
                   tf.FixedLenFeature([],tf.string),
               'instance_class':
                   tf.FixedLenFeature([],tf.string)
           }
    parsed_features = tf.parse_single_example(example_proto, features)

    image_height = tf.cast(parsed_features['image/height'], tf.int64)
    image_width = tf.cast(parsed_features['image/width'], tf.int64)
    image_shape = tf.stack([image_height, image_width, tf.constant(3, dtype=tf.int64)], axis=0)
    # [281, 500]

    # if use   tf.gfile.FastGFile(image_filename, 'rb').read() to read picture and save to tfrecord, use this
    image = tf.image.decode_jpeg(contents=parsed_features['image_raw'], channels=3)
    image_tensor = image

    # if use numpy to read and save picture, use this
    # image = tf.decode_raw(bytes=parsed_features['image_raw'],out_type=tf.uint8)
    # image_tensor = tf.reshape(image,shape=image_shape)

    # shpe: [channel,height,width]
    instance_shape = tf.stack([image_height, image_width, tf.constant(21, dtype=tf.int64), ], axis=0)
    instance_segmentation = tf.decode_raw(bytes=parsed_features['instance_segmentation'], out_type=tf.uint8)
    instance_segmentation_tensor = tf.reshape(instance_segmentation, shape=instance_shape)

    instance_class = tf.decode_raw(bytes=parsed_features['instance_class'],out_type=tf.int64)
    # instance_class_tensor = tf.reshape(instance_class,shape=[4,2])
    instance_class_tensor = instance_class


    # do something to image
    # debug = image

    return image_tensor,instance_segmentation_tensor, instance_class_tensor # , debug
    # return parsed_features["image"], parsed_features["label"]

def get_tfrecord_tensor_tf_dataset(tfrecord_list, batch_size=16):
    dataset = tf.data.TFRecordDataset(tfrecord_list)
    dataset = dataset.map(map_func = _parse_function,num_parallel_calls=None)
    NUM_EPOCHS = 50
    dataset = dataset.repeat(count=NUM_EPOCHS)

    # batched_dataset = dataset.batch(4)
    batched_dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=([None,None,3],[None,None,21],[None,]), padding_values=None)



##########################################################################
    # without eager mode
    iterator = batched_dataset.make_one_shot_iterator()
    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    image_batch_tensor, \
    instance_batch_segmentation_tensor, \
    instance_batch_class_tensor = iterator.get_next()
#########################################################################
    # eager mode


    # iterator = tfe.Iterator(batched_dataset)
    # image_batch_tensor, \
    # instance_batch_segmentation_tensor, \
    # instance_batch_class_tensor = iterator.next()

#######################################################################

    return image_batch_tensor, \
           instance_batch_segmentation_tensor, \
           instance_batch_class_tensor


def test_function():
    dataset_dir = r'./sample'
    filename = dataset_dir+'/train000.tfrecord'
    image_tensor, instance_segmentation_tensor, instance_class_tensor= get_tfrecord_tensor_tf_dataset([filename])
    sess = tf.Session()


    # when use tf.train.string_input_producer, should run start queue
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # debug = sess.run([debug])
    image, instance_segmentation, instance_class = sess.run([image_tensor,instance_segmentation_tensor, instance_class_tensor])

    show = instance_segmentation[6,:,:,1]*255
    # show = image[6]
    print(show[200])

    from PIL import Image
    img = Image.fromarray(show)
    img.save('./img.jpg')
    # print(instance_class)
    pass

if __name__ == '__main__':
    test_function()
    # get_tfrecord_tensor_tf_dataset(['./sample/train000.tfrecord'])
