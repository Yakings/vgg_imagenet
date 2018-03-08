from __future__ import print_function
import tensorflow as tf
import numpy as np
import tensorflow_net_utils
import datetime
import os
from PIL import Image

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# os.environ['CUDA_VISIBLE_DEVICES'] = ''

FLAGS = tf.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)

tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "segmentation/Model_zoo/", "Path to vgg model mat")
# tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")

tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
# tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
# MAX_ITERATION = 1
NUM_OF_CLASSESS = 21

MIDDLE_CHANNEL = 19
IMAGE_SIZE = 224


class InstanceSegmentation():
    def __init__(self):
        self.delta = 1

    def vgg_net(self, weights, image):
        layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        )
        net = {}
        current = image
        for i, name in enumerate(layers):
            print(i, name)
            kind = name[:4]
            if kind == 'conv':
                kernels, bias = weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = tensorflow_net_utils.get_variable(
                    np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
                bias = tensorflow_net_utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = tensorflow_net_utils.conv2d_basic(current, kernels, bias)
            elif kind == 'relu':
                current = tf.nn.relu(current, name=name)
                if FLAGS.debug:
                    tensorflow_net_utils.add_activation_summary(current)
            elif kind == 'pool':
                current = tensorflow_net_utils.avg_pool_2x2(current)
            net[name] = current

        return net

    def inference(self, image, keep_prob):
        """
        Semantic segmentation network definition
        :param image: input image. Should have values in range 0-255
        :param keep_prob:
        :return:
        """
        print("setting up vgg initialized conv layers ...")

        print('model path:', FLAGS.model_dir)
        model_data = tensorflow_net_utils.get_model_data(FLAGS.model_dir, MODEL_URL)
        # print(model_data)
        mean = model_data['normalization'][0][0][0]
        mean_pixel = np.mean(mean, axis=(0, 1))

        weights = np.squeeze(model_data['layers'])

        # processed_image = tensorflow_net_utils.process_image(image, mean_pixel)
        processed_image = image

        with tf.variable_scope("inference"):
            image_net = self.vgg_net(weights, processed_image)
            conv_final_layer = image_net["conv5_3"]

            # middle_vector = tf.nn.tanh(conv8)
            ###########################################
            vector_layer = image_net["conv4_3"]
            middle_vector_before = vector_layer
            W_before = tensorflow_net_utils.weight_variable([1, 1, 512, MIDDLE_CHANNEL], name="W_before")
            b_before = tensorflow_net_utils.bias_variable([MIDDLE_CHANNEL], name="b_before")
            conv_before = tensorflow_net_utils.conv2d_basic(middle_vector_before,
                                                            W_before, b_before)
            middle_vector = conv_before
            ################################################
            pool5 = tensorflow_net_utils.max_pool_2x2(conv_final_layer)

            W6 = tensorflow_net_utils.weight_variable([7, 7, 512, 4096], name="W6")
            b6 = tensorflow_net_utils.bias_variable([4096], name="b6")
            conv6 = tensorflow_net_utils.conv2d_basic(pool5, W6, b6)
            relu6 = tf.nn.relu(conv6, name="relu6")
            if FLAGS.debug:
                tensorflow_net_utils.add_activation_summary(relu6)
            relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

            W7 = tensorflow_net_utils.weight_variable([1, 1, 4096, 4096], name="W7")
            b7 = tensorflow_net_utils.bias_variable([4096], name="b7")
            conv7 = tensorflow_net_utils.conv2d_basic(relu_dropout6, W7, b7)
            relu7 = tf.nn.relu(conv7, name="relu7")
            if FLAGS.debug:
                tensorflow_net_utils.add_activation_summary(relu7)
            relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

            W8 = tensorflow_net_utils.weight_variable([1, 1, 4096, 23], name="W8")
            b8 = tensorflow_net_utils.bias_variable([23], name="b8")
            conv8 = tensorflow_net_utils.conv2d_basic(relu_dropout7, W8, b8)

            # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

            # now to upscale to actual image size
            deconv_shape1 = image_net["pool4"].get_shape()
            W_t1 = tensorflow_net_utils.weight_variable([4, 4, deconv_shape1[3].value, 23], name="W_t1")
            b_t1 = tensorflow_net_utils.bias_variable([deconv_shape1[3].value], name="b_t1")
            conv_t1 = tensorflow_net_utils.conv2d_transpose_strided(conv8, W_t1, b_t1,
                                                                    output_shape=tf.shape(image_net["pool4"]))
            fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

            deconv_shape2 = image_net["pool3"].get_shape()
            W_t2 = tensorflow_net_utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value],
                                                        name="W_t2")
            b_t2 = tensorflow_net_utils.bias_variable([deconv_shape2[3].value], name="b_t2")
            conv_t2 = tensorflow_net_utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2,
                                                                    output_shape=tf.shape(image_net["pool3"]))
            fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

            shape = tf.shape(image)
            deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], 23])
            W_t3 = tensorflow_net_utils.weight_variable([16, 16, 23, deconv_shape2[3].value], name="W_t3")
            b_t3 = tensorflow_net_utils.bias_variable([23], name="b_t3")
            conv_t3 = tensorflow_net_utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3,
                                                                    stride=8)  # vector layer

            # annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

            W9 = tensorflow_net_utils.weight_variable([3, 3, 23, 3], name="W9")
            b9 = tensorflow_net_utils.bias_variable([3], name="b9")
            conv9_1 = tensorflow_net_utils.conv2d_basic(conv_t3, W9, b9)  # output layer, same to input
            conv9 = tf.nn.tanh(conv9_1)

            # conv8 = tf.nn.tanh(conv8)

        return conv9, conv_t3, middle_vector

    def train(self, loss_val, var_list, global_steps):
        # learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_steps, 5, 0.99, staircase=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # grads = optimizer.compute_gradients(loss_val, var_list=var_list)
        grads = optimizer.compute_gradients(loss_val)
        if FLAGS.debug:
            # print(len(var_list))
            for grad, var in grads:
                tensorflow_net_utils.add_gradient_summary(grad, var)
        # return optimizer.apply_gradients(grads, global_step = global_steps)
        return optimizer.apply_gradients(grads)

    def loss(self, input, output, middle_vector, object_groundtruth):
        # loss_groundtruth = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
        #                                                                       labels=labels,
        #                                                                       name="entropy")))
        picture_loss_all = tf.square(input - output)
        picture_loss = tf.reduce_mean(picture_loss_all, axis=[1, 2, 3], keep_dims=True)

        featuremaps = tf.split(object_groundtruth, NUM_OF_CLASSESS, axis=3, name='split_featuremap')

        # shape = tf.shape(middle_vector)
        # length = shape[-1]

        list_average_vector = []

        in_instance_error = tf.constant(0, dtype=tf.float32)
        total_pix = tf.constant(0, dtype=tf.float32)

        for featuremap in featuremaps[0:]:
            # featuremap = tf.cast(featuremap,dtype=tf.float32)

            # num_pix_obj = tf.reduce_sum(featuremap,axis=[1,2])
            num_pix_obj = tf.reduce_sum(featuremap, axis=[1, 2, 3], keep_dims=True)
            num_pix_obj = tf.cast(num_pix_obj, tf.float32)

            num_pix_obj = tf.add(num_pix_obj, 0.0001)

            total_pix = tf.add(num_pix_obj, total_pix)

            size = tf.shape(featuremap)
            size = tf.slice(size, [1], [2])
            resize_middle_vector = tf.image.resize_images(images=middle_vector, size=size)
            instance_vector = tf.multiply(resize_middle_vector, tf.cast(featuremap, tf.float32), name='instance')

            # instance_shape = tf.shape(instance,name='instance_shape')
            # vector = tf.split(instance,num_or_size_splits=instance_shape[-2])
            instance_sum_vector = tf.reduce_sum(instance_vector, axis=[1, 2], keep_dims=True)

            average_vector = tf.div(instance_sum_vector, num_pix_obj)
            list_average_vector.append(average_vector)
            distance_vector = tf.subtract(instance_vector, average_vector)
            distance_vector = tf.multiply(distance_vector, tf.cast(featuremap, tf.float32))

            in_instance_error_pow = tf.reduce_sum(tf.pow(distance_vector, 2), axis=[3], keep_dims=True)

            one_in_instance_error = tf.reduce_sum(in_instance_error_pow, axis=[1, 2], keep_dims=True)
            in_instance_error = tf.add(in_instance_error, one_in_instance_error)

            # tf.summary.histogram('feature_map', featuremap)  #[0/1]
            # tf.summary.histogram('total_pix', total_pix)    #[]
            # tf.summary.histogram('average_vector', average_vector)
            # tf.summary.image('feature_map111111111',featuremap,max_outputs=5)

        in_instance_loss = tf.div(in_instance_error, total_pix)

        list_average_vector = tf.stack(list_average_vector, axis=1)
        hnk_bar = tf.reduce_mean(list_average_vector, axis=1, keep_dims=True)
        list_average_vector = tf.subtract(list_average_vector, hnk_bar)
        between_class_err = tf.reduce_sum(tf.pow(list_average_vector, 2), axis=4)
        between_class_err = tf.div(between_class_err, -self.delta)
        between_class_err = tf.exp(between_class_err)
        between_class_loss = tf.reduce_mean(between_class_err, axis=1, keep_dims=True)

        total_loss = picture_loss + in_instance_loss + between_class_loss

        batch_total_loss = tf.reduce_mean(total_loss)
        batch_picture_loss = tf.reduce_mean(picture_loss)
        batch_in_instance_loss = tf.reduce_mean(in_instance_loss)
        batch_between_class_loss = tf.reduce_mean(between_class_loss)

        return batch_total_loss, batch_picture_loss, \
               batch_in_instance_loss, batch_between_class_loss,
        # total_loss is 1*1 ????
        pass


# class AEloss(torch.nn.Module):
#     def __init__(self, delta):
#         super(AEloss, self).__init__()
#         self.delta = delta
#         return
#
#     # feature 8, 3, 400, 400
#     # obj_mask 8, 21, 400, 400
#     def forward(self, feature, obj_mask):
#         split_mask = torch.chunk(tensor=obj_mask, chunks=obj_mask.size()[1], dim=1)
#
#         loss = 0
#         total = 0
#         hnk = []
#         for mask in split_mask:
#             num_pix_obj = torch.sum(mask).float()
#             total += num_pix_obj
#             obj = torch.mul(feature, mask.float())
#             hn_bar = torch.mean(torch.mean(obj, dim=3, keepdim=True), dim=2, keepdim=True)
#             hnk.append(hn_bar)
#             obj.sub_(hn_bar)
#             err = torch.sum(torch.pow(obj, 2), dim=1)
#             loss += torch.mean(err) * num_pix_obj
#         loss = torch.div(loss, total)
#
#         hnk = torch.stack(hnk, dim=1)
#         hnk_bar = torch.mean(hnk, dim=1, keepdim=True)
#         hnk.sub_(hnk_bar)
#         err = torch.sum(torch.pow(hnk, 2), dim=2)
#         err.div_(-self.delta)
#         err = torch.exp(err)
#         loss += torch.mean(err)
#
#         return loss


# input = Variable(torch.randn(8, 3, 40, 40), requires_grad=True)
# target = Variable(torch.randn(8, 20, 40, 40).ge(0.5), requires_grad=False)
#
# print(input.size())
# print(target.size())
# criterionAE = AEloss(0.1)
# loss = criterionAE(input, target)
# print(loss.size())
# loss.backward()
# print(input.grad)


def main(argv=None):
    myinstance = InstanceSegmentation()

    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")

    # image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")
    # annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
    # annotation = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="annotation")

    dataset_path = '/home/store-1-img/sunyaqiang/voc_tfrecord/'
    # dataset_path = '/Volumes/SUNXL-FAT/Deep_Learning/data/tfrecord/voc_tfrecord/'
    # dataset_path = '/home/sunyaqiang/tensorflow/instance_segmentation/version1/data/tfrecord/voc/'
    tf_name_list = os.listdir(dataset_path)
    name_list = [dataset_path + name_one for name_one in tf_name_list]
    # print(name_list)
    # name_list = ['./utils/sample/train000.tfrecord']
    image_tensor, instance_segmentation_tensor, instance_class_tensor = get_tfrecord_tensor_tf_dataset(name_list,
                                                                                                       batch_size=FLAGS.batch_size)
    # object_groundtruth = tf.placeholder(tf.float32, shape=[None, None, None, 21], name="object_groundtruth")

    instance_segmentation_tensor = tf.cast(instance_segmentation_tensor, tf.float32)
    image_tensor = tf.cast(image_tensor, tf.float32)
    image_tensor = tf.add(tf.multiply(tf.div(image_tensor, 255.0), 2.0), -1.0)

    pred_image, logits, middle_vector = myinstance.inference(image_tensor, keep_probability)

    tf.summary.image("input_image", image_tensor, max_outputs=2)
    # tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_image", pred_image, max_outputs=2)

    W10 = tensorflow_net_utils.weight_variable([1, 1, MIDDLE_CHANNEL, 3], name="W10", trainable=False)
    b10 = tf.constant([0, 0, 0], dtype=tf.float32, name="b10")
    conv10_1 = tensorflow_net_utils.conv2d_basic(middle_vector, W10, b10)  # output layer, same to input
    max = tf.reduce_max(conv10_1, axis=[1, 2], keep_dims=True)
    min = tf.reduce_min(conv10_1, axis=[1, 2], keep_dims=True)
    lenth = tf.subtract(max, min)
    middle = tf.div(tf.add(max, min), 2)
    conv10_2 = tf.multiply(tf.div(tf.subtract(conv10_1, middle), lenth), 2)
    # conv10_3 = tf.image.resize_images(conv10_2,[500,500])
    # middle_vector_visiable = conv10_3
    # conv10_2 = tf.nn.tanh(conv10_1)
    # conv10_3 = tf.image.resize_images(conv10_1,[500,500])
    # middle_vector_visiable = tf.nn.tanh(conv10_3)
    tf.summary.image('middle_vector_visiable', conv10_2, max_outputs=2)
    # tf.summary.image("middle_vector_visiable", middle_vector_visiable, max_outputs=2)
    tf.summary.histogram('middle_vector_hist', middle_vector)
    tf.summary.histogram('conv10_2_hist', conv10_2)

    # loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
    #                                                                       labels=tf.squeeze(annotation, squeeze_dims=[3]),
    #                                                                       name="entropy")))
    loss, picture_loss, in_instance_loss, between_class_loss = \
        myinstance.loss(input=image_tensor, output=pred_image,
                        middle_vector=middle_vector,
                        object_groundtruth=instance_segmentation_tensor)

    tf.summary.scalar("total_loss", loss)
    tf.summary.scalar(name='picutre_loss', tensor=picture_loss)
    tf.summary.scalar('in_instance_loss', in_instance_loss)
    tf.summary.scalar('between_class_loss', between_class_loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            tensorflow_net_utils.add_to_regularization_and_summary(var)

    global_steps = tf.train.get_global_step()
    train_op = myinstance.train(loss, trainable_var, global_steps)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader...")

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    # if FLAGS.mode == 'train':
    #     pass

    # config = tf.ConfigProto(log_device_placement=True)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    sess = tf.Session()

    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored... from:", ckpt.model_checkpoint_path)

    if FLAGS.mode == "train":
        for itr in range(MAX_ITERATION):
            print("Step: %d begain:" % (itr))
            # train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            # if itr % 500 == 0:
            train_loss, summary_str, value_picture_loss, value_in_instance_loss, value_between_class_loss = sess.run(
                [loss, summary_op, picture_loss, in_instance_loss, between_class_loss], feed_dict=feed_dict)

            print("Step: %d, Train_loss:%g" % (itr, train_loss))
            print('picture_loss:', value_picture_loss,
                  'in_instance_loss', value_in_instance_loss,
                  'between_class_loss', value_between_class_loss)

            summary_writer.add_summary(summary_str)

            if itr % 20 == 0:
                middle_vector_value = sess.run([middle_vector], feed_dict=feed_dict)
                to_image = np.array(middle_vector_value)[0, 0, :, :, 0:3]
                # print(np.shape(middle_vector_value))
                # print(np.shape(to_image))

                # img = Image.fromarray(to_image)
                # img.save('./vecotr.jpg')

                # valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                # valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                #                                        keep_probability: 1.0})
                # print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", global_step=global_steps)
                print('saved model')

    elif FLAGS.mode == "visualize":
        pass
        # valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        # pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
        #                                             keep_probability: 1.0})
        # valid_annotations = np.squeeze(valid_annotations, axis=3)
        # pred = np.squeeze(pred, axis=3)
        #
        # # for itr in range(FLAGS.batch_size):
        # #     tensorflow_net_utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
        # #     tensorflow_net_utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
        # #     tensorflow_net_utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
        # #     print("Saved image: %d" % itr)
        # for itr in range(FLAGS.batch_size):
        #     # cv2.imshow('image',valid_images[itr])
        #     # k = cv2.waitKey(0)
        #     # if k == ord(' '):
        #     #     pass
        #     print('itr:', itr)
        #     tensorflow_net_utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
        #     tensorflow_net_utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
        #     tensorflow_net_utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
        #     print("Saved image: %d" % itr)


def test():
    import tensorflow.contrib.eager as tfe
    tfe.enable_eager_execution()

    myinstance = InstanceSegmentation()

    init = tf.truncated_normal(shape=[16, 40, 40, 3], mean=0, stddev=1.0, dtype=tf.float32, seed=None, name='init')
    input = tf.constant(value=init, dtype=tf.float32, name='input')
    init2 = tf.truncated_normal(shape=[16, 40, 40, 3], mean=0, stddev=1.0, dtype=tf.float32, seed=None, name='init')
    output = tf.constant(value=init2, dtype=tf.float32, name='output')

    init2 = tf.truncated_normal(shape=[16, 10, 10, 9], mean=0.5, stddev=1.0, dtype=tf.float32, seed=None, name='init2')
    middle_vector = tfe.Variable(initial_value=init2, trainable=True, collections=None, name='middle_vector')

    init3 = tf.truncated_normal(shape=[16, 40, 40, 21], mean=0, stddev=1.0, dtype=tf.float32, seed=None, name='init3')
    # groundtruth = tf.constant(value=init3,dtype=tf.int32,name='groundtruth')
    groundtruth = tf.greater(init3, 1.5)
    groundtruth = tf.cast(groundtruth, dtype=tf.float32)
    myinstance.loss(input, output, middle_vector, groundtruth)


def test2():
    import tensorflow.contrib.eager as tfe
    tfe.enable_eager_execution()
    from utils.paser_pascolvoc import paser_pascolvoc_instance_segmentation_np_version
    DIRECTORY_ANNOTATIONS = 'Annotations/'
    DIRECTORY_IMAGES = 'JPEGImages/'
    DIRECTORY_SEGMENTATIONCLASS = 'SegmentationClass/'
    DIRECTORY_SEGMENTATIONOBJECT = 'SegmentationObject/'
    IMAGE_TYPE = 'jpg'
    SEGMENTATION_TYPE = 'png'
    ANNOTATION_TYPE = 'xml'
    dataset_dir = r'/Volumes/TOSHIBA/Deep_Learning/All_Data/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/'
    # dataset_dir = r'/home/store-1-img/VOC2007/VOCdevkit/VOC2012/'
    path = os.path.join(dataset_dir, DIRECTORY_IMAGES)
    filenames = sorted(os.listdir(path))

    filename = filenames[13]
    img_name = filename[:-4]
    name = img_name
    image_filename = os.path.join(dataset_dir, DIRECTORY_IMAGES, name + '.' + IMAGE_TYPE)
    segmentationClass_filename = os.path.join(dataset_dir, DIRECTORY_SEGMENTATIONCLASS, name + '.' + SEGMENTATION_TYPE)
    # annotation_filename = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS, name + '.' + ANNOTATION_TYPE)
    segmentationObject_filename = os.path.join(dataset_dir, DIRECTORY_SEGMENTATIONOBJECT,
                                               name + '.' + SEGMENTATION_TYPE)
    image_tensor, instance_labels = paser_pascolvoc_instance_segmentation_np_version(segmentationObject_filename,
                                                                                     segmentationClass_filename)

    image_tensor = tf.cast(image_tensor, tf.float32)
    image_tensor = tf.add(tf.multiply(tf.div(image_tensor, 255.0), 2.0), -1.0)

    myinstance = InstanceSegmentation()
    keep_probability = 1
    pred_image, logits, middle_vector = myinstance.inference(image_tensor, keep_probability)
    myinstance = InstanceSegmentation()

    # groundtruth = tf.constant(value=init3,dtype=tf.int32,name='groundtruth')
    myinstance.loss(image_tensor, output=pred_image, middle_vector=middle_vector, object_groundtruth=instance_labels)


if __name__ == "__main__":
    tf.app.run()
    # test()

    ########POSCAL VOC :color to class
    # [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    # [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    # [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    # [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    # [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    # [0, 64, 128]

    pass
