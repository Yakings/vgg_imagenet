"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf
import numpy as np

# from tensorflow_vgg_master import utils as mast_utils

# from tensorflow_vgg_master import vgg19_trainable as vgg19
from vgg_add_featuremaps import add_sequence as vgg19


from vgg_add_featuremaps import utils as read_utils
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


FLAGS = tf.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)

# tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "../log/vgg_changed/", "path to logs directory")
tf.flags.DEFINE_string("trained_dir", "../trained/vgg_changed/", "path to logs directory")

# tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
# tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
# tf.flags.DEFINE_string("model_dir", "segmentation/Model_zoo/", "Path to vgg model mat")
# tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
#
# tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
# tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
# tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")






CLASSES = 1001
NUM_EPOCH = 500



def test():
    img1 = mast_utils.load_image("./test_data/tiger.jpeg")
    img1_true_result = [1 if i == 292 else 0 for i in range(1000)]  # 1-hot result for tiger

    batch1 = img1.reshape((1, 224, 224, 3))

    with tf.device('/cpu:0'):
        sess = tf.Session()

        images = tf.placeholder(tf.float32, [1, 224, 224, 3])
        true_out = tf.placeholder(tf.float32, [1, 1000])
        train_mode = tf.placeholder(tf.bool)

        vgg = vgg19.Vgg19('./pretrained-models/vgg19.npy')
        vgg.build(images, train_mode)

        # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
        print(vgg.get_var_count())

        sess.run(tf.global_variables_initializer())

        # test classification
        prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
        mast_utils.print_prob(prob[0], './synset.txt')

        # simple 1-step training
        cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
        train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
        sess.run(train, feed_dict={images: batch1, true_out: [img1_true_result], train_mode: True})

        # test classification again, should have a higher probability about tiger
        prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
        mast_utils.print_prob(prob[0], './synset.txt')

        # test save
        vgg.save_npy(sess, './trained_models/test-save.npy')

def top_k_error(predictions, labels, k=1):
    batch_size = 32.0 #tf.shape(predictions)[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=k))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size


def main(argv=None):



    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")

    # image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")
    # annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
    # annotation = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="annotation")


    # data_dir = '/Volumes/TOSHIBA/Deep_Learning/All_Data/ImageNet/ILSVRC2012/ImageNet2012Train/'
    data_dir = '/home/store-1-img/zhaobo/ImageNet/ILSVRC2012/ImageNet2012Train/'
    name_labels, number_to_label = read_utils.read_filename_label_to_tuple(data_dir=data_dir)
    # print(name_labels)
    # print(number_to_label)
    # image_tensor, label_tensor = _read_function(name_labels[0][0],name_labels[0][1])
    name_labels_np = np.array(name_labels)
    np.random.shuffle(name_labels_np)
    print(name_labels_np)
    name_np = name_labels_np[:, 0]
    label_np = name_labels_np[:, 1]
    label_np = np.array(label_np,dtype=np.int32)
    print(label_np)

    image_batch_tensor, label_tensor = read_utils.get_input_tensor_tf_dataset(data_list=(name_np, label_np))
    one_hot_label_tensor = tf.one_hot(label_tensor, depth=CLASSES)

    with tf.device('/gpu:0'):
    # if 1 == 1:

        images = image_batch_tensor
        true_out = one_hot_label_tensor
        train_mode = tf.placeholder(tf.bool)

        vgg = vgg19.Vgg19('./pretrained-models/vgg19.npy')
        vgg.build(images, train_mode)

        # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
        print(vgg.get_var_count())


        # test classification
        # prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
        # mast_utils.print_prob(prob[0], './synset.txt')

        # simple 1-step training
        loss = tf.losses.softmax_cross_entropy(onehot_labels=true_out,logits=vgg.fc8)
        # loss = tf.reduce_sum(tf.square(tf.subtract(vgg.prob, true_out)))


        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        # train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

    top1_error = top_k_error(vgg.prob,label_tensor,k=1)

    print("Setting up summary op...")
    tf.summary.image('image_batch_tensor',image_batch_tensor)
    tf.summary.scalar(name="total_loss", tensor=loss)
    tf.summary.scalar(name="top1_error", tensor=top1_error)

    summary_op = tf.summary.merge_all()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)
    for i in range(NUM_EPOCH):
        total_step = 10000
        for j in range(total_step):
            _, loss_value = sess.run([train_op,loss], feed_dict={train_mode: True})
            print('epoch:',i, '---', j,',loss:',loss_value)

            if j%10 == 0:
                print('write summary:',i,'--',j)
                summary_str = sess.run(summary_op, feed_dict={train_mode: True})
                step = i*total_step +j
                summary_writer.add_summary(summary_str,step)
                summary_writer.flush()

                if j%10000 == 0:
                    saver.save(sess, FLAGS.trained_dir)
                    print('saving ckpt!')

    # test classification again, should have a higher probability about tiger
        # prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
        # mast_utils.print_prob(prob[0], './synset.txt')

        # test save
    vgg.save_npy(sess, './trained_models_changed/test-save.npy')





#    tf.summary.histogram('middle_vector_hist',middle_vector)


    # loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
    #                                                                       labels=tf.squeeze(annotation, squeeze_dims=[3]),
    #                                                                       name="entropy")))




if __name__ == "__main__":
    tf.app.run()