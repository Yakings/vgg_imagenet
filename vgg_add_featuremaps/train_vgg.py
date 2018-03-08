"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf
import numpy as np

# from tensorflow_vgg_master import utils as mast_utils

from tensorflow_vgg_master import vgg19_trainable as vgg19
# from tensorflow_vgg_master import vgg16_trainable as vgg19
# from tensorflow_vgg_master import vgg11_trainable as vgg19

# from vgg_add_featuremaps import add_sequence as vgg19


from vgg_add_featuremaps import utils as read_utils
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


FLAGS = tf.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)

# tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "../log/vgg_origin/", "path to logs directory")
tf.flags.DEFINE_string("trained_dir", "../trained/vgg_origin/", "path to logs directory")

# tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
# tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
# tf.flags.DEFINE_string("model_dir", "segmentation/Model_zoo/", "Path to vgg model mat")
# tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
#
# tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
# tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
# tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")






CLASSES = 1000
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
    batch_size = tf.constant(72.0) #tf.shape(predictions)[0]
    in_topk = tf.to_float(tf.nn.in_top_k(predictions, labels, k=k))
    num_correct = tf.reduce_sum(in_topk)
    return tf.div(tf.subtract(batch_size, num_correct),batch_size)




def average_gradients(tower_grads):
    average_grads = []
    # print(tower_grads)
    for grad_and_vars in zip(*tower_grads):
        print(grad_and_vars)
        grads = []
        for g, _ in grad_and_vars:
            print('g:',g)
            expanded_g = tf.expand_dims(g,0)
            grads.append(expanded_g)
        grad = tf.concat(values=grads, axis=0)
        grad = tf.reduce_mean(grad,axis=0)
        v = grad_and_vars[0][1]
        grad_and_var =(grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def multiGPU_trian():
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")

    # image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")
    # annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
    # annotation = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="annotation")


    # data_dir = '/Volumes/TOSHIBA/Deep_Learning/All_Data/ImageNet/ILSVRC2012/ImageNet2012Train/'
    data_dir = '/home/store-1-img/zhaobo/ImageNet/ILSVRC2012/ImageNet2012Train/'
    name_labels, number_to_label = read_utils.read_filename_label_to_tuple(data_dir=data_dir)
    # print(name_labels)
    print('number_to_label:',number_to_label)
    # image_tensor, label_tensor = _read_function(name_labels[0][0],name_labels[0][1])
    name_labels_np = np.array(name_labels)
    np.random.shuffle(name_labels_np)
    print('name_labels_np:',name_labels_np)
    name_np = name_labels_np[:, 0]
    label_np = name_labels_np[:, 1]
    label_np = np.array(label_np, dtype=np.int32)
    print('label_np',label_np)


    # for i in range(len(name_np)):
    #     index = label_np[i]
    #     name = number_to_label[index]
    #     # print(name,name_np[i])
    #     import re
    #     flags = re.search(name, name_np[i], re.I)
    #     if flags is None:
    #         print('not match!!')


    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        image_batch_tensor, label_tensor = read_utils.get_input_tensor_tf_dataset(data_list=(name_np, label_np))
        train_mode = tf.placeholder(tf.bool)


        # Decay the learning rate exponentially based on the number of steps.
        # learning_rate = tf.train.exponential_decay(learning_rate = 1.0,
        #                                            global_step=global_step,
        #                                            decay_steps=1000,
        #                                            decay_rate=0.9,
        #                                            staircase=True)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)


        optimizer = tf.train.AdamOptimizer(0.001)
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(4):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_id%d' % i) as scope:
                        one_hot_label_tensor = tf.one_hot(label_tensor, depth=CLASSES)
                        images = image_batch_tensor
                        true_out = one_hot_label_tensor
                        vgg = vgg19.Vgg19('./pretrained-models/vgg19.npy')
                        vgg.build(images, train_mode)

                        # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
                        print(i, vgg.get_var_count())

                        # test classification
                        # prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
                        # mast_utils.print_prob(prob[0], './synset.txt')

                        # simple 1-step training
                        # loss = tf.losses.softmax_cross_entropy(onehot_labels=true_out, logits=vgg.fc8)
                        # loss = tf.reduce_sum(tf.square(tf.subtract(vgg.prob, true_out)))
                        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=label_tensor, logits=vgg.fc8, name='cross_entropy_per_example')
                        loss = tf.reduce_mean(cross_entropy, name='cross_entropy')

                        tf.get_variable_scope().reuse_variables()
                        grads = optimizer.compute_gradients(loss)
                        print(len(tf.trainable_variables()))
                        tower_grads.append(grads)

        # in cpu
        grads = average_gradients(tower_grads)
        apply_gradient_op = optimizer.apply_gradients(grads,global_step=global_step)
        train_op = apply_gradient_op


        top1_error = top_k_error(vgg.prob, label_tensor, k=1)
        print("Setting up summary op...")

        rgb = tf.cast(image_batch_tensor, dtype=tf.float32)
        rgb_scaled = tf.subtract(tf.div(rgb, 127.5), 1.0)
        # rgb_scaled = tf.image.resize_images(rgb_scaled, (224, 224))
        tf.summary.image('image_batch_tensor', rgb_scaled)

        conv_show = tf.slice(vgg.conv2_1, (0,0,0,0), (3,112,112,3))
        tf.summary.image('conv2_1_batch_tensor', conv_show)
        tf.summary.scalar(name="total_loss", tensor=loss)
        tf.summary.scalar(name="top1_error", tensor=top1_error)
        # tf.summary.scalar(name='learning rate', tensor=learning_rate)
        tf.summary.histogram(name='score', values=vgg.prob)
        tf.summary.histogram(name='rgb_scaled', values=rgb_scaled)
        tf.summary.histogram(name='conv1_2_batch_tensor', values=conv_show)
        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)
        for i in range(NUM_EPOCH):
            total_step = 10000
            for j in range(total_step):
                _, loss_value = sess.run([train_op, loss], feed_dict={train_mode: True})
                print('epoch:', i, '---', j, ',loss:', loss_value)

                if j % 10 == 0:
                    print('write summary:', i, '--', j)
                    # logistic, lable_value = sess.run([vgg.prob, label_tensor], feed_dict={train_mode: True})
                    # print(logistic, lable_value)

                    summary_str = sess.run(summary_op, feed_dict={train_mode: True})
                    step = i * total_step + j
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                    if j % 10000 == 0:
                        saver.save(sess, FLAGS.trained_dir,global_step=global_step)
                        print('saving ckpt!')
        vgg.save_npy(sess, './trained_models_origin/test-save.npy')

def oneGPU_train():
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
    label_np = np.array(label_np, dtype=np.int32)
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
        loss = tf.losses.softmax_cross_entropy(onehot_labels=true_out, logits=vgg.fc8)
        # loss = tf.reduce_sum(tf.square(tf.subtract(vgg.prob, true_out)))


        # train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

    top1_error = top_k_error(vgg.prob, label_tensor, k=1)

    print("Setting up summary op...")
    tf.summary.image('image_batch_tensor', image_batch_tensor)
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
            _, loss_value = sess.run([train_op, loss], feed_dict={train_mode: True})
            print('epoch:', i, '---', j, ',loss:', loss_value)

            if j % 10 == 0:

                print('write summary:', i, '--', j)

                summary_str = sess.run(summary_op, feed_dict={train_mode: True})
                step = i * total_step + j
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

                if j % 10000 == 0:
                    saver.save(sess, FLAGS.trained_dir)
                    print('saving ckpt!')


                    # test classification again, should have a higher probability about tiger
                    # prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
                    # mast_utils.print_prob(prob[0], './synset.txt')

                    # test save
    vgg.save_npy(sess, './trained_models_origin/test-save.npy')





    #    tf.summary.histogram('middle_vector_hist',middle_vector)


    # loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
    #                                                                       labels=tf.squeeze(annotation, squeeze_dims=[3]),
    #                                                                       name="entropy")))


def main(argv=None):
    multiGPU_trian()


if __name__ == "__main__":
    tf.app.run()