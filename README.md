# vgg_imagenet
train VGG in ImageNet dataset with multiGPUs



# multiGPUs
vgg_add_featuremaps/trian_vgg.py: training VGG with multiGPUs.

Attention:
  When use multiGPUs, tf.Variable() in one GPU training model should be replaced wit tf.get_varaible(), to achieve the paraments shared. In this file, it has ben replaced.

# read ImageNet pictures.
vgg_add_featuremaps/utils.py: read ImageNet dataset with tf.data.Dataset()
