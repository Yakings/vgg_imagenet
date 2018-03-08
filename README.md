# vgg_imagenet
train VGG in ImageNet dataset with multiGPUs



# multiGPUs
vgg_add_featuremaps/trian_vgg.py: training VGG with multiGPUs
# when use multiGPUs, tf.Variable() should be replaced wit tf.get_varaible(), to achieve the paraments shared.

# read ImageNet pictures.
vgg_add_featuremaps/utils.py: read ImageNet dataset with tf.data.Dataset()
