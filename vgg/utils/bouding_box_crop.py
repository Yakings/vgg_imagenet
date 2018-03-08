import tensorflow as tf
# from Pillow import Image
from PIL import Image
import numpy as np
def bounding_box_crop_test():
    img = tf.gfile.FastGFile('/Users/sunxiaolang/Downloads/12.png','rb').read()
    img = tf.image.decode_png(img)

    img_1 = tf.cast(img,tf.float32)

    img_original = tf.image.draw_bounding_boxes(images=[img_1],boxes=[[[0,0,0.1,0.1]]])
    imgsize = tf.shape(img)

    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(image_size = imgsize,bounding_boxes=[[[0,0,0.1,0.1]]],
                                                                        min_object_covered = 0.3,use_image_if_no_bounding_boxes = False)

    imgnew = tf.slice(img, begin, size)
    imgnew = tf.cast(imgnew,tf.float32)
    imgshow = tf.image.draw_bounding_boxes(images=[imgnew],boxes=bbox_for_draw)
    # imgshow = tf.cast(imgshow,tf.uint8)

    sess = tf.Session()

    img_original = sess.run(img_original)
    img= sess.run(imgshow)
    img = img.astype(np.uint8)
    img_original = img_original.astype(np.uint8)
    # print(img)

    img = Image.fromarray(img[0])
    img_original = Image.fromarray(img_original[0])

    img.show()
    img_original.show()

if __name__ == '__main__':
    bounding_box_crop_test()