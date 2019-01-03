"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from cleverhans.attacks import MultiModelIterativeMethod
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import inception_resnet_v2

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path1', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path2', '', 'Path to checkpoint for adversarial trained inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path3', '', 'Path to checkpoint for adversarial trained inception-resnet network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 10, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
    filenames: list file names without path of each image
    Lenght of this list could be less than batch_size, in this case only
    first few images of the result are elements of the minibatch.
    images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]

    filepaths = tf.gfile.Glob(os.path.join(input_dir, '*.png'))

    for count, filepath in enumerate(filepaths):

        with tf.gfile.Open(filepath) as f:
            image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
    images: array with minibatch of images
    filenames: list of filenames without path
    If number of file names in this list less than number of images in
    the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            # img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            img = np.round(255.0 * (images[i, :, :, :] + 1.0) * 0.5).astype(np.uint8)

            Image.fromarray(img).save(f, format='PNG')


class InceptionModel(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes, scope=''):
        self.num_classes = num_classes
        self.built = False
        self.scope = scope

    def __call__(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
                x_input, num_classes=self.num_classes, is_training=False, reuse=reuse, scope=self.scope)

        self.built = True
        output = end_points['Predictions']
        # Strip off the extra reshape op at the output
        probs = output.op.inputs[0]
        return probs


class IrNetModel(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes, scope=''):
        self.num_classes = num_classes
        self.built = False
        self.scope = scope

    def __call__(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            _, end_points = inception_resnet_v2.inception_resnet_v2(x_input, num_classes=self.num_classes,
                                                                    reuse=reuse, is_training=False, scope=self.scope)

        self.built = True
        output = end_points['Predictions']

        # Strip off the extra reshape op at the output
        probs = output.op.inputs[0]
        return probs


def main(_):

    start_time = time.time()

    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():

        # ---------------------------------
        # define graph

        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        model1 = InceptionModel(num_classes, scope='sc1')
        model2 = InceptionModel(num_classes, scope='sc2')
        model3 = IrNetModel(num_classes, scope='sc3')

        method = MultiModelIterativeMethod([model1, model2, model3])
        x_adv = method.generate(x_input, eps=eps, clip_min=-1., clip_max=1., nb_iter=17)

        # ---------------------------------
        # set input

        all_vars = tf.global_variables()
        # print(all_vars)

        unique_name_headers = set([k.name.split('/')[0] for k in all_vars])

        model1_vars = [k for k in all_vars if k.name.startswith('sc1')]
        model2_vars = [k for k in all_vars if k.name.startswith('sc2')]
        model3_vars = [k for k in all_vars if k.name.startswith('sc3')]

        # name of variable `my_var:0` corresponds `my_var` for loader
        model1_keys = [s.name.replace('sc1', 'InceptionV3')[:-2] for s in model1_vars]
        model2_keys = [s.name.replace('sc2', 'InceptionV3')[:-2] for s in model2_vars]
        model3_keys = [s.name.replace('sc3', 'InceptionResnetV2')[:-2] for s in model3_vars]

        saver1 = tf.train.Saver(dict(zip(model1_keys, model1_vars)))
        saver2 = tf.train.Saver(dict(zip(model2_keys, model2_vars)))
        saver3 = tf.train.Saver(dict(zip(model3_keys, model3_vars)))

        session_creator = tf.train.ChiefSessionCreator(master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:

            saver1.restore(sess, FLAGS.checkpoint_path1)
            saver2.restore(sess, FLAGS.checkpoint_path2)
            saver3.restore(sess, FLAGS.checkpoint_path3)

            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, FLAGS.output_dir)

    elapsed_time = time.time() - start_time
    print('elapsed time: {0:.0f} [s]'.format(elapsed_time))

if __name__ == '__main__':
    tf.app.run()
