#!/usr/bin/env python -W ignore::DeprecationWarning
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf #Need tensorflow v1 since v2.0 does not have tf.contrib (even in tf.compat.v1)
import numpy as np

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

from tensorflow.python.summary import summary

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

tf.logging.set_verbosity(tf.logging.ERROR)

##############################################################
# LoadModel Class # To make this applicable to any frozen_model.pb file, need to find input and ouput placeholders
##############################################################
class LoadModel(object):

  def __init__(self, model_filepath):

    # The file path of model
    self.model_filepath = model_filepath
    # Initialize the model
    #self.load_graph(model_filepath = self.model_filepath)
      
  def load_graph(self):
    '''
    Load trained model.
    '''
    print('Loading model...')
    model_filepath = self.model_filepath
    self.graph = tf.Graph()

    with tf.gfile.GFile(model_filepath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    print('Check out the input placeholders:')
    nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
    for node in nodes:
        print(node)
    
    print('Check out the output nodes')
    out_nodes = [n.name + '=>' +  n.op for n in graph_def.node if n.op in ( 'Softmax')]
    for node in out_nodes:
        print(node)

    
    print('Model loading complete!')

    '''
    # Get layer names
    layers = [op.name for op in self.graph.get_operations()]
    for layer in layers:
        print(layer)


    # Check out the weights of the nodes
    weight_nodes = [n for n in graph_def.node if n.op == 'Const']
    for n in weight_nodes:
        print("Name of the node - %s" % n.name)
        # print("Value - " )
        # print(tensor_util.MakeNdarray(n.attr['value'].tensor))
    '''
    # In this version, tf.InteractiveSession and tf.Session could be used interchangeably. 
    # self.sess = tf.InteractiveSession(graph = self.graph)
    #self.sess = tf.Session(graph = self.graph)
    
    #def test(data):
    #  output_tensor = self.graph.get_tensor_by_name("import/softmax_tensor:0")
    #  def get_output():
    #    with tf.Session(graph=self.graph) as sess: 
    #      coord = tf.train.Coordinator()
    #      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
          # Define input tensor
          # self.input = tf.placeholder(np.float32, shape = [100, 224, 224, 3], name='input_tensor')
    #      self.input = data
    #      tf.import_graph_def(graph_def, {'input_tensor': self.input})

          # sess.run(tf.global_variables_initializer())
          # print("evaluating data")
          # data = sess.run(data) 
          # print("done evaluating")

          # print("data type")
          # print(type(data))
          # print(data.shape)
          #self.graph.finalize()

          # Know your output node name
          #output_tensor = self.graph.get_tensor_by_name("import/softmax_tensor:0")
    #      print("running model")
    #      output = sess.run(self.output_tensor) #feed_dict = {self.input: data}  

    #      coord.request_stop()
    #      coord.join(threads)
    #      sess.close()
    #      return output

    #  return output_tensor, get_output
    def test(data):
         #init = tf.global_variables_initializer()
         #self.sess.run(init)
         #self.graph.finalize()
        self.sess = tf.Session(graph=self.graph)
        #with self.sess as sess: 
        #self.input = data
        #self.input = tf.placeholder(np.float32, shape = [100, 224, 224, 3], name='input_tensor'
        tf.import_graph_def(graph_def, {'input_tensor':data})# self.input})
         #    sess.run(tf.global_variables_initializer())
             # print("evaluating data")
             # data = sess.run(data) 
         #   tf.train.start_queue_runners(sess) 
         #    data = data.eval(sess1)
         #  print("done evaluating")
             # print("data type")
             # print(type(data))
             # print(data.shape)
         # Know your output node name
        output_tensor = self.graph.get_tensor_by_name("import/softmax_tensor:0")
        print("running model")
            #output = sess.run(output_tensor)#, feed_dict = {self.input: data})  

        return output_tensor

    # Resnet_v1_50 default image size 
    # as defined in tensorflow/models-1.13.0/research/slim/nets/resnet_v1.py#L282
    test.default_image_size = 224
    return test



# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import math
# import tensorflow as tf

# from datasets import dataset_factory
# from nets import nets_factory
# from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

# tf.app.flags.DEFINE_string(
#     'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'model_dir', 'resnet50_v1-5.pb', 'The directory of the model to evaluate')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_bool(
    'quantize', False, 'whether to use quantized graph or not.')

tf.app.flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('eval_dir', '',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')

FLAGS = tf.app.flags.FLAGS


def _convert_to_example(filename, image_buffer, label, synset, height, width):
  """Build an Example proto for an example.
  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/synset': _bytes_feature(synset),
      'image/format': _bytes_feature(image_format),
      'image/filename': _bytes_feature(os.path.basename(filename)),
      'image/encoded': _bytes_feature(image_buffer)}))
  return example

def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.
  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.
  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.
  Returns:
    the cropped (and resized) image.
  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.slice(image, offsets, cropped_shape)
  return tf.reshape(image, cropped_shape)


def _central_crop(image_list, crop_height, crop_width):
  """Performs central crops of the given image list.
  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.
  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    outputs.append(_crop(image, offset_height, offset_width,
                         crop_height, crop_width))
  return outputs

def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.
  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.
  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])
  return resized_image

def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.
  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.
  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.
  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.to_int32(tf.rint(height * scale))
  new_width = tf.to_int32(tf.rint(width * scale))
  return new_height, new_width

def _resize_image(image, height, width):
  """Simple wrapper around tf.resize_images.
  This is primarily to make sure we use the same `ResizeMethod` and other
  details each time.
  Args:
    image: A 3-D image `Tensor`.
    height: The target height for the resized image.
    width: The target width for the resized image.
  Returns:
    resized_image: A 3-D tensor containing the resized image. The first two
      dimensions have the shape [height, width].
  """
  return tf.image.resize_images(
      image, [height, width], method=tf.image.ResizeMethod.BILINEAR,
      align_corners=False)

def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  print("=================================starting main======================")

  tf.logging.set_verbosity(tf.logging.INFO)

  model = LoadModel(FLAGS.model_dir)
  network_fn = model.load_graph()
  graph = model.graph
  print("Loaded model!")

  with graph.as_default():
    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    # network_fn = nets_factory.get_network_fn(
    #     FLAGS.model_name,
    #     num_classes=(dataset.num_classes - FLAGS.labels_offset),
    #     is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
    print("\n\n\n###################image###################### ", image)
    print("\n\n\n###################label###################### ", label)
    label -= FLAGS.labels_offset

    tf_global_step = tf.train.get_or_create_global_step()
    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)
    print("########################################################################image.shape!######################################################################")
    print(image.shape)

    #image.reshape([224,224,3])
    #image.set_shape([224,224,3])
    image = _aspect_preserving_resize(image, 256)
    image = _central_crop([image], 224, 224)[0]
    image.set_shape([224, 224, 3])
    image = tf.to_float(image)
    #tf.image.resize(image, [224,224])
    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, labels = tf.train.batch(
      [image, label],
      batch_size=FLAGS.batch_size,
      num_threads=FLAGS.num_preprocessing_threads,
      capacity= 5 * FLAGS.batch_size) #5 *

    model.input = images

    ####################
    # Define the model #
    ####################
    print("########################################################################images.shape!######################################################################")
    print(images.shape)
    #images_np = images.eval()
    
    #network_fn.output_tensor =v#get_tensor_by_name("import/softmax_tensor:0")
# self.graph.get_tensor_by_name("import/softmax_tensor:0")
    logits =  network_fn(images) #network_fn(images)

    print("evaluated!")
    #if FLAGS.quantize:
    #  tf.contrib.quantize.create_eval_graph()

    #if FLAGS.moving_average_decay:
    #  variable_averages = tf.train.ExponentialMovingAverage(
    #     FLAGS.moving_average_decay, tf_global_step)
    #  variables_to_restore = variable_averages.variables_to_restore(
    #     slim.get_model_variables())
    #  variables_to_restore[tf_global_step.op.name] = tf_global_step
    #else:
    #  variables_to_restore = slim.get_variables_to_restore()


    # predictions = tf.argmax(logits, 1)
    # labels = tf.squeeze(labels)

    # # Define the metrics:
    # names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    #  'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
    #  'Recall_5': slim.metrics.streaming_recall_at_k(
    #      logits, labels, 5),
    # })

    # # Print the summaries to screen.
    # for name, value in names_to_values.items():
    #   summary_name = 'eval/%s' % name
    #   op = tf.summary.scalar(summary_name, value, collections=[])
    #   op = tf.Print(op, [value], summary_name)
    #   tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)


#    with model.sess as sess:
#      coord = tf.train.Coordinator()
#      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#      init = tf.global_variables_initializer()
#      summaries_update = tf.get_collection_ref(tf.GraphKeys.SUMMARIES) 
#      sess.run(init)
#      sess.run(summaries_update)
#      sess.run(logits)
#      print("evaluations")
#      for op in summaries_update:
#        print(op.eval())
      #sess.run(op)
#      coord.request_stop()
#      coord.join(threads)
#      sess.close()
    # print("done!")
    #results = run_fn()
    #results = network_fn(images)
    with model.sess as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      writer = tf.summary.FileWriter("output", sess.graph)
      sess.run(tf.global_variables_initializer())#,feed_dict = {model.input: data}))
      output = sess.run(logits)#, feed_dict = {model.input: data})
      labels = sess.run(labels)#, feed_dict = {model.input: data}) 
      writer.close()
      coord.request_stop()
      coord.join(threads)
      sess.close()

    # labels = np.argmax(labels)
    test_prediction = np.argmax(output, axis = 0).reshape((-1,1))
    print("labels")
    print(labels)
    print("test_prediction")
    print(test_prediction)
    test_accuracy = model_accuracy(label = labels, prediction = test_prediction)

    print('Test Accuracy: %f' % test_accuracy)
   # TODO(sguada) use num_epochs=1
    # if FLAGS.max_num_batches:
    #   num_batches = FLAGS.max_num_batches
    # else:
    #   # This ensures that we make a single pass over all of the data.
    #   num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    # if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    #   checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    # else:
    #   checkpoint_path = FLAGS.checkpoint_path

    # tf.logging.info('Evaluating %s' % checkpoint_path)

    # slim.evaluation.evaluate_once(
    #  master=FLAGS.master,
    #  checkpoint_path=checkpoint_path,
    #  logdir=FLAGS.eval_dir,
    #  num_evals=num_batches,
    #  eval_op=list(names_to_updates.values()),
    #  variables_to_restore=None) #variables_to_restore=variables_to_restore)

def model_accuracy(label, prediction):

    # Evaluate the trained model
    return np.sum(label == prediction) / len(prediction)

if __name__ == '__main__':
  tf.app.run()
