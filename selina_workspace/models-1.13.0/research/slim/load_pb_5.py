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

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


# We use our "load_graph" function
graph = load_graph("./models/frozen_model.pb")

# We can verify that we can access the list of operations in the graph
for op in graph.get_operations():
    print(op.name)     # <--- printing the operations snapshot below
    # prefix/Placeholder/inputs_placeholder
    # ...
    # prefix/Accuracy/predictions

# We access the input and output nodes
x = graph.get_tensor_by_name('prefix/input_neurons:0')
y = graph.get_tensor_by_name('prefix/prediction_restore:0')

# We launch a Session
with tf.Session(graph=graph) as sess:

    test_features = [[0.377745556,0.009904444,0.063231111,0.009904444,0.003734444,0.002914444,0.008633333,0.000471111,0.009642222,0.05406,0.050163333,7e-05,0.006528889,0.000314444,0.00649,0.043956667,0.016816667,0.001644444,0.016906667,0.00204,0.027342222,0.13864]]
        # compute the predicted output for test_x
    pred_y = sess.run( y, feed_dict={x: test_features} )
    print(pred_y)
