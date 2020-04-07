import tensorflow as tf
import sys
from tensorflow.python.platform import gfile

self.graph = tf.Graph()

with tf.gfile.GFile('resnet50_v1.5.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

print('Check out the input placeholders:')
nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
for node in nodes:
    print(node)

with self.graph.as_default():
    # Define input tensor
    self.input = tf.placeholder(np.float32, shape = [None, 224, 224, 3], name='input')
    tf.import_graph_def(graph_def, {'input': self.input})

self.graph.finalize()

print('Model loading complete!')

# Get layer names
layers = [op.name for op in self.graph.get_operations()]
for layer in layers:
    print(layer)
