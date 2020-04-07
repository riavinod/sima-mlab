#!usr/bin/env python
import sys 
import tensorflow as tf #import tensorflow.compat.v1 as tf #Must use tf1.0
import numpy as np

def load_graph(model_filepath):
    '''
    Load trained model.
    '''
    print('Loading model...')
    graph = tf.Graph()

    with tf.gfile.GFile(model_filepath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    print('Check out the input placeholders:')
    nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
    for node in nodes:
        print(node)

    with graph.as_default():
        # Define input tensor
        input = tf.placeholder(np.float32, shape = [None, 224, 224, 3], name='input_tensor')
        tf.import_graph_def(graph_def, {'input_tensor': input})

    graph.finalize()

    print('Model loading complete!')

    '''
    # Get layer names
    layers = [op.name for op in graph.get_operations()]
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
    # sess = tf.InteractiveSession(graph = graph)
    sess = tf.Session(graph = graph)

modeldir = sys.argv[1] #"/Users/selina/Desktop/mlabsima/sima-project/resnet50v1-5_dir/resnet50_v1-5.pb"
# datasetdir = sys.argv[2]  #"$HOME/sean_workspace/imagenet-data/raw-data"

print("Model Directory (system argv[1]) = " + modeldir)
# print("Dataset Directory (system arv[2]) = " + datasetdir)

# Initialize the model
load_graph(model_filepath = modeldir)
print("finished loading!")

