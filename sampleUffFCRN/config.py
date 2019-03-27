import graphsurgeon as gs
import tensorflow as tf

resize_bilinear = gs.create_node("ResizeBilinear")


namespace_plugin_map = {
    "ResizeBilinear": resize_bilinear
}

def preprocess(dynamic_graph):
    print('poop')
    # Now create a new graph by collapsing namespaces
    dynamic_graph.collapse_namespaces(namespace_plugin_map)
    # Remove the outputs, so we just have a single output node (NMS).
    dynamic_graph.remove(dynamic_graph.graph_outputs, remove_exclusive_dependencies=False)
