import graphsurgeon as gs
import tensorflow as tf

resize_nearest_0 = gs.create_plugin_node("up_sampling2d_1/ResizeNearestNeighbor",
                                          op="ResizeNearestNeighbor",
                                          nbInputChannels=3,
                                          inputHeight=240,
                                          inputWidth=320
                                          )
resize_nearest_1 = gs.create_plugin_node("up_sampling2d_2/ResizeNearestNeighbor",
                                          op="ResizeNearestNeighbor",
                                          nbInputChannels=3,
                                          inputHeight=480,
                                          inputWidth=640
                                          )

namespace_plugin_map = {
    # conversions for upsampling layers with nearest neighbor interpolation
    "up_sampling2d_1/ResizeNearestNeighbor": resize_nearest_0,
    "up_sampling2d_2/ResizeNearestNeighbor": resize_nearest_1
}

def preprocess(dynamic_graph):
    # Now create a new graph by collapsing namespaces
    dynamic_graph.collapse_namespaces(namespace_plugin_map)
    # Remove the outputs, so we just have a single output node (NMS).
    dynamic_graph.remove(dynamic_graph.graph_outputs, remove_exclusive_dependencies=False)
