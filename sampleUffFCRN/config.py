import graphsurgeon as gs
import tensorflow as tf

resize_bilinear_0 = gs.create_node("ResizeBilinear0")
resize_bilinear_1 = gs.create_node("ResizeBilinear1")
resize_bilinear_2 = gs.create_node("ResizeBilinear2")
resize_bilinear_3 = gs.create_node("ResizeBilinear3")
resize_bilinear_4 = gs.create_node("ResizeBilinear4")

resize_nearest_0 = gs.create_node("ResizeNearestNeighbor0")
resize_nearest_1 = gs.create_node("ResizeNearestNeighbor1")
resize_nearest_2 = gs.create_node("ResizeNearestNeighbor2")
resize_nearest_3 = gs.create_node("ResizeNearestNeighbor3")
resize_nearest_4 = gs.create_node("ResizeNearestNeighbor4")

namespace_plugin_map = {
    # conversions for 5 bilinear upsampling layers
    "up_sampling2d_1/ResizeBilinear": resize_bilinear_0,
    "up_sampling2d_2/ResizeBilinear": resize_bilinear_1,
    "up_sampling2d_3/ResizeBilinear": resize_bilinear_2,
    "up_sampling2d_4/ResizeBilinear": resize_bilinear_3,
    "up_sampling2d_5/ResizeBilinear": resize_bilinear_4,

    # conversions for 5 upsampling layers with nearest neighbor interpolation
    "up_sampling2d_1/ResizeNearestNeighbor": resize_nearest_0,
    "up_sampling2d_2/ResizeNearestNeighbor": resize_nearest_1,
    "up_sampling2d_3/ResizeNearestNeighbor": resize_nearest_2,
    "up_sampling2d_4/ResizeNearestNeighbor": resize_nearest_3,
    "up_sampling2d_5/ResizeNearestNeighbor": resize_nearest_4
}

def preprocess(dynamic_graph):
    # Now create a new graph by collapsing namespaces
    dynamic_graph.collapse_namespaces(namespace_plugin_map)
    # Remove the outputs, so we just have a single output node (NMS).
    dynamic_graph.remove(dynamic_graph.graph_outputs, remove_exclusive_dependencies=False)
