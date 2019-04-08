import graphsurgeon as gs
import tensorflow as tf

resize_nearest_0 = gs.create_plugin_node("up_sampling2d_1/ResizeNearestNeighbor",
                                          op="ResizeNearestNeighbor",
                                          nbInputChannels=1024,
                                          inputHeight=15,
                                          inputWidth=20
                                          )
resize_nearest_1 = gs.create_plugin_node("up_sampling2d_2/ResizeNearestNeighbor",
                                          op="ResizeNearestNeighbor",
                                          nbInputChannels=512,
                                          inputHeight=30,
                                          inputWidth=40
                                          )
resize_nearest_2 = gs.create_plugin_node("up_sampling2d_3/ResizeNearestNeighbor",
                                          op="ResizeNearestNeighbor",
                                          nbInputChannels=256,
                                          inputHeight=60,
                                          inputWidth=80
                                          )
resize_nearest_3 = gs.create_plugin_node("up_sampling2d_4/ResizeNearestNeighbor",
                                          op="ResizeNearestNeighbor",
                                          nbInputChannels=128,
                                          inputHeight=120,
                                          inputWidth=160
                                          )
resize_nearest_4 = gs.create_plugin_node("up_sampling2d_5/ResizeNearestNeighbor",
                                          op="ResizeNearestNeighbor",
                                          nbInputChannels=64,
                                          inputHeight=240,
                                          inputWidth=320
                                          )

namespace_plugin_map = {
    # conversions for upsampling layers with nearest neighbor interpolation
    "up_sampling2d_1/ResizeNearestNeighbor": resize_nearest_0,
    "up_sampling2d_2/ResizeNearestNeighbor": resize_nearest_1,
    "up_sampling2d_3/ResizeNearestNeighbor": resize_nearest_2,
    "up_sampling2d_4/ResizeNearestNeighbor": resize_nearest_3,
    "up_sampling2d_5/ResizeNearestNeighbor": resize_nearest_4
}

def preprocess(dynamic_graph):
    # Now create a new graph by collapsing namespaces
    dynamic_graph.collapse_namespaces(namespace_plugin_map)
