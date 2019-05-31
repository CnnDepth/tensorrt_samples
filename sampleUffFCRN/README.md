# Custom TensorRT sample "sampleUffFCRN"

This sample was developed to convert fully-convolutional neural networks from UFF format to TensorRT engine format. It supports Upsampling and Slice layers, and also supports Interleaving layer which speeds up deconvolutional blocks in decoder (see [this paper](https://arxiv.org/pdf/1606.00373.pdf)).  
This sample can convert models that take a batch of RGB images as input and give a matrix of the same width and height as output. In other words, it can convert models with input shape Nx3xHxW and output shape NxHxW.  
This sample was tested with TensorRT version 5.0.2.6.

**NOTE:** For now, Slice layer may be used only once in model. Its input dimensions are defined in `stridedSlicePlugin.h` file. If you want to use Slice layers in your model, you have to change its input dimensions manually.

## Usage

This sample can be run as:

`./sample_uff_fcrn [-h] params`

Params:
* `--uff` - path to the UFF model you want to convert
* `--uffInput` - name of the input layer in the UFF model
* `--output` - name of the output layer in the UFF model
* `--height` - height of input and output of the UFF model
* `--width` - width of input and output of the UFF model
* `--engine` - desired path to target TensorRT engine. If not set, engine will be created and tested, but not saved on disk
* `--fp16` - whether to use FP16 mode or not
* `--processImage` - path to PPM image file to run test inference on. The result of inference will be saved into `depth.ppm` file. If not set, no test inference will be run

Execution example:
`./sample_uff_fcrn --uff=./model.uff --uffInput=Placeholder --output=MarkOutput0 --height=256 --width=256 --engine=./engine.trt --fp16 --processImage=./image.ppm`