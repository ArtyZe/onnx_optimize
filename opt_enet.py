import onnx
import os
from onnx.optimizer import *
import numpy as np
from onnx import shape_inference
from onnx import version_converter
from onnx.helper import make_tensor, make_node, TensorProto, make_tensor_value_info
to_remove = []
delete_op_list = ["Gather", "Unsqueeze", "Concat", "Shape", "Constant"]
# from onnx import register_custom_op_symbolic
ONNX_DTYPE = {
    0: TensorProto.FLOAT,
    1: TensorProto.FLOAT,
    2: TensorProto.UINT8,
    3: TensorProto.INT8,
    4: TensorProto.UINT16,
    5: TensorProto.INT16,
    6: TensorProto.INT32,
    7: TensorProto.INT64,
    8: TensorProto.STRING,
    9: TensorProto.BOOL
}

def infer_model_shape(onnx_model):
    nodes = onnx_model.graph.node
    tensors = onnx_model.graph.initializer
    onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = u'1'
    for input_info in onnx_model.graph.input:
        input_info.type.tensor_type.shape.dim[0].dim_param = u'1'
        input_shape = input_info.type.tensor_type.shape.dim
        reinput_shape_flag = 0
        # if model has dynamic input shape, we can not run shape inference, so fix input shape
        for index, shape in enumerate(input_shape[1:]):
            if not all(['0' < x < '9' for x in shape.dim_param.split()]):
                logger.error('The model has dynamic input shape, please input your input shape')
                reinput_shape_flag = 1
                break
        if reinput_shape_flag:
            reinput_shape = input("input for example: 1,3,416,416 :")
            reinput_shape = reinput_shape.split(",")
            assert(len(input_shape) == len(reinput_shape))
            for index , shape in enumerate(input_shape):
                shape.dim_param = reinput_shape[index].encode("utf8")
    for i, tensor in enumerate(tensors):
        value_info = make_tensor_value_info(tensor.name, ONNX_DTYPE[tensor.data_type], tensor.dims)
        onnx_model.graph.input.insert(i+1, value_info) # because 0 is for placeholder, so start index is 1
    inferred_onnx_model = shape_inference.infer_shapes(onnx_model)
    # print('After shape inference, the shape info of Y is:\n{}'.format(inferred_onnx_model.graph.value_info))

    return inferred_onnx_model

def elimate_useless_constant_node(model):
    nodes = model.graph.node
    node_inputs = []
    for node in nodes:
        if node.op_type != "Constant":
            node_inputs.extend(node.input)
    for node in nodes:
        if node.op_type == "Constant" and all([output not in node_inputs for output in node.output]):
            print("delete useless constant node: ", node.output)
            nodes.remove(node)
    return model

def optimize_onnx(nodes, to_remove):
    # print(to_remove)
    updated = True
    while updated:
        updated = False
        for node in nodes:
            # ensure all useless op been removed, loop will break
            if all([output in to_remove for output in node.output]) and node.op_type in delete_op_list:
                print("delete node: ", node.output)
                nodes.remove(node)
                to_remove.extend(node.input)
                updated = True  

def optimize_model(model):
    nodes = model.graph.node 
    for i, node in enumerate(nodes):
        if node.op_type == "MaxUnpool":
            print("process maxunpool node: ", node.output)
            optimize_onnx(model.graph.node, [node.input[2]])
            new_node = make_node(
                'MaxUnpool',
                inputs = [node.input[0], node.input[1]],
                outputs = node.output,
                # basicly, kernel shape need to get from onnxruntime inference result, however onnxruntime can not run because of 
                # wrong maxunpool format, so here I just set it to [2,2]
                kernel_shape = [2,2],
                strides = [2, 2]
            )
            # replace old maxunpool node with new maxunpool node
            print("replace node: ", new_node.output)
            model.graph.node.remove(node)
            model.graph.node.insert(i, new_node)
    return model

onnxfile = "D:\work\models\pytorch\PyTorch-ENet/Enet_onnx.onnx"
onnx_model = onnx.load(onnxfile)
print("step1...")
# delete shape->gather->unsequeeze ops and replace maxunpool op
onnx_model = optimize_model(onnx_model)
# delete useless constant ops
print("step2...")
onnx_model = elimate_useless_constant_node(onnx_model)
onnx.checker.check_model(onnx_model)
print("step3...")
onnx_model = infer_model_shape(onnx_model)
onnx.save(onnx_model, "./new.onnx")
print("success...")
