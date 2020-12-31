import onnx
import numpy as np
import onnx.helper as helper
from onnxsim import simplify
from onnx import shape_inference, TensorProto, version_converter
import logging
import copy
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[ONNXSTRIP]")

onnx_model_path = "renset_v2_50.onnx"
start_tensor_names = ["image"]
end_tensor_names = ["conv2d_58.tmp_0", "conv2d_66.tmp_0", "conv2d_74.tmp_0"]

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
        value_info = helper.make_tensor_value_info(tensor.name, ONNX_DTYPE[tensor.data_type], tensor.dims)
        onnx_model.graph.input.insert(i+1, value_info) # because 0 is for placeholder, so start index is 1
    inferred_onnx_model = shape_inference.infer_shapes(onnx_model)
    # print('After shape inference, the shape info of Y is:\n{}'.format(inferred_onnx_model.graph.value_info))

    return inferred_onnx_model

def save_model(onnx_model, path):
    # onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, path)
    
def get_node_by_output_name(name, nodes):
    return_nodes = []
    for node in nodes:
        if name in node.output:
            return_nodes.append(node)
    return return_nodes

def get_node_by_input_name(name, nodes):
    return_nodes = []
    for node in nodes:
        if name in node.input:
            return_nodes.append(node)
    return return_nodes

def get_value_info_by_name(name, model):
    for x in model.graph.value_info:
        if x.name == name:
            return x
    return None
                       
def strip_model_from_end_to_start(input_names, output_names, model):
    if len(input_names) == 0 and len(output_names) == 0:
        return model
    nodes = model.graph.node     
    nodes_to_keep = []
    need_remove = []
    # Breadth first search to find all the nodes that we should keep.
    next_to_visit = []
    for x in output_names:
        next_to_visit += get_node_by_output_name(x, nodes)
    nodes_to_keep_list = []
    while next_to_visit:
        node = next_to_visit[0]
        del next_to_visit[0]
        # Already visited this node.
        if node in nodes_to_keep_list:
            continue
        nodes_to_keep.append(node)
        nodes_to_keep_list.append(node)
        for node_input in node.input:
            if node_input in input_names:
                continue
            next_to_visit += get_node_by_output_name(node_input, nodes)    
    for node in nodes:
        if node not in nodes_to_keep_list:
            need_remove.append(node)
    for i in need_remove:
        nodes.remove(i)
    return model
                      
def strip_model_from_start_to_end(input_names, output_names, model):
    if len(input_names) == 0 and len(output_names) == 0:
        return model
    nodes = model.graph.node     
    nodes_to_keep = []
    need_remove = []
    # Breadth first search to find all the nodes that we should keep.
    next_to_visit = []
    for x in input_names:
        next_to_visit += get_node_by_input_name(x, nodes)
    nodes_to_keep_list = []
    while next_to_visit:
        node = next_to_visit[0]
        del next_to_visit[0]
        # Already visited this node.
        if node in nodes_to_keep_list:
            continue
        nodes_to_keep.append(node)
        nodes_to_keep_list.append(node)
        for node_output in node.output:
            if node_output in output_names:
                continue
            next_to_visit += get_node_by_input_name(node_output, nodes)    
    for node in nodes:
        if node not in nodes_to_keep_list:
            need_remove.append(node)
    for i in need_remove:
        nodes.remove(i)
    return model

def replace_orig_input_node(model, start_node_names):
    if len(start_node_names) == 0 or model.graph.input[0].name in start_node_names:
        return model
    graph = model.graph
    del graph.input[0]
    for node_name in start_node_names:
        tensor_info = get_value_info_by_name(node_name, model)
        if tensor_info is not None:
            tensor_shape = tensor_info.type.tensor_type.shape.dim
            tensor_shape_list = [tensor_shape[i].dim_value for i in range(0, len(tensor_shape))]
            value_info = helper.make_tensor_value_info(node_name, ONNX_DTYPE[tensor_info.type.tensor_type.elem_type], tensor_shape_list)
            graph.input.append(value_info)
    return model

def replace_orig_output_node(model, end_node_names):
    if len(end_node_names) == 0 or model.graph.output[0].name in end_node_names:
        return model
    graph = model.graph 
    del graph.output[0]
    for node_name in end_node_names:
        tensor_info = get_value_info_by_name(node_name, model)
        if tensor_info is not None:
            tensor_shape = tensor_info.type.tensor_type.shape.dim
            # watch out if output node shape is 2 dims
            tensor_shape_list = [tensor_shape[i].dim_value for i in range(0, len(tensor_shape))]
            value_info = helper.make_tensor_value_info(node_name, ONNX_DTYPE[tensor_info.type.tensor_type.elem_type], tensor_shape_list)
            graph.output.append(value_info)
    return model   

# load model
logger.info("step0: load model...")
onnx_model = onnx.load(onnx_model_path)

# model shape inference
logger.info("step1: infer model shape...")
onnx_model = infer_model_shape(onnx_model)

# convert model input node to start nodes
logger.info("step2: replace model input node...")
onnx_model = replace_orig_input_node(onnx_model, start_tensor_names)

# convert model output node to end nodes
logger.info("step3: replace model output node...")
onnx_model = replace_orig_output_node(onnx_model, end_tensor_names)

# strip model from end node to start node
logger.info("step4: strip model from end to start...")
onnx_model = strip_model_from_end_to_start(start_tensor_names, end_tensor_names, onnx_model)

# strip model from start node to end node, if your start node is in middle model, turn off it,
# or your useful node beside start node will gone
logger.info("step5: strip model from start to end...")
# onnx_model = strip_model_from_start_to_end(start_tensor_names, end_tensor_names, onnx_model)

logger.info("step6: simply model...")
# onnx_model, check = simplify(onnx_model)
# assert check

# check and save onnx model
logger.info("step7: check and save model...")
save_model(onnx_model, "./new.onnx")
logger.info("save model successfully...")



