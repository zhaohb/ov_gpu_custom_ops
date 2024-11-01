from openvino.runtime import Core, Model, Tensor, Type
import openvino.runtime as ov
from openvino.runtime import opset13 as opset

def model():
    q = opset.parameter([1,8,100,32], Type.f32, name='q')
    k = opset.parameter([1,8,100,32], Type.f32, name='k')
    v = opset.parameter([1,8,100,32], Type.f32, name='v')

    q_1 = opset.parameter([1,8,100,32], Type.f32, name='q_1')
    k_1 = opset.parameter([1,8,100,32], Type.f32, name='k_1')
    v_1 = opset.parameter([1,8,100,32], Type.f32, name='v_1')

    q_add = opset.add(q, q_1, name="q_add")
    k_add = opset.add(k, k_1, name="k_add")
    v_add = opset.add(v, v_1, name="v_add")

    sdpa_op = opset.scaled_dot_product_attention(q_add, k_add, v_add, name="sdpa")
    sdpa_op.set_friendly_name("sdpa")
    sdpa_op_1 = opset.scaled_dot_product_attention(q_add, k_add, v_add, name="sdpa1")
    sdpa_op_1.set_friendly_name("sdpa1")
    Result = opset.result(sdpa_op, name='output')
    Result_1 = opset.result(sdpa_op_1, name='output1')
    return Model([Result, Result_1],[q, k, v, q_1, k_1, v_1])

core = Core()
m = model()
# for input, input_name in zip(m.inputs, ["q", "k", "v"]):
#     input.get_tensor().set_names({input_name})
# for output, output_name in zip(m.outputs, ["output", "output1"]):
#     output.get_tensor().set_names({output_name})
ov.save_model(m, "custom_sdpa.xml")
