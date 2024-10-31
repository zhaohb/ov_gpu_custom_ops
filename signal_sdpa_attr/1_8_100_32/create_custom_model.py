from openvino.runtime import Core, Model, Tensor, Type
import openvino.runtime as ov
from openvino.runtime import opset13 as opset

def model():
    q = opset.parameter([1,8,100,32], Type.f32, name='q')
    k = opset.parameter([1,8,100,32], Type.f32, name='k')
    v = opset.parameter([1,8,100,32], Type.f32, name='v')
    sdpa_op = opset.scaled_dot_product_attention(q, k, v, name="sdpa")
    sdpa_op.set_friendly_name("sdpa")
    Result = opset.result(sdpa_op, name='output')
    return Model([Result],[q, k, v])

core = Core()
m = model()
for input, input_name in zip(m.inputs, ["q", "k", "v"]):
    input.get_tensor().set_names({input_name})
for output, output_name in zip(m.outputs, ["output"]):
    output.get_tensor().set_names({output_name})
ov.save_model(m, "custom_sdpa.xml")
