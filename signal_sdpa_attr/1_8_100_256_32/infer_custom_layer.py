import openvino as ov
import numpy as np
import time

q = np.random.random((1, 8, 100, 32))
k = np.random.random((1, 8, 256, 32))
v = np.random.random((1, 8, 256, 32))
mask = np.random.random((1, 8, 1, 256))*0

def infer():
    ov_model_path = "./custom_sdpa.xml"

    core = ov.Core()
    core.add_extension("/home/test/hongbo/opencl_demo/detr_custom_op/cpu_custom_op_mask/build/libopenvino_template_extension.so")
    core.set_property("GPU", {"CONFIG_FILE": "/home/test/hongbo/opencl_demo/detr_custom_op/signal_sdpa_attr/1_8_100_256_32/mask_custom_layer.xml"})

    relu_model = core.read_model(ov_model_path)
    relu_com_model = core.compile_model(relu_model, "GPU.1")

    request = relu_com_model.create_infer_request()

    input_dict = {"q": q, "k": k, "v": v, "mask": mask}
    # print(testInput, testInput.shape)
    time_list = []
    for i in range(10000):
        start = time.time()
        request.start_async(input_dict, share_inputs=True)
        request.wait()
        end = time.time()
        time_list.append((end-start)*1000)
    result = np.array(request.output_tensors[0].data.copy())
    return result, np.array(time_list).mean()

# 
def infer_origin():
    ov_model_path = "./origin/custom_sdpa.xml"

    core = ov.Core()

    relu_model = core.read_model(ov_model_path)

    relu_com_model = core.compile_model(relu_model, "GPU.1")

    request = relu_com_model.create_infer_request()

    input_dict = {"q": q, "k": k, "v": v, "mask": mask}
    # print(testInput, testInput.shape)
    time_list = []
    for i in range(10000):
        start = time.time()
        request.start_async(input_dict, share_inputs=True)
        request.wait()
        end = time.time()
        time_list.append((end-start)*1000)
    result = np.array(request.output_tensors[0].data.copy())
    return result, np.array(time_list).mean()

print("test custom op: ")
custom_op_result, avg_time = infer()
print('result sum: ', custom_op_result[0][0])
print("Average time for 1000 inferences: ", avg_time)
print(custom_op_result.shape)
print('\n')
print("-------------------")
print('\n')
print("test origin: ")
origin_result, origin_avg_time = infer_origin()
print('result sum: ', origin_result[0][0])
print("Average time for 1000 inferences: ", origin_avg_time)
print(origin_result.shape)
