import openvino as ov
import numpy as np
import time

q = np.random.random((1, 8, 100, 32))
k = np.random.random((1, 8, 100, 32))
v = np.random.random((1, 8, 100, 32))

q_1 = np.random.random((1, 8, 256, 32))
k_1 = np.random.random((1, 8, 256, 32))
v_1 = np.random.random((1, 8, 256, 32))

def infer():
    ov_model_path = "./custom_sdpa.xml"

    core = ov.Core()
    core.add_extension("/home/test/hongbo/opencl_demo/detr_custom_op/cpu_custom_op/build/libopenvino_template_extension.so")
    core.set_property("GPU", {"CONFIG_FILE": "/home/test/hongbo/opencl_demo/detr_custom_op/two_sdpa/custom_layer.xml"})

    relu_model = core.read_model(ov_model_path)

    relu_com_model = core.compile_model(relu_model, "GPU.1")

    runtime_model = relu_com_model.get_runtime_model()
    ov.save_model(runtime_model, "exec.xml")

    request = relu_com_model.create_infer_request()

    # input_dict = {"q": q, "k": k, "v": v, "q_1": q_1, "k_1": k_1, "v_1": v_1}
    input_dict = {"q": q, "k": k, "v": v}
    # print(testInput, testInput.shape)

    time_list = []
    for i in range(3500):
        start = time.time()
        request.start_async(input_dict, share_inputs=True)
        request.wait()
        end = time.time()
        time_list.append((end-start)*1000)
    result = np.array(request.output_tensors[0].data.copy())
    result_1 = np.array(request.output_tensors[1].data.copy())
    return result, result_1, np.array(time_list).mean()

print("test: ")
custom_op_result,  custom_op_result_1, avg_time = infer()
print('result sum: ', custom_op_result.sum())
print('result sum: ', custom_op_result_1.sum())
print("Average time for 1000 inferences: ", avg_time)
print(custom_op_result.shape)
print('\n')

