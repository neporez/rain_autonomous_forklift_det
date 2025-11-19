import time
import torch
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import pycuda.autoinit
from rd3d.api import demo
import ctypes
from utils import load_plugins
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--engine")
print([pc.name for pc in trt.get_plugin_registry().plugin_creator_list])
model, dataloader, args = demo(parser)
model.cuda()
model.eval()

frame_id = [0]
bs = len(frame_id)
batch_dict = dataloader.dataset.collate_batch([dataloader.dataset[fid] for fid in frame_id])
dataloader.dataset.load_data_to_gpu(batch_dict)
points = batch_dict['points'].view(bs, -1, 5)[..., 1:].contiguous().cpu().numpy()

print(batch_dict['image_shape'])

print(points.shape)


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []

    class HostDeviceMem(object):
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
            return self.__str__()

    for binding in engine:

        dims = engine.get_binding_shape(binding)
        size = trt.volume(dims) * engine.max_batch_size  # The maximum batch size which can be used for inference.
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):  # Determine whether a binding is an input binding.
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings

def trt_inf():
    logger = trt.Logger(trt.Logger.ERROR)
    with open(args.engine, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    h_inputs = {'points': points}
    d_inputs, h_outputs, d_outputs = {}, {}, {}

    with engine.create_execution_context() as context:
        stream = cuda.Stream()
        context.set_optimization_profile_async(0, stream.handle)
        context.set_binding_shape(engine.get_binding_index("points"), (bs, 16384, 4))
        assert context.all_binding_shapes_specified

        # Allocate
        for binding in engine:
            idx = engine.get_binding_index(binding)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            shape = context.get_binding_shape(idx)
            size = trt.volume(shape)
            if engine.binding_is_input(binding):
                d_inputs[binding] = cuda.mem_alloc(h_inputs[binding].nbytes)
            else:
                h_outputs[binding] = cuda.pagelocked_empty(size, dtype)
                d_outputs[binding] = cuda.mem_alloc(h_outputs[binding].nbytes)

        def infer_once():
            for key in h_inputs:
                cuda.memcpy_htod_async(d_inputs[key], h_inputs[key], stream)
            context.execute_async_v2(
                bindings=[int(d_inputs[k]) for k in d_inputs] + [int(d_outputs[k]) for k in d_outputs],
                stream_handle=stream.handle)
            for key in h_outputs:
                cuda.memcpy_dtoh_async(h_outputs[key], d_outputs[key], stream)
            stream.synchronize()

        # 🔥 워밍업
        for _ in range(10):
            infer_once()

        # 📏 측정
        timings = []
        for _ in range(500):
            t1 = time.time()
            infer_once()
            t2 = time.time()
            timings.append((t2 - t1) * 1000)

        print(f"Average inference time: {np.mean(timings):.2f} ms")

        return h_outputs['boxes'].reshape(bs, 256, -1), \
               h_outputs['scores'].reshape(bs, 256, -1)



# trt_inf()
r1, r2 = trt_inf()
# print(t1.shape, t1.dtype, '\n', t1)
# print(t2.shape, t2.dtype, '\n', t2)
print(r1.shape, r1.dtype, '\n', r1)
print(r2.shape, r2.dtype, '\n', r2)

# import matplotlib.pyplot as plt
#
# plt.subplot(1, 2, 1)
# norm = np.linalg.norm(t1, axis=-1, keepdims=True).astype(np.uint8)
# hist = plt.hist(norm.ravel(), bins=20, alpha=0.5)
# norm = np.linalg.norm(r1, axis=-1, keepdims=True).astype(np.uint8)
# hist = plt.hist(norm.ravel(), bins=20, alpha=0.5)
# plt.subplot(1, 2, 2)
# norm = np.linalg.norm(t2, axis=-1, keepdims=True).astype(np.uint8)
# hist = plt.hist(norm.ravel(), bins=20, alpha=0.5)
# norm = np.linalg.norm(r2, axis=-1, keepdims=True).astype(np.uint8)
# hist = plt.hist(norm.ravel(), bins=20, alpha=0.5)
# plt.show()
#

from rd3d.utils.viz_utils import viz_scenes

xyz = points[-1]
box = r1[-1].reshape((-1, 8))[..., :7]
viz_scenes(
    (xyz, box),
)