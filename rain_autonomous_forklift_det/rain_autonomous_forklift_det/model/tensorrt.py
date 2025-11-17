import torch
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np

import os
import atexit
import subprocess
import sys

import ctypes
ctypes.CDLL(os.path.expanduser("~/RobDet3D/plugins/lib/librd3d_trt_plugin.so"))

from rain_autonomous_forklift_det.utils.decorator import measure_time

cuda.init()
device = cuda.Device(torch.cuda.current_device())
ctx = device.retain_primary_context()
ctx.push()

def restore_jetson_clocks():
    os.system(f"sudo -S jetson_clocks --restore")
    print("jetson_clocks 설정 복원됨.")

atexit.register(restore_jetson_clocks)

class IASSD():
    def __init__(self, engine_path, bs=1, num_points=65536):
        self._engine_path = engine_path
        self._bs = bs
        self._num_points = num_points
        self._infer_engine = None

        if os.path.exists('/usr/bin/jetson_clocks'):
            print("\n" + "="*50)
            print("jetson_clocks를 활성화합니다. 비밀번호를 입력하세요:")
            print("="*50 + "\n")
            sys.stdout.flush()  # 버퍼 비우기
            
            subprocess.run(['sudo', 'jetson_clocks'], check=False)
            
            print("\njetson_clocks 활성화 완료!\n")
            sys.stdout.flush()

        self._initialize_infer_engine(self._engine_path, self._bs, self._num_points)
    
    def _initialize_infer_engine(self, engine_path, bs, num_points):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        trt_context = engine.create_execution_context()
        stream = cuda.Stream()

        input_tensor_name = "points"
        trt_context.set_input_shape(input_tensor_name, (bs, num_points, 4))

        # Verify all shapes are set
        assert trt_context.all_shape_inputs_specified

        dummy_input = np.zeros((bs, num_points, 4), dtype=np.float32)

        # Allocate host/device memory
        d_inputs = {}
        h_outputs = {}
        d_outputs = {}

        for idx in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(idx)
            tensor_mode = engine.get_tensor_mode(tensor_name)
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
            shape = trt_context.get_tensor_shape(tensor_name)
            size = int(np.prod(shape))

            if tensor_mode == trt.TensorIOMode.INPUT:
                d_inputs[tensor_name] = cuda.mem_alloc(dummy_input.nbytes)
            else:
                h_outputs[tensor_name] = cuda.pagelocked_empty((size,), dtype)  # <- 수정된 부분
                d_outputs[tensor_name] = cuda.mem_alloc(h_outputs[tensor_name].nbytes)


        self._infer_engine = {
            "engine": engine,
            "context": trt_context,
            "stream": stream,
            "bs": bs,
            "num_points": num_points,
            "d_inputs": d_inputs,
            "h_outputs": h_outputs,
            "d_outputs": d_outputs,
            "input_tensor_name": input_tensor_name
        }

    @measure_time('Model Inference')
    def infer_trt(self, input_data):
        context = self._infer_engine["context"]
        stream = self._infer_engine["stream"]

        # 1. 입력 복사
        cuda.memcpy_htod_async(
            self._infer_engine["d_inputs"][self._infer_engine["input_tensor_name"]],
            input_data,
            stream
        )

        # 2. 모든 텐서 바인딩 직접 설정
        for name, ptr in self._infer_engine["d_inputs"].items():
            context.set_tensor_address(name, int(ptr))

        for name, ptr in self._infer_engine["d_outputs"].items():
            context.set_tensor_address(name, int(ptr))

        # 3. 실행 (이제 bindings 인자는 안 넣음!)
        context.execute_async_v3(stream.handle)

        # 4. 출력 복사
        for name in self._infer_engine["h_outputs"]:
            cuda.memcpy_dtoh_async(
                self._infer_engine["h_outputs"][name],
                self._infer_engine["d_outputs"][name],
                stream
            )

        stream.synchronize()
        return self._infer_engine["h_outputs"]
    
    def get_stream(self):
        return self._infer_engine["stream"]

    
