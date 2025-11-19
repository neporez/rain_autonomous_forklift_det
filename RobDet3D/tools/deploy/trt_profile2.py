from rd3d.api import demo

def evaluate(engine_file, dataloader, use_build_in):
    import torch
    import numpy as np
    import pycuda.driver as cuda
    import tensorrt as trt
    import pycuda.autoinit
    import nvtx
    from utils import load_plugins
    from utils.profiler import MyProfiler

    logger = trt.Logger(trt.Logger.ERROR)

    with open(engine_file, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    bs = dataloader.batch_size
    with engine.create_execution_context() as context:
        stream = cuda.Stream()

        input_name = "points"
        context.set_input_shape(input_name, (bs, 65536, 4))  # TensorRT 10.x

        assert context.all_shape_inputs_specified

        batch_dict = next(iter(dataloader))
        dataloader.dataset.load_data_to_gpu(batch_dict)

        # Prepare host input
        h_inputs = {
            'points': batch_dict['points'].view(bs, -1, 5)[..., 1:].contiguous().cpu().numpy()
        }

        d_inputs = {}
        h_outputs = {}
        d_outputs = {}

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            shape = context.get_tensor_shape(name)
            size = int(np.prod(shape))

            if mode == trt.TensorIOMode.INPUT:
                d_inputs[name] = cuda.mem_alloc(h_inputs[name].nbytes)
            else:
                h_outputs[name] = cuda.pagelocked_empty((size,), dtype)
                d_outputs[name] = cuda.mem_alloc(h_outputs[name].nbytes)

        # Set tensor address before inference
        for name in d_inputs:
            context.set_tensor_address(name, int(d_inputs[name]))
        for name in d_outputs:
            context.set_tensor_address(name, int(d_outputs[name]))

        # CUDA event for timing
        start_event = cuda.Event()
        end_event = cuda.Event()

        def infer(show_time=False):
            # Host → Device
            cuda.memcpy_htod_async(d_inputs[input_name], h_inputs[input_name], stream)

            # nvtx.range_push("execute_async_v3")
           

            context.execute_async_v3(stream.handle)

            
            # nvtx.range_pop()

            # Device → Host
            for name in h_outputs:
                cuda.memcpy_dtoh_async(h_outputs[name], d_outputs[name], stream)

            stream.synchronize()

        # Warm-up
        print("Warming up...")
        for i in range(10):
            infer()

        # Profiler 연결
        if use_build_in:
            context.profiler = MyProfiler([
                "FPSampling", "HAVSampling", "GridBallQuery", "BallQuery", "ForeignNode", "NMSBEV"
            ])

        print("Profiling started.")
        cuda.start_profiler()

        for i in range(50 if use_build_in else 2):
            infer(show_time=True)

        cuda.stop_profiler()
        print("Profiling finished.")

        if use_build_in:
            context.report_to_profiler()
            context.profiler.print()

        print("done")

def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=Path)
    parser.add_argument('--build_in', action='store_true', default=False)
    _, dataloader, args = demo(parser)

    use_build_in = args.build_in
    engine_file = args.engine

    evaluate(engine_file, dataloader, use_build_in)

if __name__ == "__main__":
    main()
