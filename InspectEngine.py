import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
engine_path = "yolov8n.engine"

with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
    print("✅ TensorRT Engine cargado correctamente")
    print("📦 Tensores disponibles:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        mode = engine.get_tensor_mode(name)
        print(f"🔹 {name}: {shape} ({'INPUT' if mode == trt.TensorIOMode.INPUT else 'OUTPUT'})")
