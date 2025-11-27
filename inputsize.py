import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
with open("best.engine", "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
    print("✅ Engine cargado")

    n_bindings = engine.num_io_tensors
    print(f"🔎 Número de tensores de entrada/salida: {n_bindings}")

    for i in range(n_bindings):
        name = engine.get_tensor_name(i)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        shape = engine.get_tensor_shape(name)
        print(f"{'Entrada' if is_input else 'Salida'} → {name} — forma: {shape}")
