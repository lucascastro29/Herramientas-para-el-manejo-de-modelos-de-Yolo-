from ultralytics import YOLO

# Cargar el modelo
model = YOLO('yolov8n.pt')

# Exportar a ONNX
model.export(format='onnx')


