from ultralytics import YOLO

# Cargar el modelo
model = YOLO('yolov8n.pt')


print(model.names)
