import cv2
import os
import time

# Configuración
TRAIN_RATIO = 0.8
output_base = "dataset/images"
train_dir = os.path.join(output_base, "train")
val_dir = os.path.join(output_base, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Inicializar cámara con resolución 1280x720 y cronometrar
start = time.time()
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Leer un frame para forzar inicialización completa
ret, frame = cap.read()
if not ret:
    raise RuntimeError("No se pudo capturar imagen.")
frame_height, frame_width = frame.shape[:2]

init_time = time.time() - start
print(f"⏱️ Tiempo en inicializar la cámara y capturar primer frame: {init_time:.2f} segundos")

# Variables
pose_images = []
global_counter = 1
total_photos_saved = 0

# Posiciones de botones
button_height = 50
button_width = 180
button_margin_bottom = 70  # Subimos los botones

button_take_photo = ((10, frame_height - button_height - button_margin_bottom),
                     (10 + button_width, frame_height - button_margin_bottom))

button_new_pose = ((frame_width - button_width - 10, frame_height - button_height - button_margin_bottom),
                   (frame_width - 10, frame_height - button_margin_bottom))

clicked_button = None

def draw_buttons(img):
    cv2.rectangle(img, button_take_photo[0], button_take_photo[1], (0, 255, 0), -1)
    cv2.putText(img, "Sacar Foto", (button_take_photo[0][0] + 20, button_take_photo[1][1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.rectangle(img, button_new_pose[0], button_new_pose[1], (255, 200, 0), -1)
    cv2.putText(img, "Nueva Pose", (button_new_pose[0][0] + 15, button_new_pose[1][1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

def mouse_callback(event, x, y, flags, param):
    global clicked_button
    if event == cv2.EVENT_LBUTTONDOWN:
        if button_take_photo[0][0] <= x <= button_take_photo[1][0] and button_take_photo[0][1] <= y <= button_take_photo[1][1]:
            clicked_button = 's'
        elif button_new_pose[0][0] <= x <= button_new_pose[1][0] and button_new_pose[0][1] <= y <= button_new_pose[1][1]:
            clicked_button = 'n'

cv2.namedWindow("Captura de dataset")
cv2.setMouseCallback("Captura de dataset", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar imagen.")
        break

    display = frame.copy()
    pose_text = f"Fotos actuales en esta pose: {len(pose_images)}"
    total_text = f"Total guardadas: {total_photos_saved}"
    cv2.putText(display, pose_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display, total_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    draw_buttons(display)
    cv2.imshow("Captura de dataset", display)

    key = cv2.waitKey(1) & 0xFF
    action = None

    if key == ord('s') or clicked_button == 's':
        action = 's'
    elif key == ord('n') or clicked_button == 'n':
        action = 'n'
    elif key == ord('q'):
        print("Saliendo...")
        break

    if action == 's':
        pose_images.append(frame.copy())
        print(f"Foto tomada. Total en esta pose: {len(pose_images)}")
    elif action == 'n':
        if pose_images:
            total = len(pose_images)
            train_count = int(total * TRAIN_RATIO)
            for i, img in enumerate(pose_images):
                filename = f"imagen_{global_counter:04d}.jpg"
                path = os.path.join(train_dir if i < train_count else val_dir, filename)
                cv2.imwrite(path, img)
                print(f"Guardado en: {path}")
                global_counter += 1
                total_photos_saved += 1
            pose_images = []
            print("Distribución 80/20 completada para esta pose.")
        else:
            print("No hay fotos en esta pose para distribuir.")

    clicked_button = None

cap.release()
cv2.destroyAllWindows()
