import cv2
import config
import numpy as np
from ultralytics import YOLO

# ==================== КЛАССИФИКАЦИЯ ============================

# принимает путь до картинки, порог увернности и порог IOU
def classification_image(image_path: str, model_name:str) -> np.ndarray:
    if model_name == config.CLF_MODEL_LIST[0]:
      model = YOLO("yolo8n-clf.pt")
    else:
      model = YOLO("yolo8s-clf.pt")

    detections = model.predict(source=image_path)
    result_np_image = detections[0].plot()
    result_np_image = cv2.cvtColor(result_np_image, cv2.COLOR_BGR2RGB)
    return result_np_image

# ==================== СЕГМЕНТАЦИЯ ============================

# принимает путь до картинки, порог увернности и порог IOU
def segmentation_image(image_path: str, model_name:str, conf: float, iou: float) -> np.ndarray:
    if model_name == config.SEG_MODEL_LIST[0]:
      model = YOLO("yolo8n-seg.pt")
    else:
      model = YOLO("yolo8m-seg.pt")

    detections = model.predict(source=image_path, conf=conf, iou=iou)
    result_np_image = detections[0].plot()
    result_np_image = cv2.cvtColor(result_np_image, cv2.COLOR_BGR2RGB)
    return result_np_image