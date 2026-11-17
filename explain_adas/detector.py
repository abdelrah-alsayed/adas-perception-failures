from ultralytics import YOLO

def load_model(weights_path):
     """Load a YOLO model from a weights file."""
     return YOLO(weights_path)

def detect(model, image_path, conf=0.25, one_to_one=False, device=0):
    """Run the model on one image. Return a list of detections."""
    extra = {"nms": False} if one_to_one else {}
    result = model.predict(image_path, imgsz=640, conf=conf, device=device, verbose=False, **extra)[0]
    detections = []
    for box, score, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        detections.append({
            "name": result.names[int(cls)],
            "score": float(score),
            "box": [float(v) for v in box],
        })
    return detections
    