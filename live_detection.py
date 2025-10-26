# import cv2
# from ultralytics import YOLO

# model = YOLO('yolo11n.pt')
# cap = cv2.VideoCapture("rtsp://Tm9YW22vy38LUQww:Hg0tkDOilPJerUDq@192.168.86.35/live0", cv2.CAP_FFMPEG)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         results = model(frame)
#         annotated_frame = results[0].plot()
#         # Display frame
#         cv2.imshow('Live Detection', annotated_frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# cap.release()
# cv2.destroyAllWindows()



import math
import os
import cv2
import torch
from ultralytics import YOLO
from urllib.parse import quote
max_frames   = 500  
target_conf  = 0.5
frame_count  = 0
should_stop = False
near_threshold = 150


rtsp_url = "rtsp://Tm9YW22vy38LUQww:Hg0tkDOilPJerUDq@192.168.86.35/live0"
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
target_class = "remote"
model = YOLO("yolo11n.pt")
names = model.names  # dict: {class_id: class_name}

def extract_detections(model, frame, conf_thresh=0.25):
    """Return list of {cls, name, conf, x1,y1,x2,y2} for one result frame."""
    # model = YOLO("yolo11n.pt")
    results = model(frame)
    res = results[0]
    dets = []
    boxes = res.boxes  # ultralytics.engine.results.Boxes
    if boxes is None or boxes.shape[0] == 0:
        return dets

    xyxy = boxes.xyxy.cpu().numpy()            # (N,4)
    conf = boxes.conf.cpu().numpy()            # (N,)
    cls  = boxes.cls.cpu().numpy().astype(int) # (N,)

    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        if c < conf_thresh:
            continue
        dets.append({
            "class_id": int(k),
            "name": names[int(k)],
            "confidence": float(c),
            "bbox": [float(x1), float(y1), float(x2), float(y2)]
        })
    return dets

def box_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def distance(bbox1, bbox2):
    c1 = box_center(bbox1)
    c2 = box_center(bbox2)
    return math.dist(c1, c2)

def detectfromlivefeed(rstp_url, custom_model, target_class, target_conf):
    max_frames   = 500  
    target_conf  = 0.5
    frame_count  = 0
    should_stop = False
    near_threshold = 150

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Stream read failed.")
            break

        # Run inference (you can also pass device='mps'|'cuda'|'cpu')
        

        model = YOLO("yolo11n.pt")
        detections = extract_detections(model, frame, conf_thresh=0.25)

        print("detections:", detections )

        found = any(d["name"] == target_class and d["confidence"] >= target_conf
                    for d in detections)
        if found:
            print(f"✅ Detected {target_class}. Stopping.")
            should_stop = True
        else:
            model = YOLO("runs/detect/train11/weights/best.pt")
            detections_customdata = extract_detections(model, frame, conf_thresh=0.25)
        
        # detections_coco = extract_detections(model, frame, conf_thresh=0.25)
        # # detections_custom = extract_detections(custom_model, frame, conf_thresh=0.25)
        # detections = detections_coco 

        # # Check if target class found
        # targets = [d for d in detections if d["name"] == target_class and d["confidence"] >= target_conf]
        # for target in targets:
        #     near_objects = []
        #     for other in detections:
        #         if other is target:
        #             continue
        #         dist = distance(target["bbox"], other["bbox"])
        #         if dist < near_threshold:
        #             near_objects.append(other["name"])

        #     if near_objects:
        #         near_str = ", ".join(set(near_objects))
        #         print(f"✅ Detected {target_class} near {near_str}")
        #     else:
        #         print(f"✅ Detected {target_class} but no nearby objects")

        #     should_stop = True
        #     break  # stop chec

        if should_stop:
            break

        frame_count += 1
        if frame_count >= max_frames:
            print("⏳ Max frames reached, stopping.")
            break

        

        cv2.imshow("Live Detection (RTSP)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
custom_model = YOLO("runs/detect/train11/weights/best.pt")
detectfromlivefeed(rtsp_url, custom_model, "remote", target_conf)
