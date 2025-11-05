"""
Spatial relationship analysis for object location descriptions.
"""
import numpy as np
from typing import List, Dict, Tuple
from config.settings import SPATIAL_DISTANCE_THRESHOLD, SPATIAL_POSITION_THRESHOLD

class SpatialAnalyzer:
    """Analyzes spatial relationships between detected objects."""

    @staticmethod
    def get_relationship(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int], name1: str, name2: str) -> str:
        x1_center = (bbox1[0] + bbox1[2]) / 2
        y1_center = (bbox1[1] + bbox1[3]) / 2
        x2_center = (bbox2[0] + bbox2[2]) / 2
        y2_center = (bbox2[1] + bbox2[3]) / 2
        dx = x2_center - x1_center
        dy = y2_center - y1_center
        relationships = []
        if abs(dy) > SPATIAL_POSITION_THRESHOLD:
            if dy > 0:
                relationships.append("above")
            else:
                relationships.append("below")
        if abs(dx) > SPATIAL_POSITION_THRESHOLD:
            if dx > 0:
                relationships.append("to the left of")
            else:
                relationships.append("to the right of")
        distance = np.sqrt(dx**2 + dy**2)
        if distance < SPATIAL_DISTANCE_THRESHOLD:
            return f"near the {name2}"
        elif relationships:
            return " and ".join(relationships) + f" the {name2}"
        else:
            return f"near the {name2}"

    @staticmethod
    def generate_description(target_object: str, all_detections: List[Dict], image_shape: Tuple[int, int, int]) -> str:
        if not all_detections:
            return "No objects detected to describe location."
        target_det = None
        other_objects = []
        for det in all_detections:
            if target_object and target_object.lower() in det['class'].lower():
                target_det = det
            else:
                other_objects.append(det)
        if not target_det:
            return f"Could not find {target_object} in the image."
        if not other_objects:
            bbox = target_det['bbox']
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            img_height, img_width = image_shape[:2]
            h_pos = "center"
            if x_center < img_width * 0.33:
                h_pos = "left side"
            elif x_center > img_width * 0.67:
                h_pos = "right side"
            v_pos = "middle"
            if y_center < img_height * 0.33:
                v_pos = "top"
            elif y_center > img_height * 0.67:
                v_pos = "bottom"
            return f"The {target_det['class']} is in the {v_pos} {h_pos} of the image."
        target_bbox = target_det['bbox']
        target_center = np.array([
            (target_bbox[0] + target_bbox[2]) / 2,
            (target_bbox[1] + target_bbox[3]) / 2
        ])
        closest_obj = None
        min_distance = float('inf')
        for obj in other_objects:
            obj_bbox = obj['bbox']
            obj_center = np.array([
                (obj_bbox[0] + obj_bbox[2]) / 2,
                (obj_bbox[1] + obj_bbox[3]) / 2
            ])
            distance = np.linalg.norm(target_center - obj_center)
            if distance < min_distance:
                min_distance = distance
                closest_obj = obj
        if closest_obj:
            relationship = SpatialAnalyzer.get_relationship(
                target_det['bbox'], closest_obj['bbox'], target_det['class'], closest_obj['class'])
            return f"The {target_det['class']} is {relationship}."
        return f"Found the {target_det['class']}."
