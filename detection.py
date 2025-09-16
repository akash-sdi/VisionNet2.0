"""
VisionNet Object Detection Module
================================
YOLOv8-based real-time object detection for multi-camera tracking.
"""

import cv2
import numpy as np
import torch
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time

try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")
    YOLO = None

from config import Config

# Setup logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    YOLOv8-based object detection system for VisionNet.
    Supports real-time detection with configurable parameters.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize object detector.

        Args:
            model_path: Path to YOLOv8 model weights
            device: Device for inference ('cpu', 'cuda', 'auto')
        """
        self.model_path = model_path or Config.YOLO_MODEL
        self.device = device or Config.YOLO_DEVICE
        self.confidence = Config.YOLO_CONFIDENCE
        self.iou_threshold = Config.YOLO_IOU
        self.track_classes = Config.TRACK_CLASSES

        # Performance tracking
        self.inference_times = []
        self.frame_count = 0

        # Initialize model
        self.model = None
        self.class_names = {}
        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model"""
        try:
            if YOLO is None:
                raise ImportError("ultralytics package not available")

            logger.info(f"Loading YOLOv8 model: {self.model_path}")
            self.model = YOLO(self.model_path)

            # Set device
            if torch.cuda.is_available() and 'cuda' in self.device:
                self.model.to('cuda')
                logger.info("Using GPU acceleration")
            else:
                self.model.to('cpu')
                logger.info("Using CPU inference")

            # Get class names
            self.class_names = self.model.names
            logger.info(f"Model loaded successfully. Classes: {len(self.class_names)}")

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            # Create dummy model for testing
            self._create_dummy_model()

    def _create_dummy_model(self):
        """Create dummy model for testing when YOLO is not available"""
        logger.warning("Creating dummy detector for testing")
        self.class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle'}
        self.model = None

    def detect(self, frame: np.ndarray, return_crops: bool = False) -> List[Dict]:
        """
        Perform object detection on a frame.

        Args:
            frame: Input image frame
            return_crops: Whether to return cropped detection images

        Returns:
            List of detection dictionaries
        """
        start_time = time.time()

        try:
            if self.model is None:
                # Return dummy detections for testing
                return self._dummy_detections(frame, return_crops)

            # Run YOLOv8 inference
            results = self.model(
                frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                classes=self.track_classes,
                verbose=False
            )

            detections = []

            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    # Extract detection data
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)

                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        x1, y1, x2, y2 = map(int, box)

                        # Ensure coordinates are valid
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)

                        if x2 <= x1 or y2 <= y1:
                            continue

                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'class_id': int(class_id),
                            'class_name': self.class_names.get(class_id, 'unknown'),
                            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                            'area': (x2 - x1) * (y2 - y1)
                        }

                        # Add cropped image if requested
                        if return_crops:
                            crop = frame[y1:y2, x1:x2]
                            detection['crop'] = crop

                        detections.append(detection)

            # Update performance metrics
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)

            self.frame_count += 1

            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def _dummy_detections(self, frame: np.ndarray, return_crops: bool = False) -> List[Dict]:
        """Generate dummy detections for testing"""
        height, width = frame.shape[:2]

        # Create some dummy detections
        detections = []

        # Add a dummy person detection in the center
        if np.random.random() > 0.5:  # 50% chance
            x1 = width // 4
            y1 = height // 4
            x2 = 3 * width // 4
            y2 = 3 * height // 4

            detection = {
                'bbox': [x1, y1, x2, y2],
                'confidence': 0.8 + np.random.random() * 0.2,
                'class_id': 0,
                'class_name': 'person',
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'area': (x2 - x1) * (y2 - y1)
            }

            if return_crops:
                detection['crop'] = frame[y1:y2, x1:x2]

            detections.append(detection)

        return detections

    def detect_and_track(self, frame: np.ndarray, track_history: Optional[Dict] = None) -> List[Dict]:
        """
        Perform detection with YOLO's built-in tracking.

        Args:
            frame: Input frame
            track_history: Dictionary to store track history

        Returns:
            List of tracked detections
        """
        try:
            if self.model is None:
                return self._dummy_tracked_detections(frame, track_history)

            # Use YOLO's built-in tracking
            results = self.model.track(
                frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                classes=self.track_classes,
                persist=True,
                verbose=False
            )

            tracked_detections = []

            if results and len(results) > 0:
                result = results[0]

                if (result.boxes is not None and 
                    hasattr(result.boxes, 'id') and 
                    result.boxes.id is not None):

                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    track_ids = result.boxes.id.cpu().numpy().astype(int)

                    for box, conf, class_id, track_id in zip(boxes, confidences, class_ids, track_ids):
                        x1, y1, x2, y2 = map(int, box)

                        # Ensure valid coordinates
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)

                        if x2 <= x1 or y2 <= y1:
                            continue

                        center = ((x1 + x2) // 2, (y1 + y2) // 2)

                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'class_id': int(class_id),
                            'class_name': self.class_names.get(class_id, 'unknown'),
                            'track_id': int(track_id),
                            'center': center,
                            'area': (x2 - x1) * (y2 - y1)
                        }

                        tracked_detections.append(detection)

                        # Update track history
                        if track_history is not None:
                            if track_id not in track_history:
                                track_history[track_id] = []

                            track_history[track_id].append(center)

                            # Keep only recent history
                            if len(track_history[track_id]) > Config.TRACK_BUFFER_SIZE:
                                track_history[track_id].pop(0)

            return tracked_detections

        except Exception as e:
            logger.error(f"Track detection failed: {e}")
            return []

    def _dummy_tracked_detections(self, frame: np.ndarray, track_history: Optional[Dict] = None) -> List[Dict]:
        """Generate dummy tracked detections for testing"""
        detections = self._dummy_detections(frame)

        # Add dummy track IDs
        for i, detection in enumerate(detections):
            detection['track_id'] = i + 1

            if track_history is not None:
                track_id = detection['track_id']
                if track_id not in track_history:
                    track_history[track_id] = []

                track_history[track_id].append(detection['center'])
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)

        return detections

    def visualize_detections(self, frame: np.ndarray, detections: List[Dict], 
                           show_conf: bool = True, show_class: bool = True,
                           line_thickness: int = 2) -> np.ndarray:
        """
        Draw detection bounding boxes on frame.

        Args:
            frame: Input frame
            detections: List of detection dictionaries
            show_conf: Whether to show confidence scores
            show_class: Whether to show class names
            line_thickness: Thickness of bounding box lines

        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            class_name = detection['class_name']

            # Choose color based on class
            class_id = detection.get('class_id', 0)
            color = self._get_class_color(class_id)

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, line_thickness)

            # Prepare label
            label_parts = []
            if show_class:
                label_parts.append(class_name)
            if show_conf:
                label_parts.append(f"{conf:.2f}")
            if 'track_id' in detection:
                label_parts.append(f"ID:{detection['track_id']}")

            label = " ".join(label_parts)

            # Draw label background and text
            if label:
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )

                # Draw label background
                cv2.rectangle(
                    annotated_frame, 
                    (x1, y1 - label_h - baseline - 5), 
                    (x1 + label_w, y1), 
                    color, -1
                )

                # Draw label text
                cv2.putText(
                    annotated_frame, label, (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )

        return annotated_frame

    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for a specific class"""
        colors = [
            (255, 0, 0),    # Red for person
            (0, 255, 0),    # Green for bicycle
            (0, 0, 255),    # Blue for car
            (255, 255, 0),  # Cyan for motorcycle
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        return colors[class_id % len(colors)]

    def draw_tracks(self, frame: np.ndarray, track_history: Dict, 
                   max_trail_length: int = 30) -> np.ndarray:
        """
        Draw tracking trails on frame.

        Args:
            frame: Input frame
            track_history: Dictionary of track histories
            max_trail_length: Maximum length of trails to draw

        Returns:
            Frame with track trails
        """
        annotated_frame = frame.copy()

        for track_id, points in track_history.items():
            if len(points) > 1:
                # Generate consistent color for track
                np.random.seed(track_id)
                color = tuple(np.random.randint(0, 255, 3).tolist())

                # Draw trail
                recent_points = points[-max_trail_length:]
                for i in range(1, len(recent_points)):
                    # Vary thickness based on recency
                    thickness = max(1, int(3 * (i / len(recent_points))))
                    cv2.line(annotated_frame, recent_points[i-1], recent_points[i], color, thickness)

                # Draw current position
                if points:
                    cv2.circle(annotated_frame, points[-1], 5, color, -1)
                    cv2.putText(
                        annotated_frame, f"ID:{track_id}", 
                        (points[-1][0] + 10, points[-1][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )

        return annotated_frame

    def get_performance_stats(self) -> Dict:
        """Get detection performance statistics"""
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

        return {
            'frames_processed': self.frame_count,
            'average_inference_time': avg_inference_time,
            'fps': fps,
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence,
            'iou_threshold': self.iou_threshold,
            'track_classes': self.track_classes
        }

    def reset_stats(self):
        """Reset performance statistics"""
        self.inference_times.clear()
        self.frame_count = 0

# Test function
def test_detector():
    """Test the object detector"""
    print("Testing Object Detector...")

    # Initialize detector
    detector = ObjectDetector()

    # Create test image
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Test detection
    detections = detector.detect(test_frame, return_crops=True)
    print(f"Found {len(detections)} detections")

    # Test tracking
    track_history = {}
    tracked = detector.detect_and_track(test_frame, track_history)
    print(f"Tracked {len(tracked)} objects")

    # Test visualization
    annotated = detector.visualize_detections(test_frame, detections)
    print("Visualization test completed")

    # Print stats
    stats = detector.get_performance_stats()
    print(f"Performance stats: {stats}")

if __name__ == "__main__":
    test_detector()
