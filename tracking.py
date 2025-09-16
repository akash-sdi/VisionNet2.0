
# VisionNet DeepSORT Tracking Module
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import logging
from config import Config

logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class DeepSORTTracker:
    """
    DeepSORT-based multi-object tracking class for VisionNet system
    """

    def __init__(self, max_age=70, n_init=3, max_iou_distance=0.7, max_dist=0.2, max_cosine_distance=0.4):
        """
        Initialize DeepSORT tracker

        Args:
            max_age (int): Maximum number of missed misses before track is deleted
            n_init (int): Number of consecutive detections before track is confirmed
            max_iou_distance (float): Maximum intersection-over-union distance for association
            max_dist (float): Maximum distance threshold for ReID features
            max_cosine_distance (float): Maximum cosine distance for appearance features
        """
        self.max_age = max_age or Config.DEEPSORT_MAX_AGE
        self.n_init = n_init or Config.DEEPSORT_N_INIT
        self.max_iou_distance = max_iou_distance or Config.DEEPSORT_MAX_IOU_DISTANCE
        self.max_dist = max_dist or Config.DEEPSORT_MAX_DIST
        self.max_cosine_distance = max_cosine_distance

        # Initialize DeepSORT
        try:
            self.tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                max_iou_distance=self.max_iou_distance,
                max_cosine_distance=self.max_cosine_distance,
                nn_budget=100,
                override_track_class=None,
                embedder="mobilenet",
                half=True,
                bgr=True,
                embedder_gpu=Config.USE_GPU,
                embedder_model_name=None,
                embedder_wts=None,
                polygon=False,
                today=None
            )
            logger.info("DeepSORT tracker initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DeepSORT: {e}")
            raise

        # Track management
        self.track_history = defaultdict(list)
        self.track_colors = {}
        self.active_tracks = set()

    def update(self, detections, frame=None):
        """
        Update tracker with new detections

        Args:
            detections (list): List of detection dictionaries with keys:
                              - bbox: [x1, y1, x2, y2]
                              - confidence: float
                              - class_id: int
            frame (np.ndarray): Current frame for feature extraction

        Returns:
            list: List of tracked objects with track IDs
        """
        try:
            # Convert detections to DeepSORT format
            if not detections:
                tracks = self.tracker.update_tracks([], frame=frame)
                return self._format_tracks(tracks)

            # Prepare detection data for DeepSORT
            bboxes = []
            confidences = []
            class_ids = []

            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                # Convert to [left, top, width, height] format
                bbox = [x1, y1, x2 - x1, y2 - y1]
                bboxes.append(bbox)
                confidences.append(det['confidence'])
                class_ids.append(det.get('class_id', 0))

            # Update tracker
            tracks = self.tracker.update_tracks(
                raw_detections=list(zip(bboxes, confidences, class_ids)),
                frame=frame
            )

            # Format and return results
            return self._format_tracks(tracks)

        except Exception as e:
            logger.error(f"Tracker update failed: {e}")
            return []

    def _format_tracks(self, tracks):
        """
        Format tracks for consistent output

        Args:
            tracks: DeepSORT track objects

        Returns:
            list: Formatted track dictionaries
        """
        formatted_tracks = []
        current_track_ids = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            current_track_ids.add(track_id)

            # Get bounding box
            ltwh = track.to_ltwh()
            left, top, width, height = ltwh
            bbox = [int(left), int(top), int(left + width), int(top + height)]

            # Calculate center point
            center_x = int(left + width / 2)
            center_y = int(top + height / 2)
            center = (center_x, center_y)

            # Update track history
            self.track_history[track_id].append(center)
            # Keep only recent history (30 frames)
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

            # Generate color for new tracks
            if track_id not in self.track_colors:
                np.random.seed(track_id)
                self.track_colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())

            formatted_track = {
                'track_id': track_id,
                'bbox': bbox,
                'center': center,
                'class_id': track.get_det_class() if hasattr(track, 'get_det_class') else 0,
                'confidence': track.get_det_conf() if hasattr(track, 'get_det_conf') else 1.0,
                'age': track.age,
                'hits': track.hits,
                'time_since_update': track.time_since_update,
                'state': 'confirmed',
                'color': self.track_colors[track_id]
            }

            formatted_tracks.append(formatted_track)

        # Update active tracks
        self.active_tracks = current_track_ids

        # Clean up old track histories
        self._cleanup_old_tracks()

        return formatted_tracks

    def _cleanup_old_tracks(self):
        """Remove history for tracks that are no longer active"""
        to_remove = []
        for track_id in self.track_history:
            if track_id not in self.active_tracks:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.track_history[track_id]
            if track_id in self.track_colors:
                del self.track_colors[track_id]

    def get_track_history(self, track_id=None):
        """
        Get track history

        Args:
            track_id (int, optional): Specific track ID. If None, returns all histories.

        Returns:
            dict or list: Track history/histories
        """
        if track_id is not None:
            return self.track_history.get(track_id, [])
        return dict(self.track_history)

    def visualize_tracks(self, frame, tracks, show_trails=True, trail_length=30):
        """
        Visualize tracks on frame

        Args:
            frame (np.ndarray): Input frame
            tracks (list): List of track dictionaries
            show_trails (bool): Whether to show track trails
            trail_length (int): Length of trails to show

        Returns:
            np.ndarray: Annotated frame
        """
        annotated_frame = frame.copy()

        # Draw trails first (so they appear behind boxes)
        if show_trails:
            for track in tracks:
                track_id = track['track_id']
                if track_id in self.track_history:
                    points = self.track_history[track_id]
                    if len(points) > 1:
                        color = track['color']

                        # Draw trail
                        for i in range(1, min(len(points), trail_length)):
                            thickness = max(1, int(2 * (i / trail_length)))
                            cv2.line(annotated_frame, points[-i-1], points[-i], color, thickness)

        # Draw bounding boxes and labels
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['track_id']
            color = track['color']
            confidence = track.get('confidence', 1.0)

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Prepare label
            label = f"ID:{track_id}"
            if 'class_name' in track:
                label = f"{track['class_name']} {label}"
            if confidence < 1.0:
                label += f" {confidence:.2f}"

            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

            # Draw label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw center point
            center = track['center']
            cv2.circle(annotated_frame, center, 3, color, -1)

        return annotated_frame

    def get_tracker_stats(self):
        """Get tracker statistics"""
        return {
            'active_tracks': len(self.active_tracks),
            'total_track_histories': len(self.track_history),
            'tracker_params': {
                'max_age': self.max_age,
                'n_init': self.n_init,
                'max_iou_distance': self.max_iou_distance,
                'max_dist': self.max_dist,
                'max_cosine_distance': self.max_cosine_distance
            }
        }

    def reset_tracker(self):
        """Reset the tracker and clear all tracks"""
        try:
            # Reinitialize tracker
            self.tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                max_iou_distance=self.max_iou_distance,
                max_cosine_distance=self.max_cosine_distance,
                nn_budget=100
            )

            # Clear histories
            self.track_history.clear()
            self.track_colors.clear()
            self.active_tracks.clear()

            logger.info("Tracker reset successfully")

        except Exception as e:
            logger.error(f"Failed to reset tracker: {e}")

# Example usage combining detection and tracking
if __name__ == "__main__":
    from detection import ObjectDetector

    # Initialize detector and tracker
    detector = ObjectDetector()
    tracker = DeepSORTTracker()

    # Test with webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        detections = detector.detect(frame)

        # Update tracker
        tracks = tracker.update(detections, frame)

        # Visualize results
        annotated_frame = tracker.visualize_tracks(frame, tracks)

        # Display statistics
        stats = tracker.get_tracker_stats()
        cv2.putText(annotated_frame, f"Active Tracks: {stats['active_tracks']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display
        cv2.imshow('VisionNet Tracking', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

