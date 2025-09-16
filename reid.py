"""
VisionNet Re-Identification Module
=================================
OSNet-based person re-identification for cross-camera tracking.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import pickle
import time
from collections import defaultdict

try:
    import torchreid
    TORCHREID_AVAILABLE = True
except ImportError:
    print("Warning: torchreid not installed. Install with: pip install torchreid")
    TORCHREID_AVAILABLE = False

from config import Config

# Setup logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class SimpleReID:
    """Simple ReID implementation when torchreid is not available"""

    def __init__(self, feature_dim: int = 512):
        self.feature_dim = feature_dim
        self.model = self._create_simple_feature_extractor()

    def _create_simple_feature_extractor(self):
        """Create a simple CNN for feature extraction"""
        import torch.nn as nn

        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((8, 4)),
                    nn.Flatten(),
                    nn.Linear(256 * 8 * 4, 512),
                    nn.ReLU(),
                    nn.Linear(512, self.feature_dim)
                )

            def forward(self, x):
                return F.normalize(self.features(x), p=2, dim=1)

        model = SimpleCNN()
        model.eval()
        return model

    def __call__(self, x):
        with torch.no_grad():
            return self.model(x)

class OSNetReID:
    """
    OSNet-based person re-identification for cross-camera tracking.
    Extracts appearance features for identity matching across cameras.
    """

    def __init__(self, model_name: str = 'osnet_x1_0', device: Optional[str] = None):
        """
        Initialize OSNet ReID model.

        Args:
            model_name: OSNet model variant
            device: Device for inference
        """
        self.model_name = model_name
        self.device = "cpu"
        self.model = None
        self.feature_cache = {}
        self.similarity_threshold = Config.OSNET_THRESHOLD

        # Input preprocessing parameters
        self.input_size = (256, 128)  # Standard ReID input size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Performance tracking
        self.extraction_times = []

        self._load_model()

    def _load_model(self):
        """Load OSNet model for feature extraction"""
        try:
            if TORCHREID_AVAILABLE:
                self.model = torchreid.models.build_model(
                    name=self.model_name,
                    num_classes=1000,  # Will be ignored for feature extraction
                    loss='triplet',
                    pretrained=True
                )
                self.model.eval()
                self.model.to(self.device)
                logger.info(f"Loaded OSNet model: {self.model_name}")
                self.use_torchreid = True
            else:
                # Fallback to simple feature extractor
                logger.warning("torchreid not available, using simple feature extractor")
                self.model = SimpleReID()
                self.model.model.to(self.device)
                self.use_torchreid = False

        except Exception as e:
            logger.error(f"Failed to load ReID model: {e}")
            self.model = SimpleReID()
            self.model.model.to(self.device)
            self.use_torchreid = False

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for ReID model.

        Args:
            image: Input image in BGR format

        Returns:
            Preprocessed tensor
        """
        try:
            if image is None or image.size == 0:
                return None

            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Resize to standard ReID input size
            image = cv2.resize(image, self.input_size)

            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0

            # Apply ImageNet normalization
            for i in range(3):
                image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]

            # Convert to tensor and add batch dimension
            tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
            return tensor.to(self.device)

        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None

    def extract_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract ReID features from image.

        Args:
            image: Input image (person crop)

        Returns:
            Feature vector
        """
        start_time = time.time()

        try:
            if image is None or image.size == 0:
                return None

            # Preprocess image
            tensor = self.preprocess_image(image)
            if tensor is None:
                return None

            # Extract features
            with torch.no_grad():
                if self.use_torchreid:
                    features = self.model(tensor)
                else:
                    features = self.model(tensor)

                features = F.normalize(features, p=2, dim=1)
                features = features.cpu().numpy().flatten()

            # Update performance metrics
            extraction_time = time.time() - start_time
            self.extraction_times.append(extraction_time)
            if len(self.extraction_times) > 100:
                self.extraction_times.pop(0)

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None

    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors.

        Args:
            features1: First feature vector
            features2: Second feature vector

        Returns:
            Similarity score (0-1)
        """
        try:
            if features1 is None or features2 is None:
                return 0.0

            # Ensure features are normalized
            features1 = features1 / (np.linalg.norm(features1) + 1e-12)
            features2 = features2 / (np.linalg.norm(features2) + 1e-12)

            # Compute cosine similarity
            similarity = np.dot(features1, features2)

            # Convert to 0-1 range
            similarity = (similarity + 1) / 2

            return float(np.clip(similarity, 0, 1))

        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0

    def match_tracks_across_cameras(self, camera1_tracks: List[Dict], camera2_tracks: List[Dict],
                                  frame1: np.ndarray, frame2: np.ndarray) -> List[Dict]:
        """
        Match tracks between two cameras using ReID features.

        Args:
            camera1_tracks: Tracks from camera 1
            camera2_tracks: Tracks from camera 2
            frame1: Frame from camera 1
            frame2: Frame from camera 2

        Returns:
            List of matched track pairs with similarity scores
        """
        matches = []

        try:
            if not camera1_tracks or not camera2_tracks:
                return matches

            # Extract features for camera 1 tracks
            features1 = {}
            for track in camera1_tracks:
                x1, y1, x2, y2 = track['bbox']
                # Ensure valid crop
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(frame1.shape[1], x2)
                y2 = min(frame1.shape[0], y2)

                if x2 > x1 and y2 > y1:
                    crop = frame1[y1:y2, x1:x2]
                    features = self.extract_features(crop)
                    if features is not None:
                        features1[track['track_id']] = features

            # Extract features for camera 2 tracks
            features2 = {}
            for track in camera2_tracks:
                x1, y1, x2, y2 = track['bbox']
                # Ensure valid crop
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(frame2.shape[1], x2)
                y2 = min(frame2.shape[0], y2)

                if x2 > x1 and y2 > y1:
                    crop = frame2[y1:y2, x1:x2]
                    features = self.extract_features(crop)
                    if features is not None:
                        features2[track['track_id']] = features

            # Compare all pairs
            for id1, feat1 in features1.items():
                for id2, feat2 in features2.items():
                    similarity = self.compute_similarity(feat1, feat2)

                    if similarity > self.similarity_threshold:
                        matches.append({
                            'camera1_id': id1,
                            'camera2_id': id2,
                            'similarity': similarity,
                            'timestamp': time.time()
                        })

            # Sort by similarity (highest first)
            matches.sort(key=lambda x: x['similarity'], reverse=True)

            # Remove conflicts (ensure one-to-one matching)
            used_cam1 = set()
            used_cam2 = set()
            final_matches = []

            for match in matches:
                cam1_id = match['camera1_id']
                cam2_id = match['camera2_id']

                if cam1_id not in used_cam1 and cam2_id not in used_cam2:
                    final_matches.append(match)
                    used_cam1.add(cam1_id)
                    used_cam2.add(cam2_id)

            return final_matches

        except Exception as e:
            logger.error(f"Cross-camera matching failed: {e}")
            return []

    def get_performance_stats(self) -> Dict:
        """Get ReID performance statistics"""
        avg_extraction_time = np.mean(self.extraction_times) if self.extraction_times else 0
        fps = 1.0 / avg_extraction_time if avg_extraction_time > 0 else 0

        return {
            'model_name': self.model_name,
            'device': self.device,
            'similarity_threshold': self.similarity_threshold,
            'average_extraction_time': avg_extraction_time,
            'extraction_fps': fps,
            'cache_size': len(self.feature_cache),
            'use_torchreid': getattr(self, 'use_torchreid', False)
        }

class CrossCameraTracker:
    """
    Manages cross-camera tracking using ReID features.
    Maintains global identities across multiple camera views.
    """

    def __init__(self, reid_model: Optional[OSNetReID] = None):
        """
        Initialize cross-camera tracker.

        Args:
            reid_model: OSNetReID instance
        """
        self.reid_model = reid_model or OSNetReID()
        self.global_id_map = {}  # Maps (camera_id, local_id) -> global_id
        self.camera_tracks = {}  # Stores latest tracks for each camera
        self.cross_camera_matches = []
        self.next_global_id = 1
        self.match_history = defaultdict(list)

        # Temporal constraints
        self.temporal_window = Config.TEMPORAL_THRESHOLD / Config.VIDEO_FPS  # seconds
        self.max_global_id_age = Config.GLOBAL_ID_TIMEOUT  # seconds

    def update_camera_tracks(self, camera_id: int, tracks: List[Dict], 
                           frame: np.ndarray) -> List[Dict]:
        """
        Update tracks for a specific camera.

        Args:
            camera_id: Camera identifier
            tracks: List of tracks from this camera
            frame: Current frame from this camera

        Returns:
            Tracks with updated global IDs
        """
        timestamp = time.time()

        # Store tracks for this camera
        self.camera_tracks[camera_id] = {
            'tracks': tracks,
            'frame': frame,
            'timestamp': timestamp
        }

        # Update global IDs
        updated_tracks = []
        for track in tracks:
            local_id = track['track_id']
            camera_local_key = (camera_id, local_id)

            # Check if this track has a global ID
            global_id = self.global_id_map.get(camera_local_key)

            if global_id is None:
                # Assign new global ID
                global_id = self.next_global_id
                self.global_id_map[camera_local_key] = global_id
                self.next_global_id += 1

            # Update track with global ID
            updated_track = track.copy()
            updated_track['global_id'] = global_id
            updated_track['local_id'] = local_id
            updated_track['camera_id'] = camera_id
            updated_track['timestamp'] = timestamp

            updated_tracks.append(updated_track)

        # Clean up old global IDs
        self._cleanup_old_global_ids(timestamp)

        return updated_tracks

    def find_cross_camera_matches(self, temporal_window: Optional[float] = None) -> List[Dict]:
        """
        Find matches between tracks from different cameras.

        Args:
            temporal_window: Time window for matching (seconds)

        Returns:
            Cross-camera matches
        """
        temporal_window = temporal_window or self.temporal_window
        current_time = time.time()
        matches = []

        # Get active cameras with recent data
        active_cameras = []
        for cam_id, data in self.camera_tracks.items():
            if current_time - data['timestamp'] <= temporal_window:
                active_cameras.append(cam_id)

        # Compare tracks between all camera pairs
        for i in range(len(active_cameras)):
            for j in range(i + 1, len(active_cameras)):
                cam1_id = active_cameras[i]
                cam2_id = active_cameras[j]

                cam1_data = self.camera_tracks[cam1_id]
                cam2_data = self.camera_tracks[cam2_id]

                # Find matches between these cameras
                camera_matches = self.reid_model.match_tracks_across_cameras(
                    cam1_data['tracks'],
                    cam2_data['tracks'],
                    cam1_data['frame'],
                    cam2_data['frame']
                )

                # Add camera information to matches
                for match in camera_matches:
                    match['camera1'] = cam1_id
                    match['camera2'] = cam2_id
                    matches.append(match)

        # Store match history
        for match in matches:
            key = (match['camera1'], match['camera1_id'], 
                   match['camera2'], match['camera2_id'])
            self.match_history[key].append({
                'similarity': match['similarity'],
                'timestamp': match['timestamp']
            })

        self.cross_camera_matches = matches
        return matches

    def _cleanup_old_global_ids(self, current_time: float):
        """Clean up old global ID mappings"""
        to_remove = []

        for camera_local_key, global_id in self.global_id_map.items():
            camera_id, local_id = camera_local_key

            # Check if camera still has recent data
            if camera_id in self.camera_tracks:
                camera_data = self.camera_tracks[camera_id]
                if current_time - camera_data['timestamp'] > self.max_global_id_age:
                    # Check if local track still exists
                    track_exists = any(
                        track['track_id'] == local_id 
                        for track in camera_data['tracks']
                    )
                    if not track_exists:
                        to_remove.append(camera_local_key)
            else:
                to_remove.append(camera_local_key)

        for key in to_remove:
            del self.global_id_map[key]

    def get_global_track_summary(self) -> Dict:
        """Get summary of all global tracks across cameras"""
        current_time = time.time()

        # Count active cameras
        active_cameras = sum(
            1 for data in self.camera_tracks.values()
            if current_time - data['timestamp'] <= self.temporal_window
        )

        # Count total tracks
        total_tracks = sum(
            len(data['tracks']) for data in self.camera_tracks.values()
            if current_time - data['timestamp'] <= self.temporal_window
        )

        # Count unique global IDs
        active_global_ids = set(self.global_id_map.values())

        summary = {
            'total_cameras': len(self.camera_tracks),
            'active_cameras': active_cameras,
            'total_local_tracks': total_tracks,
            'unique_global_tracks': len(active_global_ids),
            'cross_camera_matches': len(self.cross_camera_matches),
            'next_global_id': self.next_global_id,
            'cameras_status': {}
        }

        # Add per-camera status
        for cam_id, data in self.camera_tracks.items():
            is_active = current_time - data['timestamp'] <= self.temporal_window
            summary['cameras_status'][cam_id] = {
                'active_tracks': len(data['tracks']) if is_active else 0,
                'last_update': data['timestamp'],
                'is_active': is_active
            }

        return summary

    def reset(self):
        """Reset the cross-camera tracker"""
        self.global_id_map.clear()
        self.camera_tracks.clear()
        self.cross_camera_matches.clear()
        self.match_history.clear()
        self.next_global_id = 1
        logger.info("Cross-camera tracker reset")

class FeatureDatabase:
    """
    Manages persistent storage of ReID features for long-term matching.
    """

    def __init__(self, db_path: str = "reid_features.db"):
        """
        Initialize feature database.

        Args:
            db_path: Path to database file
        """
        self.db_path = Path(db_path)
        self.features = {}
        self.metadata = {}
        self.load_database()

    def add_feature(self, track_id: int, camera_id: int, features: np.ndarray, 
                   timestamp: Optional[float] = None):
        """Add feature vector to database"""
        timestamp = timestamp or time.time()
        key = f"cam_{camera_id}_track_{track_id}"

        self.features[key] = {
            'features': features,
            'timestamp': timestamp,
            'track_id': track_id,
            'camera_id': camera_id
        }

    def search_similar_features(self, query_features: np.ndarray, 
                              threshold: float = 0.7, exclude_camera: Optional[int] = None) -> List[Dict]:
        """
        Search for similar features in database.

        Args:
            query_features: Query feature vector
            threshold: Similarity threshold
            exclude_camera: Camera ID to exclude from search

        Returns:
            Similar features with metadata
        """
        similar = []

        for key, data in self.features.items():
            if exclude_camera and data['camera_id'] == exclude_camera:
                continue

            stored_features = data['features']
            if stored_features is not None and query_features is not None:
                # Compute cosine similarity
                similarity = np.dot(query_features, stored_features) / (
                    np.linalg.norm(query_features) * np.linalg.norm(stored_features) + 1e-12
                )

                if similarity > threshold:
                    similar.append({
                        'similarity': float(similarity),
                        'track_id': data['track_id'],
                        'camera_id': data['camera_id'],
                        'timestamp': data['timestamp'],
                        'key': key
                    })

        # Sort by similarity
        similar.sort(key=lambda x: x['similarity'], reverse=True)
        return similar

    def save_database(self):
        """Save database to file"""
        try:
            data = {
                'features': self.features,
                'metadata': self.metadata
            }
            with open(self.db_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Feature database saved to {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to save database: {e}")

    def load_database(self):
        """Load database from file"""
        try:
            if self.db_path.exists():
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                self.features = data.get('features', {})
                self.metadata = data.get('metadata', {})
                logger.info(f"Feature database loaded from {self.db_path}")
            else:
                logger.info("No existing feature database found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            self.features = {}
            self.metadata = {}

    def cleanup_old_features(self, max_age_hours: int = 24):
        """Remove features older than specified age"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        to_remove = []
        for key, data in self.features.items():
            if current_time - data['timestamp'] > max_age_seconds:
                to_remove.append(key)

        for key in to_remove:
            del self.features[key]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old features")

# Test function
def test_reid():
    """Test the ReID module"""
    print("Testing ReID Module...")

    # Initialize ReID model
    reid_model = OSNetReID()

    # Test feature extraction
    dummy_image = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
    features = reid_model.extract_features(dummy_image)

    if features is not None:
        print(f"✓ Feature extraction successful: {features.shape}")

        # Test similarity computation
        features2 = reid_model.extract_features(dummy_image)
        similarity = reid_model.compute_similarity(features, features2)
        print(f"✓ Similarity computation: {similarity:.3f}")
    else:
        print("✗ Feature extraction failed")

    # Test cross-camera tracker
    cross_tracker = CrossCameraTracker(reid_model)

    # Test with dummy tracks
    dummy_tracks = [{
        'track_id': 1,
        'bbox': [100, 100, 200, 300],
        'confidence': 0.8,
        'class_id': 0
    }]

    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    updated_tracks = cross_tracker.update_camera_tracks(0, dummy_tracks, dummy_frame)
    print(f"✓ Cross-camera tracking: {len(updated_tracks)} tracks with global IDs")

    # Test feature database
    feature_db = FeatureDatabase()
    if features is not None:
        feature_db.add_feature(1, 0, features)
        similar = feature_db.search_similar_features(features, threshold=0.5)
        print(f"✓ Feature database: {len(similar)} similar features found")

    # Print performance stats
    stats = reid_model.get_performance_stats()
    print(f"✓ ReID performance: {stats}")

if __name__ == "__main__":
    test_reid()
