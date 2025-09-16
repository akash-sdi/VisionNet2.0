"""
VisionNet Multi-Camera Management Module
=======================================
Manages multiple camera streams and coordinates tracking across cameras.
"""

import cv2
import numpy as np
import threading
import queue
import time
import logging
from typing import List, Dict, Optional, Any
from collections import defaultdict, deque
from pathlib import Path
import json

from config import Config
from detection import ObjectDetector
from tracking import DeepSORTTracker
from reid import OSNetReID, CrossCameraTracker

# Setup logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class CameraStream:
    """
    Individual camera stream handler with threaded processing.
    """

    def __init__(self, camera_id: int, source: Any, detector: Optional[ObjectDetector] = None,
                 tracker: Optional[DeepSORTTracker] = None):
        """
        Initialize camera stream.

        Args:
            camera_id: Unique camera identifier
            source: Camera source (int for webcam, str for file/RTSP)
            detector: ObjectDetector instance
            tracker: DeepSORTTracker instance
        """
        self.camera_id = camera_id
        self.source = source
        self.detector = detector or ObjectDetector()
        self.tracker = tracker or DeepSORTTracker()

        # Video capture
        self.cap = None
        self.is_running = False
        self.thread = None
        self._stop_event = threading.Event()

        # Frame processing
        self.frame_queue = queue.Queue(maxsize=5)
        self.detection_queue = queue.Queue(maxsize=20)
        self.track_queue = queue.Queue(maxsize=20)

        # Statistics
        self.stats = {
            'frames_processed': 0,
            'detections_count': 0,
            'tracks_count': 0,
            'fps': 0.0,
            'last_update': time.time(),
            'errors': 0,
            'dropped_frames': 0
        }

        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.last_frame_time = time.time()

        # Video recording
        self.video_writer = None
        self.save_video = Config.SAVE_VIDEO

    def start(self) -> bool:
        """Start camera stream processing"""
        try:
            logger.info(f"Starting camera {self.camera_id} with source: {self.source}")

            # Initialize video capture
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise Exception(f"Failed to open camera source: {self.source}")

            # Set camera properties
            self._configure_camera()

            # Test frame read
            ret, test_frame = self.cap.read()
            if not ret:
                raise Exception("Failed to read test frame from camera")

            logger.info(f"Camera {self.camera_id} test frame: {test_frame.shape}")

            # Initialize video writer if needed
            if self.save_video:
                self._init_video_writer(test_frame.shape)

            # Start processing thread
            self.is_running = True
            self._stop_event.clear()
            self.thread = threading.Thread(target=self._process_stream, daemon=True)
            self.thread.start()

            logger.info(f"Camera {self.camera_id} started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start camera {self.camera_id}: {e}")
            self._cleanup()
            return False

    def _configure_camera(self):
        """Configure camera properties"""
        try:
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

            # Set FPS
            self.cap.set(cv2.CAP_PROP_FPS, Config.VIDEO_FPS)

            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            logger.info(f"Camera {self.camera_id} configured")

        except Exception as e:
            logger.warning(f"Failed to configure camera {self.camera_id}: {e}")

    def _init_video_writer(self, frame_shape: tuple):
        """Initialize video writer for recording"""
        try:
            output_dir = Config.DATA_DIR / "recordings"
            output_dir.mkdir(exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"camera_{self.camera_id}_{timestamp}.mp4"

            height, width = frame_shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*Config.VIDEO_CODEC)

            self.video_writer = cv2.VideoWriter(
                str(output_path), fourcc, Config.VIDEO_FPS, (width, height)
            )

            logger.info(f"Video recording initialized for camera {self.camera_id}: {output_path}")

        except Exception as e:
            logger.error(f"Failed to initialize video writer for camera {self.camera_id}: {e}")

    def stop(self):
        """Stop camera stream processing"""
        logger.info(f"Stopping camera {self.camera_id}")

        self.is_running = False
        self._stop_event.set()

        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                logger.warning(f"Camera {self.camera_id} thread did not stop gracefully")

        self._cleanup()
        logger.info(f"Camera {self.camera_id} stopped")

    def _cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
            self.cap = None

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        # Clear queues
        self._clear_queue(self.frame_queue)
        self._clear_queue(self.detection_queue)
        self._clear_queue(self.track_queue)

    def _clear_queue(self, q: queue.Queue):
        """Clear a queue"""
        try:
            while not q.empty():
                q.get_nowait()
        except queue.Empty:
            pass

    def _process_stream(self):
        """Main processing loop for camera stream"""
        frame_count = 0
        consecutive_failures = 0
        max_failures = 10

        logger.info(f"Starting processing loop for camera {self.camera_id}")

        while self.is_running and not self._stop_event.is_set():
            try:
                start_time = time.time()

                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        logger.error(f"Camera {self.camera_id}: Too many consecutive frame read failures")
                        break
                    logger.warning(f"Camera {self.camera_id}: Failed to read frame (attempt {consecutive_failures})")
                    time.sleep(0.1)
                    continue

                consecutive_failures = 0
                frame_count += 1

                # Resize frame if needed
                if frame.shape[1] != Config.RESIZE_WIDTH or frame.shape[0] != Config.RESIZE_HEIGHT:
                    frame = cv2.resize(frame, (Config.RESIZE_WIDTH, Config.RESIZE_HEIGHT))

                # Add timestamp
                timestamp = time.time()

                # Record video if enabled
                if self.video_writer:
                    self.video_writer.write(frame)

                # Detect objects
                detections = self.detector.detect(frame, return_crops=True)

                # Update tracker
                tracks = self.tracker.update(detections, frame)

                # Update queues (non-blocking)
                frame_data = {
                    'frame': frame.copy(),
                    'timestamp': timestamp,
                    'frame_id': frame_count
                }

                detection_data = {
                    'detections': detections,
                    'timestamp': timestamp,
                    'frame_id': frame_count
                }

                track_data = {
                    'tracks': tracks,
                    'timestamp': timestamp,
                    'frame_id': frame_count
                }

                # Try to add to queues, drop if full
                self._add_to_queue(self.frame_queue, frame_data)
                self._add_to_queue(self.detection_queue, detection_data)
                self._add_to_queue(self.track_queue, track_data)

                # Update statistics
                self._update_stats(frame_count, len(detections), len(tracks), start_time, timestamp)

                # Control processing rate
                self._control_frame_rate(start_time)

            except Exception as e:
                logger.error(f"Camera {self.camera_id} processing error: {e}")
                self.stats['errors'] += 1
                time.sleep(0.1)

        logger.info(f"Processing loop ended for camera {self.camera_id}")

    def _add_to_queue(self, q: queue.Queue, data: Dict):
        """Add data to queue, drop oldest if full"""
        try:
            q.put_nowait(data)
        except queue.Full:
            try:
                q.get_nowait()  # Remove oldest
                q.put_nowait(data)  # Add new
                self.stats['dropped_frames'] += 1
            except queue.Empty:
                pass

    def _update_stats(self, frame_count: int, detections_count: int, tracks_count: int,
                     start_time: float, timestamp: float):
        """Update camera statistics"""
        process_time = time.time() - start_time
        fps = 1.0 / process_time if process_time > 0 else 0.0

        self.fps_counter.append(fps)

        self.stats.update({
            'frames_processed': frame_count,
            'detections_count': detections_count,
            'tracks_count': tracks_count,
            'fps': np.mean(self.fps_counter),
            'last_update': timestamp,
            'process_time': process_time
        })

    def _control_frame_rate(self, start_time: float):
        """Control processing frame rate"""
        target_frame_time = 1.0 / Config.VIDEO_FPS
        process_time = time.time() - start_time

        if process_time < target_frame_time:
            sleep_time = target_frame_time - process_time
            time.sleep(sleep_time)

    def get_latest_frame(self) -> Optional[Dict]:
        """Get the latest frame from queue"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def get_latest_tracks(self) -> Optional[Dict]:
        """Get the latest tracks from queue"""
        try:
            return self.track_queue.get_nowait()
        except queue.Empty:
            return None

    def get_latest_detections(self) -> Optional[Dict]:
        """Get the latest detections from queue"""
        try:
            return self.detection_queue.get_nowait()
        except queue.Empty:
            return None

    def get_stats(self) -> Dict:
        """Get camera statistics"""
        stats = self.stats.copy()
        stats['is_running'] = self.is_running
        stats['queue_sizes'] = {
            'frames': self.frame_queue.qsize(),
            'detections': self.detection_queue.qsize(),
            'tracks': self.track_queue.qsize()
        }
        return stats

class MultiCameraManager:
    """
    Manages multiple camera streams and coordinates cross-camera tracking.
    """

    def __init__(self, camera_sources: Optional[List] = None):
        """
        Initialize multi-camera manager.

        Args:
            camera_sources: List of camera sources
        """
        self.camera_sources = camera_sources or Config.CAMERA_SOURCES
        self.cameras = {}
        self.cross_camera_tracker = CrossCameraTracker()

        # System state
        self.is_running = False
        self.fusion_thread = None
        self._stop_fusion = threading.Event()

        # Data fusion
        self.fusion_method = Config.FUSION_METHOD
        self.global_tracks = {}
        self.system_stats = {
            'total_cameras': 0,
            'active_cameras': 0,
            'total_tracks': 0,
            'global_tracks': 0,
            'system_fps': 0.0,
            'uptime': 0.0
        }

        self.start_time = time.time()

    def add_camera(self, camera_id: int, source: Any) -> bool:
        """
        Add a new camera to the system.

        Args:
            camera_id: Unique camera identifier
            source: Camera source

        Returns:
            True if camera added successfully
        """
        try:
            detector = ObjectDetector()
            tracker = DeepSORTTracker()
            camera = CameraStream(camera_id, source, detector, tracker)

            self.cameras[camera_id] = camera
            logger.info(f"Added camera {camera_id} with source: {source}")
            return True

        except Exception as e:
            logger.error(f"Failed to add camera {camera_id}: {e}")
            return False

    def start_all_cameras(self) -> bool:
        """Start all camera streams"""
        success_count = 0

        # Add cameras from sources
        for i, source in enumerate(self.camera_sources):
            if self.add_camera(i, source):
                success_count += 1

        if success_count == 0:
            logger.error("No cameras could be added")
            return False

        # Start camera streams
        started_count = 0
        for camera_id, camera in self.cameras.items():
            if camera.start():
                started_count += 1
                logger.info(f"Started camera {camera_id}")
            else:
                logger.error(f"Failed to start camera {camera_id}")

        if started_count == 0:
            logger.error("No cameras could be started")
            return False

        # Start fusion thread
        self.is_running = True
        self._stop_fusion.clear()
        self.fusion_thread = threading.Thread(target=self._fusion_loop, daemon=True)
        self.fusion_thread.start()

        logger.info(f"Multi-camera system started: {started_count}/{len(self.cameras)} cameras active")
        return True

    def stop_all_cameras(self):
        """Stop all camera streams"""
        logger.info("Stopping multi-camera system")

        # Stop fusion thread
        self.is_running = False
        self._stop_fusion.set()

        if self.fusion_thread and self.fusion_thread.is_alive():
            self.fusion_thread.join(timeout=5.0)

        # Stop all cameras
        for camera_id, camera in self.cameras.items():
            try:
                camera.stop()
                logger.info(f"Stopped camera {camera_id}")
            except Exception as e:
                logger.error(f"Error stopping camera {camera_id}: {e}")

        logger.info("Multi-camera system stopped")

    def _fusion_loop(self):
        """Main data fusion loop"""
        logger.info("Starting data fusion loop")

        while self.is_running and not self._stop_fusion.is_set():
            try:
                start_time = time.time()

                # Update fusion data
                self._update_fusion_data()

                # Update system stats
                self._update_system_stats()

                # Control fusion rate
                fusion_time = time.time() - start_time
                target_time = 1.0 / 30  # 30 Hz fusion rate

                if fusion_time < target_time:
                    time.sleep(target_time - fusion_time)

            except Exception as e:
                logger.error(f"Fusion loop error: {e}")
                time.sleep(0.1)

        logger.info("Data fusion loop stopped")

    def _update_fusion_data(self):
        """Update fused tracking data from all cameras"""
        current_time = time.time()
        all_camera_tracks = {}

        # Collect tracks from all cameras
        for camera_id, camera in self.cameras.items():
            if not camera.is_running:
                continue

            latest_tracks = camera.get_latest_tracks()
            latest_frame_data = camera.get_latest_frame()

            if latest_tracks and latest_frame_data:
                # Update cross-camera tracker
                updated_tracks = self.cross_camera_tracker.update_camera_tracks(
                    camera_id, 
                    latest_tracks['tracks'], 
                    latest_frame_data['frame']
                )

                all_camera_tracks[camera_id] = {
                    'tracks': updated_tracks,
                    'timestamp': latest_tracks['timestamp'],
                    'frame': latest_frame_data['frame']
                }

        # Find cross-camera matches
        matches = self.cross_camera_tracker.find_cross_camera_matches()

        # Update global tracks
        self._update_global_tracks(all_camera_tracks, matches)

    def _update_global_tracks(self, camera_tracks: Dict, matches: List[Dict]):
        """Update global track information"""
        self.global_tracks = {}

        # Aggregate tracks by global ID
        for camera_id, data in camera_tracks.items():
            for track in data['tracks']:
                global_id = track.get('global_id', track['track_id'])

                if global_id not in self.global_tracks:
                    self.global_tracks[global_id] = {
                        'global_id': global_id,
                        'cameras': {},
                        'last_seen': track.get('timestamp', time.time()),
                        'total_detections': 0,
                        'confidence': 0.0
                    }

                self.global_tracks[global_id]['cameras'][camera_id] = track
                self.global_tracks[global_id]['total_detections'] += 1
                self.global_tracks[global_id]['confidence'] = max(
                    self.global_tracks[global_id]['confidence'],
                    track.get('confidence', 0.0)
                )

    def _update_system_stats(self):
        """Update overall system statistics"""
        current_time = time.time()
        active_cameras = 0
        total_tracks = 0
        total_fps = 0.0

        for camera in self.cameras.values():
            stats = camera.get_stats()
            if stats['is_running'] and current_time - stats['last_update'] < 5.0:
                active_cameras += 1
                total_tracks += stats['tracks_count']
                total_fps += stats['fps']

        self.system_stats.update({
            'total_cameras': len(self.cameras),
            'active_cameras': active_cameras,
            'total_tracks': total_tracks,
            'global_tracks': len(self.global_tracks),
            'system_fps': total_fps,
            'uptime': current_time - self.start_time
        })

    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        camera_status = {}
        for camera_id, camera in self.cameras.items():
            camera_status[camera_id] = camera.get_stats()

        cross_camera_summary = self.cross_camera_tracker.get_global_track_summary()

        return {
            'system_stats': self.system_stats,
            'camera_status': camera_status,
            'cross_camera_summary': cross_camera_summary,
            'global_tracks': len(self.global_tracks),
            'fusion_method': self.fusion_method,
            'is_running': self.is_running
        }

    def get_visualization_data(self) -> Dict:
        """Get data for dashboard visualization"""
        visualization_data = {
            'cameras': {},
            'global_tracks': self.global_tracks,
            'cross_camera_matches': self.cross_camera_tracker.cross_camera_matches,
            'heatmap_data': self._generate_heatmap_data(),
            'timestamp': time.time()
        }

        # Get latest data from each camera
        for camera_id, camera in self.cameras.items():
            if not camera.is_running:
                continue

            latest_frame = camera.get_latest_frame()
            latest_tracks = camera.get_latest_tracks()

            if latest_frame and latest_tracks:
                visualization_data['cameras'][camera_id] = {
                    'frame': latest_frame['frame'],
                    'tracks': latest_tracks['tracks'],
                    'timestamp': latest_frame['timestamp'],
                    'stats': camera.get_stats()
                }

        return visualization_data

    def _generate_heatmap_data(self) -> Dict:
        """Generate heatmap data for visualization"""
        heatmap_data = {}

        for camera_id, camera in self.cameras.items():
            if not camera.is_running:
                continue

            # Create density map based on track positions
            grid_shape = (Config.RESIZE_HEIGHT // 20, Config.RESIZE_WIDTH // 20)
            density_map = np.zeros(grid_shape, dtype=np.float32)

            # Get track history from tracker
            track_history = camera.tracker.get_track_history()

            for track_id, points in track_history.items():
                for x, y in points:
                    # Map to density grid
                    grid_x = min(int(x // 20), grid_shape[1] - 1)
                    grid_y = min(int(y // 20), grid_shape[0] - 1)

                    if 0 <= grid_x < grid_shape[1] and 0 <= grid_y < grid_shape[0]:
                        density_map[grid_y, grid_x] += 1

            # Normalize
            if density_map.max() > 0:
                density_map = density_map / density_map.max()

            heatmap_data[camera_id] = density_map.tolist()

        return heatmap_data

    def export_system_data(self, output_path: str):
        """Export system data to JSON file"""
        try:
            export_data = {
                'timestamp': time.time(),
                'system_status': self.get_system_status(),
                'visualization_data': self.get_visualization_data(),
                'config': {
                    'camera_sources': self.camera_sources,
                    'fusion_method': self.fusion_method
                }
            }

            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj

            export_data = convert_numpy(export_data)

            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"System data exported to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export system data: {e}")

class VisionNetCore:
    """
    Core VisionNet system integrating all components.
    """

    def __init__(self, camera_sources: Optional[List] = None):
        """
        Initialize VisionNet core system.

        Args:
            camera_sources: List of camera sources
        """
        self.camera_sources = camera_sources or Config.CAMERA_SOURCES
        self.multi_camera_manager = MultiCameraManager(self.camera_sources)
        self.is_running = False
        self.start_time = None

    def start_system(self) -> bool:
        """Start the complete VisionNet system"""
        try:
            logger.info("Starting VisionNet system")
            self.start_time = time.time()

            success = self.multi_camera_manager.start_all_cameras()
            if success:
                self.is_running = True
                logger.info("VisionNet system started successfully")
            else:
                logger.error("Failed to start VisionNet system")

            return success

        except Exception as e:
            logger.error(f"Failed to start VisionNet system: {e}")
            return False

    def stop_system(self):
        """Stop the complete VisionNet system"""
        try:
            logger.info("Stopping VisionNet system")
            self.multi_camera_manager.stop_all_cameras()
            self.is_running = False
            logger.info("VisionNet system stopped")

        except Exception as e:
            logger.error(f"Error stopping VisionNet system: {e}")

    def get_system_data(self) -> Dict:
        """Get comprehensive system data for dashboard"""
        return {
            'system_status': self.multi_camera_manager.get_system_status(),
            'visualization_data': self.multi_camera_manager.get_visualization_data(),
            'is_running': self.is_running,
            'uptime': time.time() - self.start_time if self.start_time else 0
        }

# Test function
def test_multicamera():
    """Test the multi-camera system"""
    print("Testing Multi-Camera System...")

    # Test with dummy sources (will fail gracefully if no cameras)
    test_sources = [0]  # Try webcam

    try:
        # Initialize system
        visionnet = VisionNetCore(test_sources)

        print("Starting system...")
        success = visionnet.start_system()

        if success:
            print("✓ System started successfully")

            # Run for a short test period
            for i in range(10):
                system_data = visionnet.get_system_data()
                active_cameras = system_data['system_status']['system_stats']['active_cameras']
                total_tracks = system_data['system_status']['system_stats']['total_tracks']

                print(f"Test {i+1}: {active_cameras} active cameras, {total_tracks} total tracks")
                time.sleep(1)

            print("Stopping system...")
            visionnet.stop_system()
            print("✓ System stopped successfully")
        else:
            print("✗ Failed to start system (likely no cameras available)")

    except Exception as e:
        print(f"✗ Test failed: {e}")

if __name__ == "__main__":
    test_multicamera()
