"""
VisionNet Configuration Module
============================
Central configuration management for the VisionNet system.
"""

import os
from pathlib import Path

class Config:
    """Base configuration class"""

    # Project Information
    PROJECT_NAME = "VisionNet"
    PROJECT_VERSION = "1.0.0"
    DEBUG = True

    # Paths
    BASE_DIR = Path(__file__).parent.absolute()
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    UPLOADS_DIR = BASE_DIR / "uploads"
    STATIC_DIR = BASE_DIR / "static"
    TEMPLATES_DIR = BASE_DIR / "templates"

    # Create directories
    for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, UPLOADS_DIR, STATIC_DIR, TEMPLATES_DIR]:
        directory.mkdir(exist_ok=True)

    # YOLOv8 Configuration
    YOLO_MODEL = "yolov8n.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    YOLO_CONFIDENCE = 0.5
    YOLO_IOU = 0.45
    YOLO_DEVICE = "cuda" if os.environ.get("CUDA_AVAILABLE", "True") == "True" else "cpu"

    # DeepSORT Configuration
    DEEPSORT_MODEL_PATH = MODELS_DIR / "deep_sort_weights"
    DEEPSORT_MAX_DIST = 0.2
    DEEPSORT_MIN_CONFIDENCE = 0.3
    DEEPSORT_NMS_MAX_OVERLAP = 1.0
    DEEPSORT_MAX_IOU_DISTANCE = 0.7
    DEEPSORT_MAX_AGE = 70
    DEEPSORT_N_INIT = 3

    # OSNet ReID Configuration
    OSNET_MODEL = "osnet_x1_0"
    OSNET_THRESHOLD = 0.6
    REID_FEATURE_DIM = 512

    # Multi-Camera Configuration
    MAX_CAMERAS = 8
    CAMERA_SOURCES = [
        # Add your camera sources here
        # Examples:
        # 0,  # Webcam
        # "rtsp://username:password@camera_ip:port/stream",
        # "/path/to/video/file.mp4",
    ]

    # Video Processing
    VIDEO_FPS = 30
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    RESIZE_WIDTH = 640
    RESIZE_HEIGHT = 480
    SAVE_VIDEO = True
    VIDEO_CODEC = "mp4v"

    # Database Configuration
    DATABASE_URL = f"sqlite:///{DATA_DIR}/visionnet.db"
    DATABASE_TRACK_MODIFICATIONS = False

    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'visionnet-dev-key-change-in-production'
    FLASK_HOST = "0.0.0.0"
    FLASK_PORT = 5000

    # Tracking Configuration
    TRACK_CLASSES = [0]  # 0 = person class in COCO dataset
    MIN_TRACK_LENGTH = 3
    MAX_TRACK_AGE = 30
    TRACK_BUFFER_SIZE = 100

    # Cross-Camera Tracking
    REID_THRESHOLD = 0.7
    SPATIAL_THRESHOLD = 100  # pixels
    TEMPORAL_THRESHOLD = 30  # frames
    GLOBAL_ID_TIMEOUT = 300  # seconds

    # Data Fusion
    FUSION_METHOD = "weighted_average"
    CONFIDENCE_WEIGHTS = True
    FUSION_WINDOW = 5  # frames

    # Performance Settings
    USE_GPU = True
    BATCH_SIZE = 1
    NUM_WORKERS = 4
    DEVICE = "cuda" if USE_GPU else "cpu"

    # Dashboard Configuration
    REFRESH_RATE = 1000  # milliseconds
    MAX_DISPLAY_TRACKS = 100
    HEATMAP_UPDATE_INTERVAL = 5000  # milliseconds
    ENABLE_SOCKETIO = True

    # Logging Configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE_MAX_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    FLASK_PORT = 5000

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = "WARNING"
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'change-this-secret-key'
    FLASK_HOST = "127.0.0.1"  # More secure for production

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    DATABASE_URL = f"sqlite:///{Config.DATA_DIR}/test_visionnet.db"

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name='default'):
    """Get configuration class by name"""
    return config.get(config_name, config['default'])
