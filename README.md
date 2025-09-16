
# VisionNet: Multi-Camera Deep Learning Approach for Target Tracking

![VisionNet Logo](https://img.shields.io/badge/VisionNet-v1.0.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

VisionNet is a state-of-the-art deep learning-based multi-camera tracking system designed for real-time person and object tracking across multiple video feeds. The system combines YOLOv8 for object detection, DeepSORT for single-camera tracking, and OSNet-based ReID models for cross-camera identity matching.

### Key Features

**Multi-Camera Tracking**: Simultaneous tracking across multiple camera feeds  
**Deep Learning Integration**: YOLOv8, DeepSORT, and OSNet models  
**Cross-Camera Re-ID**: Identity matching between different camera views  
**Real-time Dashboard**: Flask-based web interface with live visualization  
**Data Persistence**: SQLite database for tracking history and analytics  
**GPU Acceleration**: CUDA support for real-time performance  
**Modular Design**: Easy customization and extension  

### System Architecture

```
Multi-Camera Video Input → Object Detection (YOLOv8) → Single-Camera Tracking (DeepSORT)
                                     ↓                              ↓
                        Multi-Camera Data Fusion ← Cross-Camera Re-ID (OSNet)
                                     ↓
                        Flask Dashboard Visualization ← Database & Logs
                                     ↓
                        Deployment (Server/Edge Device)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for acceleration)
- Webcam or IP cameras for live tracking
- 4GB+ RAM recommended

### Quick Setup

```bash
# Clone the repository (if using git)
git clone <repository_url>
cd VisionNet

# Run automated setup
chmod +x setup.sh
./setup.sh
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv visionnet_env
source visionnet_env/bin/activate  # On Windows: visionnet_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8 model
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Usage

### 1. Dashboard Mode (Recommended)

Launch the web dashboard for interactive control:

```bash
python3 main.py --mode dashboard --cameras 0
```

Then open your browser to `http://localhost:5000`

### 2. Headless Mode

Run without GUI for server deployments:

```bash
python3 main.py --mode headless --cameras 0 --duration 300
```

### 3. Custom Camera Sources

```bash
# Multiple webcams
python3 main.py --cameras 0 1 2

# Video files
python3 main.py --cameras video1.mp4 video2.mp4

# RTSP streams
python3 main.py --cameras rtsp://user:pass@camera1/stream rtsp://user:pass@camera2/stream

# Mixed sources
python3 main.py --cameras 0 video.mp4 rtsp://camera/stream
```

### 4. Example Scripts

```bash
# Basic single camera tracking
python3 examples/basic_tracking.py

# Multi-camera system demo
python3 examples/multicamera_tracking.py

# Quick dashboard launcher
python3 examples/dashboard_launch.py
```

## Configuration

Edit `config.py` to customize system behavior:

```python
# Model Configuration
YOLO_MODEL = "yolov8n.pt"  # or yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
YOLO_CONFIDENCE = 0.5
TRACK_CLASSES = [0]  # 0 = person, add more class IDs as needed

# Performance Settings
USE_GPU = True
BATCH_SIZE = 16
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Cross-Camera Tracking
REID_THRESHOLD = 0.7
TEMPORAL_THRESHOLD = 30  # frames
```

## System Components

### 1. Object Detection (`detection.py`)
- **YOLOv8 Integration**: Real-time object detection
- **Multiple Model Support**: nano, small, medium, large, extra-large variants
- **Configurable Classes**: Track specific object types
- **GPU Acceleration**: CUDA support for real-time performance

### 2. Single-Camera Tracking (`tracking.py`)
- **DeepSORT Implementation**: Robust multi-object tracking
- **Identity Persistence**: Maintains track IDs through occlusions
- **Trajectory Visualization**: Track history and trails
- **Performance Monitoring**: FPS and accuracy metrics

### 3. Cross-Camera Re-ID (`reid.py`)
- **OSNet Integration**: State-of-the-art person re-identification
- **Feature Extraction**: Deep learning appearance features
- **Similarity Matching**: Cross-camera identity association
- **Feature Database**: Persistent storage for learned features

### 4. Multi-Camera Management (`multicamera.py`)
- **Concurrent Processing**: Parallel camera stream handling
- **Data Fusion**: Intelligent combining of multi-camera data
- **System Coordination**: Centralized management of all components
- **Performance Optimization**: Efficient resource utilization

### 5. Database & Analytics (`database.py`)
- **SQLite Storage**: Persistent tracking data and logs
- **Event Logging**: Comprehensive system event tracking
- **Analytics Engine**: Statistical analysis and reporting
- **Data Export**: CSV export for external analysis

### 6. Web Dashboard (`dashboard.py`)
- **Real-time Interface**: Flask + Socket.IO web dashboard
- **Live Visualization**: Multi-camera feed display
- **System Control**: Start/stop and configuration management
- **Analytics Display**: Performance charts and statistics

## API Reference

### REST API Endpoints

- `GET /api/status` - Get system status
- `POST /api/start` - Start tracking system
- `POST /api/stop` - Stop tracking system
- `GET /api/analytics` - Get analytics data
- `GET /api/export/csv` - Export tracking data
- `GET /api/cameras` - Get camera configuration
- `GET /api/events` - Get recent events

### Socket.IO Events

- `system_update` - Real-time system data updates
- `camera_feed` - Live camera frame updates
- `track_update` - Individual track updates
- `error` - Error notifications

## Performance Optimization

### Model Selection

Choose YOLOv8 model based on your hardware:

| Model | Size | mAP | Speed (CPU) | Speed (GPU) | Use Case |
|-------|------|-----|-------------|-------------|----------|
| YOLOv8n | 6MB | 37.3 | 80ms | 0.99ms | Resource constrained |
| YOLOv8s | 22MB | 44.9 | 128ms | 1.20ms | Balanced |
| YOLOv8m | 52MB | 50.2 | 235ms | 1.83ms | Higher accuracy |
| YOLOv8l | 87MB | 52.9 | 375ms | 2.39ms | High accuracy |
| YOLOv8x | 136MB | 53.9 | 479ms | 3.53ms | Maximum accuracy |

### Hardware Recommendations

**Minimum Requirements:**
- CPU: Intel i5-8400 / AMD Ryzen 5 2600
- RAM: 8GB
- GPU: GTX 1060 / RTX 3050 (optional)
- Storage: 10GB free space

**Recommended Setup:**
- CPU: Intel i7-10700K / AMD Ryzen 7 3700X
- RAM: 16GB
- GPU: RTX 3070 / RTX 4060 or better
- Storage: SSD with 50GB+ free space

**Production Setup:**
- CPU: Intel Xeon / AMD EPYC series
- RAM: 32GB+
- GPU: RTX 4080/4090 or A100
- Storage: NVMe SSD with 100GB+ space

## Development

### Project Structure

```
VisionNet/
├── main.py                 # Main application entry point
├── config.py              # Configuration settings
├── detection.py           # YOLOv8 object detection
├── tracking.py            # DeepSORT tracking
├── reid.py                # OSNet re-identification
├── multicamera.py         # Multi-camera management
├── database.py            # Database and analytics
├── dashboard.py           # Flask dashboard
├── requirements.txt       # Python dependencies
├── setup.sh              # Installation script
├── templates/
│   └── dashboard.html     # Dashboard web interface
├── examples/
│   ├── basic_tracking.py  # Single camera example
│   ├── multicamera_tracking.py  # Multi-camera example
│   └── dashboard_launch.py      # Dashboard launcher
├── data/                  # Data directory
├── models/               # Model weights
├── logs/                 # Log files
└── uploads/              # File uploads
```

### Adding New Features

1. **Custom Object Classes**: Modify `TRACK_CLASSES` in `config.py`
2. **New Tracking Algorithms**: Extend `tracking.py` with new tracker implementations
3. **Additional ReID Models**: Add new models in `reid.py`
4. **Dashboard Features**: Extend `dashboard.py` and `templates/dashboard.html`

### Testing

```bash
# Run basic functionality tests
python3 -m pytest tests/ -v

# Test individual components
python3 detection.py     # Test detection module
python3 tracking.py      # Test tracking module
python3 reid.py         # Test re-ID module
```

## Datasets and Evaluation

### Supported Datasets

- **DukeMTMC**: Multi-camera tracking dataset with 8 cameras
- **MOTChallenge**: Standard benchmark for tracking evaluation
- **Market1501**: Person re-identification dataset
- **CUHK03**: Large-scale person re-ID dataset

### Evaluation Metrics

- **MOTA**: Multiple Object Tracking Accuracy
- **MOTP**: Multiple Object Tracking Precision  
- **IDF1**: ID F1 Score
- **MT/ML**: Mostly Tracked/Mostly Lost targets
- **FP/FN**: False Positives/False Negatives
- **ID Sw**: Identity Switches

## Applications

###  Security & Surveillance
- Perimeter monitoring
- Access control systems
- Crowd monitoring
- Incident detection

###  Retail Analytics  
- Customer behavior analysis
- Queue management
- Heat mapping
- Loss prevention

###  Sports Analytics
- Player tracking
- Performance analysis
- Tactical insights
- Broadcast enhancement

###  Industrial Safety
- Worker safety monitoring
- Equipment tracking
- Compliance monitoring
- Accident prevention

## Troubleshooting

### Common Issues

**Issue**: Camera not detected
```bash
# Test camera access
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera failed'); cap.release()"
```

**Issue**: Low FPS performance
- Reduce video resolution in `config.py`
- Use smaller YOLO model (yolov8n.pt)
- Enable GPU acceleration
- Reduce number of concurrent cameras

**Issue**: Cross-camera matching not working
- Ensure cameras have overlapping fields of view
- Adjust `REID_THRESHOLD` in `config.py`
- Check camera synchronization

**Issue**: Dashboard not loading
- Check Flask port availability: `netstat -an | grep 5000`
- Verify firewall settings
- Check browser console for JavaScript errors

### Debugging

Enable debug mode in `config.py`:
```python
DEBUG = True
LOG_LEVEL = "DEBUG"
```

Check log files in the `logs/` directory:
- `visionnet.log`: General system logs
- `errors.log`: Error-specific logs

## Performance Benchmarks

### Test Environment
- CPU: Intel i7-10700K
- GPU: RTX 3070
- RAM: 32GB
- Camera: 1080p @ 30fps

### Results
- **Single Camera**: 28-30 FPS
- **4 Cameras**: 24-26 FPS  
- **8 Cameras**: 18-22 FPS
- **Cross-Camera Accuracy**: 85-92%

## Contributing

We welcome contributions! Please read our contributing guidelines:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run code formatting
black *.py

# Run linting
flake8 *.py

# Run tests
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Ultralytics Team**: For the excellent YOLOv8 implementation
- **DeepSORT Authors**: For the robust tracking algorithm
- **OSNet Authors**: For the person re-identification model
- **Dataset Creators**: DukeMTMC and MOTChallenge teams
- **Supervisors**: Dr. N. Leelavathy (Vice Principal)

## Citation

If you use VisionNet in your research, please cite:

```bibtex
@misc{visionnet2025,
  title={VisionNet: A Multi-Camera Deep Learning Approach for Target Tracking},
  author={Sudipta Akash and Md Sohail Ansari and Rayudu Navya Sri},
  year={2025},
  institution={Godavari Institute of Engineering & Technology},
  supervisor={Dr. N. Leelavathy}
}
```

## Contact

**Students:**
- Sudipta Akash (22551A05K3)
- Md Sohail Ansari (22551A0543)  
- Rayudu Navya Sri (22551A05C0)

**Supervisor:**
- Dr. N. Leelavathy (Vice Principal)
- Dr. B. Sujatha (HOD, CSE)

**Institution:**  
Godavari Institute of Engineering & Technology (A), Rajahmundry

---

## Quick Start Guide

### 1. Installation
```bash
git clone <repository>
cd VisionNet
chmod +x setup.sh
./setup.sh
```

### 2. Activate Environment
```bash
source visionnet_env/bin/activate
```

### 3. Run Dashboard
```bash
python3 main.py --mode dashboard --cameras 0
```

### 4. Open Browser
Navigate to `http://localhost:5000`

### 5. Start System
Click "Start System" in the dashboard interface

---

*Built with ❤️ by the VisionNet team at GIET*
