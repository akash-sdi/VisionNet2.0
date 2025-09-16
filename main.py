#!/usr/bin/env python3
"""
VisionNet Main Application
==========================
Main entry point for the VisionNet multi-camera tracking system.

Authors: Sudipta Akash, Md Sohail Ansari, Rayudu Navya Sri
Supervisor: Dr. N. Leelavathy
Institution: Godavari Institute of Engineering & Technology
"""

import sys
import argparse
import logging
import time
import signal
import cv2
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import Config, get_config
from multicamera import VisionNetCore
from database import VisionNetDatabase, EventLogger
from dashboard import create_app, socketio

# Global variables for graceful shutdown
visionnet_system = None
logger = None

def setup_logging(config_class):
    """Setup comprehensive logging for the application"""
    log_dir = Config.LOGS_DIR
    log_dir.mkdir(exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(Config.LOG_FORMAT)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config_class.LOG_LEVEL))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler for all logs
    file_handler = logging.FileHandler(log_dir / 'visionnet.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Error file handler
    error_handler = logging.FileHandler(log_dir / 'errors.log')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    return root_logger

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down...")
    cleanup_and_exit()

def cleanup_and_exit():
    """Cleanup resources and exit"""
    global visionnet_system

    try:
        if visionnet_system:
            logger.info("Stopping VisionNet system...")
            visionnet_system.stop_system()

        logger.info("VisionNet shutdown complete")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    finally:
        sys.exit(0)

def validate_camera_sources(sources):
    """
    Validate camera sources.

    Args:
        sources (list): List of camera sources

    Returns:
        list: Valid camera sources
    """
    valid_sources = []

    for source in sources:
        try:
            if isinstance(source, int):
                # Test webcam
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        valid_sources.append(source)
                        logger.info(f"Validated webcam source: {source}")
                    else:
                        logger.warning(f"Webcam {source} opened but cannot read frames")
                else:
                    logger.warning(f"Cannot open webcam {source}")
                cap.release()

            elif isinstance(source, str):
                if source.startswith(('rtsp://', 'http://', 'https://')):
                    # Network stream - assume valid for now
                    valid_sources.append(source)
                    logger.info(f"Added network stream: {source}")
                elif Path(source).exists():
                    # Video file
                    cap = cv2.VideoCapture(source)
                    if cap.isOpened():
                        valid_sources.append(source)
                        logger.info(f"Validated video file: {source}")
                    else:
                        logger.warning(f"Cannot open video file: {source}")
                    cap.release()
                else:
                    logger.warning(f"Video file not found: {source}")

        except Exception as e:
            logger.error(f"Failed to validate camera source {source}: {e}")

    return valid_sources

def test_camera_access():
    """Test if at least one camera is accessible"""
    logger.info("Testing camera access...")

    # Test default webcam
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                logger.info("✓ Default webcam (0) is accessible")
                return True
            else:
                logger.warning("✗ Default webcam opened but cannot read frames")
        else:
            logger.warning("✗ Cannot open default webcam (0)")
    except Exception as e:
        logger.warning(f"✗ Error testing default webcam: {e}")

    return False

def run_headless_mode(camera_sources, duration=None):
    """
    Run VisionNet in headless mode (no web dashboard).

    Args:
        camera_sources (list): List of camera sources
        duration (float): Optional duration to run (seconds)
    """
    global visionnet_system

    try:
        logger.info("Starting VisionNet in headless mode")
        logger.info(f"Camera sources: {camera_sources}")

        # Initialize system
        visionnet_system = VisionNetCore(camera_sources)

        # Initialize database and logging
        database = VisionNetDatabase()
        event_logger = EventLogger(database)

        # Start system
        success = visionnet_system.start_system()
        if not success:
            logger.error("Failed to start VisionNet system")
            return False

        event_logger.log_system_start()

        start_time = time.time()
        logger.info("VisionNet system started successfully")

        # Main loop
        try:
            frame_count = 0
            while True:
                # Get system data
                system_data = visionnet_system.get_system_data()

                # Log to database every 10 iterations
                if frame_count % 10 == 0:
                    database.insert_system_stats(system_data['system_status']['system_stats'])

                # Print status to console
                stats = system_data['system_status']['system_stats']
                elapsed = time.time() - start_time

                print(f"\rCameras: {stats['active_cameras']}/{stats['total_cameras']} | "
                      f"Tracks: {stats['total_tracks']} | "
                      f"Global: {stats['global_tracks']} | "
                      f"FPS: {stats['system_fps']:.1f} | "
                      f"Time: {elapsed:.0f}s", end='', flush=True)

                # Check duration limit
                if duration and elapsed >= duration:
                    logger.info(f"\nReached duration limit of {duration} seconds")
                    break

                frame_count += 1
                time.sleep(1.0)

        except KeyboardInterrupt:
            logger.info("\nReceived keyboard interrupt")

        # Cleanup
        event_logger.log_system_stop()
        visionnet_system.stop_system()

        logger.info("Headless mode completed successfully")
        return True

    except Exception as e:
        logger.error(f"Headless mode failed: {e}")
        return False

def run_dashboard_mode(camera_sources, host, port, debug):
    """
    Run VisionNet with web dashboard.

    Args:
        camera_sources (list): List of camera sources
        host (str): Flask host
        port (int): Flask port
        debug (bool): Debug mode
    """
    global visionnet_system

    try:
        logger.info("Starting VisionNet with web dashboard")
        logger.info(f"Camera sources: {camera_sources}")
        logger.info(f"Dashboard will be available at http://{host}:{port}")

        # Initialize system (will be started via dashboard)
        visionnet_system = VisionNetCore(camera_sources)

        # Create Flask app
        app = create_app()

        # Make system available to dashboard
        import dashboard
        dashboard.visionnet_system = visionnet_system

        # Run Flask-SocketIO app
        socketio.run(
            app,
            host=host,
            port=port,
            debug=debug,
            use_reloader=False,
            log_output=False  # Reduce SocketIO logging
        )

    except Exception as e:
        logger.error(f"Dashboard mode failed: {e}")
        raise

def main():
    """Main application entry point"""
    global logger, visionnet_system

    print("="*60)
    print("VisionNet: Multi-Camera Deep Learning Target Tracking")
    print("Godavari Institute of Engineering & Technology")
    print("="*60)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='VisionNet Multi-Camera Tracking System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode dashboard --cameras 0
  python main.py --mode headless --cameras 0 1 --duration 300
  python main.py --mode dashboard --cameras video.mp4
  python main.py --mode dashboard --cameras rtsp://user:pass@camera/stream
        """
    )

    parser.add_argument('--mode', choices=['dashboard', 'headless'], 
                       default='dashboard',
                       help='Run mode: dashboard (with web UI) or headless (console only)')

    parser.add_argument('--cameras', nargs='+', 
                       help='Camera sources (webcam index, video file, or RTSP URL)')

    parser.add_argument('--duration', type=float, 
                       help='Duration to run in headless mode (seconds)')

    parser.add_argument('--config', choices=['development', 'production'], 
                       default='development',
                       help='Configuration profile to use')

    parser.add_argument('--host', default=Config.FLASK_HOST,
                       help=f'Flask host address (default: {Config.FLASK_HOST})')

    parser.add_argument('--port', type=int, default=Config.FLASK_PORT,
                       help=f'Flask port number (default: {Config.FLASK_PORT})')

    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')

    parser.add_argument('--test-camera', action='store_true',
                       help='Test camera access and exit')

    args = parser.parse_args()

    # Load configuration
    config_class = get_config(args.config)

    # Setup logging
    logger = setup_logging(config_class)
    logger.info(f"VisionNet starting with {args.config} configuration")
    logger.info(f"Arguments: {vars(args)}")

    # Test camera access if requested
    if args.test_camera:
        print("Testing camera access...")
        if test_camera_access():
            print("✓ Camera test passed")
            return 0
        else:
            print("✗ Camera test failed")
            return 1

    # Determine camera sources
    if args.cameras:
        # Convert string arguments to appropriate types
        camera_sources = []
        for cam in args.cameras:
            try:
                # Try to convert to int (webcam)
                camera_sources.append(int(cam))
            except ValueError:
                # String (file path or RTSP URL)
                camera_sources.append(cam)
    else:
        # Use default sources or test webcam
        camera_sources = Config.CAMERA_SOURCES
        if not camera_sources:
            logger.info("No camera sources specified, testing default webcam...")
            if test_camera_access():
                camera_sources = [0]  # Default to webcam
            else:
                logger.error("No camera sources available")
                print("\nError: No camera sources available.")
                print("Try specifying camera sources with --cameras option.")
                print("Examples:")
                print("  --cameras 0           (webcam)")
                print("  --cameras video.mp4   (video file)")
                print("  --cameras 0 1         (multiple webcams)")
                return 1

    logger.info(f"Camera sources: {camera_sources}")

    # Validate camera sources
    if args.mode == 'headless':
        # For headless mode, validate cameras upfront
        valid_sources = validate_camera_sources(camera_sources)
        if not valid_sources:
            logger.error("No valid camera sources found")
            print("\nError: No valid camera sources found.")
            print("Please check your camera connections and sources.")
            return 1
        camera_sources = valid_sources
        logger.info(f"Valid camera sources: {camera_sources}")

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Run in selected mode
        if args.mode == 'headless':
            success = run_headless_mode(camera_sources, args.duration)
            return 0 if success else 1
        else:
            # Dashboard mode
            debug_mode = args.debug or config_class.DEBUG
            run_dashboard_mode(camera_sources, args.host, args.port, debug_mode)
            return 0

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        cleanup_and_exit()
        return 0
    except Exception as e:
        logger.error(f"Application failed: {e}")
        cleanup_and_exit()
        return 1

if __name__ == '__main__':
    sys.exit(main())
