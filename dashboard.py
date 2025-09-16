"""
VisionNet Flask Dashboard Module
===============================
Real-time web dashboard for VisionNet system monitoring and control.
"""

from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
import time
import threading
import logging
from datetime import datetime
from pathlib import Path

from config import Config
from multicamera import VisionNetCore
from database import VisionNetDatabase, EventLogger, DataAnalytics

# Setup logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY
CORS(app)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
visionnet_system = None
database = VisionNetDatabase()
event_logger = EventLogger(database)
analytics = DataAnalytics(database)
dashboard_thread = None
is_dashboard_running = False

def encode_frame_to_base64(frame):
    """Encode frame to base64 for web transmission."""
    try:
        if frame is None or frame.size == 0:
            return None

        # Resize frame for web display
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))

        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

        # Convert to base64
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{frame_base64}"

    except Exception as e:
        logger.error(f"Frame encoding failed: {e}")
        return None

def dashboard_update_loop():
    """Background thread for dashboard updates"""
    global is_dashboard_running

    logger.info("Starting dashboard update loop")

    while is_dashboard_running:
        try:
            if visionnet_system and visionnet_system.is_running:
                # Get system data
                system_data = visionnet_system.get_system_data()

                # Prepare data for web transmission
                dashboard_data = prepare_dashboard_data(system_data)

                # Emit updates to all connected clients
                socketio.emit('system_update', dashboard_data)

            time.sleep(1.0)  # Update every second

        except Exception as e:
            logger.error(f"Dashboard update error: {e}")
            time.sleep(1.0)

    logger.info("Dashboard update loop stopped")

def prepare_dashboard_data(system_data):
    """Prepare system data for dashboard transmission."""
    dashboard_data = {
        'timestamp': time.time(),
        'system_stats': system_data['system_status']['system_stats'],
        'cameras': {},
        'global_tracks': {},
        'heatmaps': {}
    }

    # Process camera data
    viz_data = system_data.get('visualization_data', {})

    for camera_id, camera_data in viz_data.get('cameras', {}).items():
        frame = camera_data.get('frame')
        tracks = camera_data.get('tracks', [])

        # Encode frame if available
        encoded_frame = None
        if frame is not None:
            # Annotate frame with tracks
            annotated_frame = annotate_frame_with_tracks(frame, tracks)
            encoded_frame = encode_frame_to_base64(annotated_frame)

        dashboard_data['cameras'][camera_id] = {
            'frame': encoded_frame,
            'tracks': len(tracks),
            'stats': camera_data.get('stats', {}),
            'timestamp': camera_data.get('timestamp')
        }

    # Add global tracks count
    dashboard_data['global_tracks'] = viz_data.get('global_tracks', {})

    # Add heatmap data (simplified)
    dashboard_data['heatmaps'] = viz_data.get('heatmap_data', {})

    return dashboard_data

def annotate_frame_with_tracks(frame, tracks):
    """Annotate frame with tracking information."""
    annotated_frame = frame.copy()

    for track in tracks:
        bbox = track.get('bbox', [0, 0, 100, 100])
        track_id = track.get('track_id', 0)
        confidence = track.get('confidence', 1.0)
        class_name = track.get('class_name', 'object')

        x1, y1, x2, y2 = bbox

        # Use track color if available
        if 'color' in track:
            color = track['color']
        else:
            # Generate consistent color
            np.random.seed(track_id)
            color = tuple(np.random.randint(0, 255, 3).tolist())

        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        # Prepare label
        label = f"{class_name} ID:{track_id}"
        if confidence < 1.0:
            label += f" {confidence:.2f}"

        # Draw label
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated_frame

# Flask Routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_system_status():
    """Get current system status"""
    try:
        if visionnet_system:
            status = visionnet_system.get_system_data()
            return jsonify(status)
        else:
            return jsonify({
                'error': 'System not initialized',
                'is_running': False,
                'system_status': {'system_stats': {'active_cameras': 0, 'total_tracks': 0}}
            })
    except Exception as e:
        logger.error(f"Status request failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/start', methods=['POST'])
def start_system():
    """Start the VisionNet system"""
    global visionnet_system, is_dashboard_running, dashboard_thread

    try:
        # Get camera sources from request
        data = request.get_json() or {}
        camera_sources = data.get('camera_sources', Config.CAMERA_SOURCES or [0])

        if not camera_sources:
            return jsonify({'error': 'No camera sources provided'}), 400

        # Stop existing system if running
        if visionnet_system and visionnet_system.is_running:
            visionnet_system.stop_system()

        # Initialize and start system
        visionnet_system = VisionNetCore(camera_sources)
        success = visionnet_system.start_system()

        if success:
            # Start dashboard updates if not already running
            if not is_dashboard_running:
                is_dashboard_running = True
                dashboard_thread = threading.Thread(target=dashboard_update_loop, daemon=True)
                dashboard_thread.start()

            # Log event
            event_logger.log_system_start()

            return jsonify({
                'message': 'System started successfully',
                'cameras': len(camera_sources),
                'sources': camera_sources
            })
        else:
            return jsonify({'error': 'Failed to start system'}), 500

    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_system():
    """Stop the VisionNet system"""
    global visionnet_system, is_dashboard_running

    try:
        if visionnet_system:
            visionnet_system.stop_system()
            event_logger.log_system_stop()

        is_dashboard_running = False

        return jsonify({'message': 'System stopped successfully'})

    except Exception as e:
        logger.error(f"Failed to stop system: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics')
def get_analytics():
    """Get analytics data"""
    try:
        hours = request.args.get('hours', 24, type=int)
        report = analytics.generate_report(hours)
        return jsonify(report)
    except Exception as e:
        logger.error(f"Analytics request failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/csv')
def export_csv():
    """Export tracking data to CSV"""
    try:
        hours = request.args.get('hours', 24, type=int)

        # Get tracking data
        tracking_data = database.get_tracking_history(hours)

        if not tracking_data:
            return jsonify({'error': 'No data to export'}), 404

        # Convert to CSV format
        import io
        import csv

        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        if tracking_data:
            writer.writerow(tracking_data[0].keys())

            # Write data
            for row in tracking_data:
                writer.writerow(row.values())

        csv_content = output.getvalue()
        output.close()

        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename=visionnet_data_{hours}h.csv"}
        )

    except Exception as e:
        logger.error(f"CSV export failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cameras')
def get_cameras():
    """Get camera configuration"""
    try:
        if visionnet_system:
            cameras_info = {}
            for camera_id, camera in visionnet_system.multi_camera_manager.cameras.items():
                cameras_info[camera_id] = {
                    'camera_id': camera_id,
                    'source': str(camera.source),
                    'stats': camera.get_stats()
                }
            return jsonify(cameras_info)
        else:
            return jsonify({'error': 'System not initialized'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/events')
def get_events():
    """Get recent events"""
    try:
        hours = request.args.get('hours', 1, type=int)
        event_type = request.args.get('type', None)

        events = database.get_event_logs(hours, event_type)

        # Format events for display
        formatted_events = []
        for event in events:
            formatted_event = {
                'id': event['id'],
                'timestamp': datetime.fromtimestamp(event['timestamp']).isoformat(),
                'type': event['event_type'],
                'description': event['description'],
                'camera_id': event['camera_id'],
                'track_id': event['track_id'],
                'severity': event['severity']
            }

            if event['metadata']:
                try:
                    formatted_event['metadata'] = json.loads(event['metadata'])
                except:
                    formatted_event['metadata'] = {}

            formatted_events.append(formatted_event)

        return jsonify(formatted_events)

    except Exception as e:
        logger.error(f"Events request failed: {e}")
        return jsonify({'error': str(e)}), 500

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected to dashboard')
    emit('connected', {'message': 'Connected to VisionNet dashboard'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected from dashboard')

@socketio.on('request_update')
def handle_update_request():
    """Handle manual update request from client"""
    try:
        if visionnet_system and visionnet_system.is_running:
            system_data = visionnet_system.get_system_data()
            dashboard_data = prepare_dashboard_data(system_data)
            emit('system_update', dashboard_data)
        else:
            emit('system_update', {
                'timestamp': time.time(),
                'system_stats': {'active_cameras': 0, 'total_tracks': 0, 'system_fps': 0},
                'cameras': {},
                'global_tracks': {},
                'heatmaps': {}
            })
    except Exception as e:
        logger.error(f"Update request failed: {e}")
        emit('error', {'message': str(e)})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

# Health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'version': Config.PROJECT_VERSION
    })

def create_app():
    """Application factory"""
    global is_dashboard_running, dashboard_thread

    # Ensure templates directory exists
    Config.TEMPLATES_DIR.mkdir(exist_ok=True)

    # Initialize database
    try:
        database._init_database()
        logger.info("Database initialized for dashboard")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    logger.info("VisionNet Dashboard initialized")

    return app

def run_dashboard(host=None, port=None, debug=None):
    """Run the dashboard application"""
    host = host or Config.FLASK_HOST
    port = port or Config.FLASK_PORT
    debug = debug if debug is not None else Config.DEBUG

    logger.info(f"Starting VisionNet Dashboard on {host}:{port}")

    # Create app
    app = create_app()

    # Run Flask-SocketIO app
    socketio.run(
        app,
        host=host,
        port=port,
        debug=debug,
        use_reloader=False  # Disable reloader to avoid threading issues
    )

# Test function
def test_dashboard():
    """Test dashboard functionality"""
    print("Testing Dashboard Module...")

    try:
        # Test app creation
        test_app = create_app()
        print("✓ Flask app created successfully")

        # Test frame encoding
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        encoded = encode_frame_to_base64(dummy_frame)
        if encoded and encoded.startswith('data:image/jpeg;base64,'):
            print("✓ Frame encoding works")
        else:
            print("✗ Frame encoding failed")

        # Test data preparation
        dummy_system_data = {
            'system_status': {'system_stats': {'active_cameras': 1, 'total_tracks': 2}},
            'visualization_data': {'cameras': {}, 'global_tracks': {}}
        }

        dashboard_data = prepare_dashboard_data(dummy_system_data)
        if 'system_stats' in dashboard_data and 'cameras' in dashboard_data:
            print("✓ Data preparation works")
        else:
            print("✗ Data preparation failed")

        print("Dashboard module test completed!")

    except Exception as e:
        print(f"✗ Dashboard test failed: {e}")

if __name__ == "__main__":
    test_dashboard()
