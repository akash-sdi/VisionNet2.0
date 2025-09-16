"""
VisionNet Database and Analytics Module
======================================
SQLite database for tracking data storage and analytics.
"""

import sqlite3
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any
import threading
import numpy as np

from config import Config

# Setup logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class VisionNetDatabase:
    """Database manager for VisionNet tracking data and analytics."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection."""
        self.db_path = db_path or (Config.DATA_DIR / "visionnet.db")
        self.db_lock = threading.Lock()

        # Ensure data directory exists
        Config.DATA_DIR.mkdir(exist_ok=True)

        self._init_database()

    def _init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create tracking data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tracking_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        camera_id INTEGER NOT NULL,
                        track_id INTEGER NOT NULL,
                        global_id INTEGER,
                        bbox_x1 INTEGER NOT NULL,
                        bbox_y1 INTEGER NOT NULL,
                        bbox_x2 INTEGER NOT NULL,
                        bbox_y2 INTEGER NOT NULL,
                        center_x INTEGER NOT NULL,
                        center_y INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        class_id INTEGER NOT NULL,
                        class_name TEXT NOT NULL,
                        frame_id INTEGER
                    )
                """)

                # Create events table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        event_type TEXT NOT NULL,
                        camera_id INTEGER,
                        track_id INTEGER,
                        global_id INTEGER,
                        description TEXT NOT NULL,
                        metadata TEXT,
                        severity TEXT DEFAULT 'INFO'
                    )
                """)

                # Create system stats table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        total_cameras INTEGER NOT NULL,
                        active_cameras INTEGER NOT NULL,
                        total_tracks INTEGER NOT NULL,
                        global_tracks INTEGER NOT NULL,
                        system_fps REAL NOT NULL,
                        metadata TEXT
                    )
                """)

                # Create indices
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracking_timestamp ON tracking_data(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracking_camera ON tracking_data(camera_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")

                conn.commit()
                logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def insert_tracking_data(self, camera_id: int, tracks: List[Dict], timestamp: Optional[float] = None):
        """Insert tracking data into database."""
        if not tracks:
            return

        timestamp = timestamp or time.time()

        try:
            with self.db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()

                    for track in tracks:
                        bbox = track.get('bbox', [0, 0, 0, 0])
                        center = track.get('center', (0, 0))

                        cursor.execute("""
                            INSERT INTO tracking_data 
                            (timestamp, camera_id, track_id, global_id, bbox_x1, bbox_y1, 
                             bbox_x2, bbox_y2, center_x, center_y, confidence, class_id, 
                             class_name, frame_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            timestamp, camera_id, track['track_id'], track.get('global_id'),
                            bbox[0], bbox[1], bbox[2], bbox[3],
                            center[0], center[1],
                            track.get('confidence', 1.0),
                            track.get('class_id', 0),
                            track.get('class_name', 'person'),
                            track.get('frame_count')
                        ))

                    conn.commit()

        except Exception as e:
            logger.error(f"Failed to insert tracking data: {e}")

    def log_event(self, event_type: str, description: str, camera_id: Optional[int] = None,
                  track_id: Optional[int] = None, global_id: Optional[int] = None,
                  metadata: Optional[Dict] = None, severity: str = 'INFO'):
        """Log an event to the database."""
        try:
            with self.db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()

                    cursor.execute("""
                        INSERT INTO events 
                        (timestamp, event_type, camera_id, track_id, global_id, 
                         description, metadata, severity)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        time.time(), event_type, camera_id, track_id, global_id,
                        description, json.dumps(metadata) if metadata else None, severity
                    ))

                    conn.commit()

        except Exception as e:
            logger.error(f"Failed to log event: {e}")

    def insert_system_stats(self, stats: Dict):
        """Insert system statistics into database."""
        try:
            with self.db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()

                    cursor.execute("""
                        INSERT INTO system_stats 
                        (timestamp, total_cameras, active_cameras, total_tracks, 
                         global_tracks, system_fps, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        time.time(),
                        stats.get('total_cameras', 0),
                        stats.get('active_cameras', 0),
                        stats.get('total_tracks', 0),
                        stats.get('global_tracks', 0),
                        stats.get('system_fps', 0.0),
                        json.dumps(stats)
                    ))

                    conn.commit()

        except Exception as e:
            logger.error(f"Failed to insert system stats: {e}")

    def get_tracking_history(self, hours: int = 24, camera_id: Optional[int] = None) -> List[Dict]:
        """Get tracking history from database."""
        try:
            cutoff_time = time.time() - (hours * 3600)

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                query = "SELECT * FROM tracking_data WHERE timestamp > ?"
                params = [cutoff_time]

                if camera_id is not None:
                    query += " AND camera_id = ?"
                    params.append(camera_id)

                query += " ORDER BY timestamp DESC LIMIT 10000"

                cursor.execute(query, params)
                rows = cursor.fetchall()

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get tracking history: {e}")
            return []

    def get_event_logs(self, hours: int = 24, event_type: Optional[str] = None) -> List[Dict]:
        """Get event logs from database"""
        try:
            cutoff_time = time.time() - (hours * 3600)

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                query = "SELECT * FROM events WHERE timestamp > ?"
                params = [cutoff_time]

                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type)

                query += " ORDER BY timestamp DESC LIMIT 1000"

                cursor.execute(query, params)
                rows = cursor.fetchall()

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get event logs: {e}")
            return []

    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                stats = {}

                # Count records in each table
                for table in ['tracking_data', 'events', 'system_stats']:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    count = cursor.fetchone()[0]
                    stats[f'{table}_count'] = count

                # Get database file size
                stats['db_size_mb'] = Path(self.db_path).stat().st_size / (1024 * 1024)

                return stats

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}

class EventLogger:
    """Event logging system for VisionNet."""

    def __init__(self, database: Optional[VisionNetDatabase] = None):
        """Initialize event logger."""
        self.database = database or VisionNetDatabase()

    def log_system_start(self):
        """Log system start event"""
        self.database.log_event('system_start', 'VisionNet system started')

    def log_system_stop(self):
        """Log system stop event"""
        self.database.log_event('system_stop', 'VisionNet system stopped')

    def log_track_created(self, camera_id: int, track_id: int, global_id: Optional[int] = None):
        """Log track creation event"""
        self.database.log_event(
            'track_created',
            f'New track created on camera {camera_id}',
            camera_id=camera_id,
            track_id=track_id,
            global_id=global_id
        )

class DataAnalytics:
    """Analytics and reporting for VisionNet data."""

    def __init__(self, database: Optional[VisionNetDatabase] = None):
        """Initialize analytics system."""
        self.database = database or VisionNetDatabase()

    def get_tracking_summary(self, hours: int = 24) -> Dict:
        """Get tracking summary statistics"""
        try:
            tracking_data = self.database.get_tracking_history(hours)

            if not tracking_data:
                return {'total_detections': 0, 'unique_tracks': 0}

            total_detections = len(tracking_data)
            unique_tracks = len(set((item['camera_id'], item['track_id']) for item in tracking_data))

            return {
                'total_detections': total_detections,
                'unique_tracks': unique_tracks,
                'period_hours': hours
            }

        except Exception as e:
            logger.error(f"Failed to generate tracking summary: {e}")
            return {'error': str(e)}

    def generate_report(self, hours: int = 24) -> Dict:
        """Generate comprehensive analytics report"""
        return {
            'generated_at': datetime.now().isoformat(),
            'period_hours': hours,
            'tracking_summary': self.get_tracking_summary(hours),
            'database_stats': self.database.get_database_stats(),
            'recent_events': self.database.get_event_logs(hours=1)[:10]
        }

# Test function
def test_database():
    """Test the database module"""
    print("Testing Database Module...")

    # Initialize database
    db = VisionNetDatabase()
    analytics = DataAnalytics(db)
    event_logger = EventLogger(db)

    print("✓ Database initialized")

    # Test event logging
    event_logger.log_system_start()
    print("✓ Event logging tested")

    # Test tracking data insertion
    test_tracks = [{
        'track_id': 1,
        'global_id': 1,
        'bbox': [100, 100, 200, 300],
        'center': (150, 200),
        'confidence': 0.85,
        'class_id': 0,
        'class_name': 'person'
    }]

    db.insert_tracking_data(camera_id=1, tracks=test_tracks)
    print("✓ Tracking data insertion tested")

    # Test analytics
    summary = analytics.get_tracking_summary(hours=1)
    report = analytics.generate_report(hours=1)
    print(f"✓ Analytics tested: {summary}")

    print("Database module test completed!")

if __name__ == "__main__":
    test_database()
