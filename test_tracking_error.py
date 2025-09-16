#!/usr/bin/env python3
"""
Simple test script to isolate the tracking error
"""

import sys
import traceback
from tracking import DeepSORTTracker

def test_error():
    """Test the tracking with detailed error reporting"""
    print("Testing DeepSORT Tracker with detailed error reporting...")

    try:
        # Initialize tracker
        print("1. Initializing tracker...")
        tracker = DeepSORTTracker()
        print("✓ Tracker initialized successfully")

        # Create test detections
        print("2. Creating test detections...")
        test_detections = [
            {
                'bbox': [100, 100, 200, 300],
                'confidence': 0.8,
                'class_id': 0,
                'class_name': 'person',
                'center': (150, 200)
            }
        ]
        print("✓ Test detections created")

        # Test frame
        print("3. Creating test frame...")
        import numpy as np
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("✓ Test frame created")

        # Test update
        print("4. Testing tracker update...")
        tracks = tracker.update(test_detections, test_frame)
        print(f"✓ Tracker update successful: {len(tracks)} tracks")

        print("5. Test completed successfully!")

    except Exception as e:
        print(f"✗ Error occurred: {e}")
        print("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    test_error()
