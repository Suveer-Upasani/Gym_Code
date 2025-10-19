from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
import mediapipe as mp
import base64
import json
from mediapipe_utils import MediaPipeProcessor

app = Flask(__name__)
processor = MediaPipeProcessor()

@app.route('/')
def home():
    return jsonify({
        "message": "MediaPipe Flask API",
        "endpoints": {
            "health": "/health",
            "process_image": "/process-image (POST)",
            "process_video_frame": "/process-video-frame (POST)"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "MediaPipe API"})

@app.route('/process-image', methods=['POST'])
def process_image():
    """
    Process a single image for pose, hand, or face detection
    Expects JSON with base64 image and processing type
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Get processing type
        processing_type = data.get('type', 'pose')  # pose, hands, face, holistic
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Could not decode image"}), 400
        
        # Process image
        results = processor.process_image(image, processing_type)
        
        return jsonify({
            "success": True,
            "type": processing_type,
            "results": results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process-video-frame', methods=['POST'])
def process_video_frame():
    """
    Process a single video frame for real-time detection
    Optimized for faster processing
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        processing_type = data.get('type', 'pose')
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Could not decode image"}), 400
        
        # Process frame (optimized for speed)
        results = processor.process_frame(image, processing_type)
        
        return jsonify({
            "success": True,
            "type": processing_type,
            "results": results,
            "timestamp": data.get('timestamp', 0)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch-process', methods=['POST'])
def batch_process():
    """
    Process multiple frames at once
    """
    try:
        data = request.get_json()
        
        if not data or 'frames' not in data:
            return jsonify({"error": "No frames data provided"}), 400
        
        frames = data['frames']
        processing_type = data.get('type', 'pose')
        results = []
        
        for frame_data in frames:
            image_data = base64.b64decode(frame_data['image'])
            np_arr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if image is not None:
                frame_results = processor.process_frame(image, processing_type)
                results.append({
                    "timestamp": frame_data.get('timestamp', 0),
                    "results": frame_results
                })
        
        return jsonify({
            "success": True,
            "processed_frames": len(results),
            "results": results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
