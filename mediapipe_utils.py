import mediapipe as mp
import cv2
import numpy as np

class MediaPipeProcessor:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Create instances with appropriate configurations
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def process_image(self, image, processing_type):
        """Process image with specified MediaPipe solution"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = {}
        
        if processing_type == 'pose':
            pose_results = self.pose.process(rgb_image)
            if pose_results.pose_landmarks:
                results['pose_landmarks'] = self._extract_landmarks(pose_results.pose_landmarks)
        
        elif processing_type == 'hands':
            hand_results = self.hands.process(rgb_image)
            results['hand_landmarks'] = []
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    results['hand_landmarks'].append(self._extract_landmarks(hand_landmarks))
            
            results['handedness'] = []
            if hand_results.multi_handedness:
                for handedness in hand_results.multi_handedness:
                    results['handedness'].append({
                        'label': handedness.classification[0].label,
                        'score': handedness.classification[0].score
                    })
        
        elif processing_type == 'face':
            face_results = self.face_mesh.process(rgb_image)
            if face_results.multi_face_landmarks:
                results['face_landmarks'] = self._extract_landmarks(face_results.multi_face_landmarks[0])
        
        elif processing_type == 'holistic':
            holistic_results = self.holistic.process(rgb_image)
            
            if holistic_results.pose_landmarks:
                results['pose_landmarks'] = self._extract_landmarks(holistic_results.pose_landmarks)
            
            if holistic_results.left_hand_landmarks:
                results['left_hand_landmarks'] = self._extract_landmarks(holistic_results.left_hand_landmarks)
            
            if holistic_results.right_hand_landmarks:
                results['right_hand_landmarks'] = self._extract_landmarks(holistic_results.right_hand_landmarks)
            
            if holistic_results.face_landmarks:
                results['face_landmarks'] = self._extract_landmarks(holistic_results.face_landmarks)
        
        return results
    
    def process_frame(self, image, processing_type):
        """Optimized version for video frame processing"""
        return self.process_image(image, processing_type)
    
    def _extract_landmarks(self, landmarks):
        """Extract landmark coordinates in a serializable format"""
        landmark_list = []
        if landmarks:
            for idx, landmark in enumerate(landmarks.landmark):
                landmark_list.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': getattr(landmark, 'visibility', 1.0)
                })
        return landmark_list
    
    def close(self):
        """Close MediaPipe instances"""
        self.pose.close()
        self.hands.close()
        self.face_mesh.close()
        self.holistic.close()
