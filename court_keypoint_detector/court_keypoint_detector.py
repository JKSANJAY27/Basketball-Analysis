from ultralytics import YOLO
import sys
sys.path.append("../")
from utils import read_stub, save_stub

class CourtKeypointDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def get_court_keypoints(self, frame, read_from_stub = False, stub_path=None):
        court_keypoints = read_stub(read_from_stub,stub_path)
        if court_keypoints is not None:
            if len(court_keypoints) == len(frame):
                return court_keypoints
        batch_size = 20
        court_keypoints = []
        for i in range(0, len(frame), batch_size):
            detections_batch = self.model.predict(frame[i:i+batch_size], conf=0.5)
            for detection in detections_batch:
                court_keypoints.append(detection.keypoints)
        save_stub(stub_path, court_keypoints)
        return court_keypoints        
        