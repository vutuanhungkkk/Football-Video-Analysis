from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2

def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)
def get_bbox_width(bbox):
    return bbox[2]-bbox[0]
def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)

class Track:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self,tracks): #  them vao tracks
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_frames(self, frames):
        batch_size=20
        detections = []
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks={"players":[],"referees":[],"ball":[]}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision) # cap nhat tracking id voi detection
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks


    def draw_rectangle(self, frame, bbox, color, track_id=None):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=3)
        if track_id is not None:
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 255), 2)
        return frame

    def draw_star(self, frame, bbox, color):
        center_x = (bbox[0] + bbox[2]) // 2
        bottom_y = bbox[3] + 10
        size = int(get_bbox_width(bbox) / 3)
        points = np.array([
            [center_x, bottom_y],
            [center_x + size, bottom_y + size],
            [center_x - size, bottom_y + size // 2],
            [center_x + size, bottom_y + size // 2],
            [center_x - size, bottom_y + size]
        ], dtype=np.int32)
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
        cv2.fillPoly(frame, [points], color)
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        points = np.array([[1350, 850], [1450, 820], [1550, 850], [1600, 900], [1550, 950], [1450, 970]], dtype=np.int32)
        cv2.fillPoly(overlay, [points], (255, 255, 255))
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        team_1_ratio = np.mean(team_ball_control[:frame_num + 1] == 1) * 100
        team_2_ratio = np.mean(team_ball_control[:frame_num + 1] == 2) * 100
        cv2.putText(frame, f"Team 1 possesion: {team_1_ratio:.1f}%", (1370, 870), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, f"Team 2 possesion: {team_2_ratio:.1f}%", (1370, 920), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0), 2)
        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_rectangle(frame, player["bbox"], color, track_id)
                if player.get('has_ball', False):
                    frame = self.draw_star(frame, player["bbox"], (0, 0, 255))
            for _, referee in referee_dict.items():
                frame = self.draw_rectangle(frame, referee["bbox"], (0, 255, 255))
            for _, ball in ball_dict.items():
                frame = self.draw_star(frame, ball["bbox"], (0, 255, 0))
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            output_video_frames.append(frame)
        return output_video_frames

