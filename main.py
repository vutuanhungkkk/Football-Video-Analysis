from tracking import Track
import cv2
import numpy as np
from assigner import Assign_Team,Assign_Player_Ball
from estimator import Estimate_Camera_movement,Estimate_Speed_Distance
from view_transformer import Transform_View

def read_video(video_path):    # luu tung frame cua video vao list
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(ouput_video_frames,output_video_path): # xuat video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()


def main():
    video_frames = read_video('input_videos/test_vid.mp4')
    tracker = Track('models/track.pt')
    tracks = tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path='stubs/track_stubs.pkl')
    tracker.add_position_to_tracks(tracks)
    camera_movement_estimator = Estimate_Camera_movement(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,read_from_stub=True,stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)
    view_transformer = Transform_View()
    view_transformer.add_transformed_position_to_tracks(tracks)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    speed_and_distance_estimator = Estimate_Speed_Distance()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    team_assigner = Assign_Team()
    team_assigner.assign_teamcolor(video_frames[0],tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],track['bbox'],player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


    player_assigner = Assign_Player_Ball()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)


    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)
    save_video(output_video_frames, 'output_videos/output_video.mp4')

if __name__ == '__main__':
    main()