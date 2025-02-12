import numpy as np 
import cv2

class Transform_View(): # chuyen doi toa do tu pixel sang toa do the gioi
    def __init__(self):
        court_width = 68 # chieu rong san bong
        court_length = 23.32 # chieu dai san bong
        self.pixel_vertices = np.array([[110, 1035], [265, 275], [910, 260],[1640, 915]]) # toa do 4 goc cua san bong
        self.target_vertices = np.array([[0,court_width],[0, 0],[court_length, 0],[court_length, court_width]]) # toa do thuc 4 goc cua san bong
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)
        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices) # persepctive_trasnformer la  ma tran chuyen doi tu toa do pixel sang toa do the gioi

    def transform_point(self,point): # chuyen doi toa do pixel cu the sang toa do the gioi
        p = (int(point[0]),int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices,p,False) >= 0 
        if not is_inside:
            return None
        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        tranform_point = cv2.perspectiveTransform(reshaped_point,self.persepctive_trasnformer)
        return tranform_point.reshape(-1,2)

    def add_transformed_position_to_tracks(self,tracks): # them vao tracks toa do the gioi cuar cac object
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_trasnformed = self.transform_point(position)
                    if position_trasnformed is not None:
                        position_trasnformed = position_trasnformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed