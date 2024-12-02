import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

#quaternion (x,y,z,w)
def qua_to_rotation_vec(quaternion):
    rotation_vec = Rotation.from_quat(quaternion)
    rotation_vec = cv2.Rodrigues(rotation_vec.as_matrix())[0]
    return rotation_vec

def rotation_vec_to_euler(rotation_vec,degree=True):
    euler = rotation_vec.reshape(3) 
    if degree:
        euler = np.rad2deg(euler)
    return euler

def create_pose_data(quaternion,translation):
    rotation_vec = qua_to_rotation_vec(quaternion)
    euler = rotation_vec_to_euler(rotation_vec)
    pose_data=np.hstack([translation,euler])
    
    return pose_data.tolist()

def draw_axis_from_qua(img, quaternion, t, K, scale=0.05, dist=None):
    rotation_vec = qua_to_rotation_vec(quaternion)
    dist =np.zeros(4, dtype=float) if dist is None else dist
    points = scale*np.float32([[1,0,0],[0,1,0],[0,0,1],[0,0,0]]).reshape(-1,3)
    axis_points, _ =cv2.projectPoints(points, rotation_vec, t, K, dist)
    axis_points = axis_points.astype(np.int32)
    img = cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[0].ravel()), (255, 0, 0), 3)
    img = cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[1].ravel()), (0, 255, 0), 3)
    img = cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[2].ravel()), (0, 0, 255), 3)
    return img
    
def draw_boxes(img, boxes):
    
    for i in range(len(boxes)):
        box = boxes[i]
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (np.array([1.,0.,0.]) * 255).astype(np.uint8).tolist()
            
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

    return img    
    
if __name__ =="__main__":
    K =np.array([[3626.9, 0.0, 773.2], [0.0, 3626.6, 5519], [0.0, 0.0, 1.0]])
    resolution = (1440, 1080)
    t =np.array([0.00036494730738922954, 0.00837632268667221, 0.13977889716625214])
    q =np.array([0.04150985740836541, -0.01681779531579537, 0.5644234711798364, 0.8242695508008724])
  
    img = cv2.imread("./data/test.jpg")

    img =draw_axis(img, q,t,K)
    
    cv2.imshow("image", img)
    cv2.waitKey(0)
