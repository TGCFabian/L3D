# Based off of https://github.com/daovietanh190499/structure-from-motion

import os
import cv2
import numpy as np

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


class Camera:
    def __init__(self, kp):
        self.kp = kp
        self.match2d3d = np.ones((len(kp),), dtype='int32') * -1
        self.Rt = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        self.reconstruct = False

    def setRt(self, R, t):
        self.Rt = np.hstack((R, t))
        self.reconstruct = True

    def getRt(self):
        return self.Rt[:3, :3], self.Rt[:3, 3]

    def getRelativeRt(self, cam2):
        return cam2.Rt[:3, :3].T.dot(self.Rt[:3, :3]), cam2.Rt[:3, :3].T.dot(self.Rt[:3, 3] - cam2.Rt[:3, 3])

    def getP(self, K):
        return np.matmul(K, self.Rt)

    def getPos(self):
        pts = np.array([[0, 0, 0]]).T
        pts = self.Rt[:3, :3].T.dot(pts) - self.Rt[:3, 3][:, np.newaxis]
        return pts[:, 0]


def get_camera_intrinsic_params(images_dir):
    K = []
    image_height, image_width, c = cv2.imread(images_dir + os.listdir(images_dir)[1]).shape
    focal_length = (27.0 / 35.0) * image_width
    K.append([focal_length, 0, image_width / 2])
    K.append([0, focal_length, image_height / 2])
    K.append([0, 0, 1])
    return np.array(K, dtype=float)


def triangulate(cam1, cam2, idx0, idx1, K):
    points_3d = cv2.triangulatePoints(cam1.getP(K), cam2.getP(K), cam1.kp[idx0].T, cam2.kp[idx1].T)
    points_3d = points_3d / points_3d[3]
    points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
    points_3d = points_3d[:, 0, :]
    point2d_ind = idx1[np.where(cam1.match2d3d[idx0] == -1)]
    for w, i in enumerate(idx0):
        if cam1.match2d3d[i] == -1:
            point_cloud.append(points_3d[w])
            cam1.match2d3d[i] = len(point_cloud) - 1
        cam2.match2d3d[idx1[w]] = cam1.match2d3d[i]
    point3d_ind = cam2.match2d3d[point2d_ind]
    x = np.hstack((cv2.Rodrigues(cam2.getRt()[0])[0].ravel(), cam2.getRt()[1].ravel(),
                   np.array(point_cloud)[point3d_ind].ravel()))
    A = ba_sparse(point3d_ind, x)
    res = least_squares(calculate_reprojection_error, x, jac_sparsity=A, x_scale='jac', ftol=1e-8,
                        args=(K, cam2.kp[point2d_ind]))
    R, t, point_3D = cv2.Rodrigues(res.x[:3])[0], res.x[3:6], res.x[6:].reshape((len(point3d_ind), 3))
    for i, j in enumerate(point3d_ind):
        point_cloud[j] = point_3D[i]
    cam2.setRt(R, t.reshape((3, 1)))


def to_ply(img_dir, point_cloud, subfix="_sparse.ply"):
    verts = point_cloud.reshape(-1, 3) * 200
    mean = np.mean(verts[:, :3], axis=0)
    temp = verts[:, :3] - mean
    dist = np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
    indx = np.where(dist < np.mean(dist) + 300)  # culls infinite points zooming off, need to remove this at some point
    verts = verts[indx]
    ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		end_header
		'''

    if not os.path.exists(img_dir + '/Point_Cloud/'):
        os.makedirs(img_dir + '/Point_Cloud/')
    with open(img_dir + '/Point_Cloud/' + img_dir.split('/')[-2] + subfix, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f')


def ba_sparse(point3d_ind, x):
    A = lil_matrix((len(point3d_ind) * 2, len(x)), dtype=int)
    A[np.arange(len(point3d_ind) * 2), :6] = 1
    for i in range(3):
        A[np.arange(len(point3d_ind)) * 2, 6 + np.arange(len(point3d_ind)) * 3 + i] = 1
        A[np.arange(len(point3d_ind)) * 2 + 1, 6 + np.arange(len(point3d_ind)) * 3 + i] = 1
    return A


def calculate_reprojection_error(x, K, point_2D):
    R, t, point_3D = x[:3], x[3:6], x[6:].reshape((len(point_2D), 3))
    reprojected_point, _ = cv2.projectPoints(point_3D, R, t, K, distCoeffs=None)
    reprojected_point = reprojected_point[:, 0, :]
    return (point_2D - reprojected_point).ravel()

def render_stuff(stuff):
    img = np.zeros((int(image_height / 8), int(image_width / 8)))

    for k in stuff:
        k = [int(i / 8) for i in k]
        cv2.drawMarker(img, k, (255, 255, 255), 1, 1)

    cv2.namedWindow("temp")  # Create a named window
    cv2.moveWindow("temp", 40, 30)  # Move it to (40,30)
    cv2.imshow("temp", img)
    cv2.waitKey(0)

if __name__ == "__main__":

    cameras = []
    point_cloud = []

    K = []
    image_height, image_width = 4032, 3024
    focal_length = (27.0 / 35.0) * image_width
    K.append([focal_length, 0, image_width / 2])
    K.append([0, focal_length, image_height / 2])
    K.append([0, 0, 1])
    K = np.array(K, dtype=float)

    for i in range(7):
        print(f"working on {i}")

        key_points = np.loadtxt(f'sfm_data/key_points_{i}.csv', delimiter=',')

        # aaagh key_points here lines up with pts1_

        camera = Camera(key_points)
        cameras.append(camera)

        if i == 0:
            continue

        pts0_ = np.loadtxt(f'sfm_data/pts0_{i}.csv', delimiter=',')
        pts1_ = np.loadtxt(f'sfm_data/pts1_{i}.csv', delimiter=',')
        idx0 = np.loadtxt(f'sfm_data/idx0{i}.csv', delimiter=',', dtype=int)
        idx1 = np.loadtxt(f'sfm_data/idx1{i}.csv', delimiter=',', dtype=int)

        E, mask = cv2.findEssentialMat(pts0_, pts1_, K, method=cv2.RANSAC, prob=0.999, threshold=1)
        idx0 = idx0[mask.ravel() == 1]
        idx1 = idx1[mask.ravel() == 1]
        _, R, t, _ = cv2.recoverPose(E, pts0_[mask.ravel() == 1], pts1_[mask.ravel() == 1], K)

        if i > 1:
            match = np.int32(np.where(cameras[i - 1].match2d3d[idx0] != -1)[0])

            if len(match) < 8:
                continue

            _, rvecs, t, _ = cv2.solvePnPRansac(
                np.float32(point_cloud)[cameras[i - 1].match2d3d[idx0[match]]],
                cameras[i].kp[idx1[match]],
                K,
                np.zeros((5, 1), dtype=np.float32),
                cv2.SOLVEPNP_ITERATIVE)

            R, _ = cv2.Rodrigues(rvecs)

        cameras[i].setRt(R, t)
        triangulate(cameras[i - 1], cameras[i], idx0, idx1, K)

    to_ply("sfm_data/", np.array(point_cloud))
    to_ply("sfm_data/", np.array([cam.getPos() for cam in cameras]), '_campos.ply')