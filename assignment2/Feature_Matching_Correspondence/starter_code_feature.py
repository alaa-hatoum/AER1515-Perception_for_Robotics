import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import csv
import os

class FrameCalib:
    """Frame Calibration

    Fields:
        p0-p3: (3, 4) Camera P matrices. Contains extrinsic and intrinsic parameters.
        r0_rect: (3, 3) Rectification matrix
        velo_to_cam: (3, 4) Transformation matrix from velodyne to cam coordinate
            Point_Camera = P_cam * R0_rect * Tr_velo_to_cam * Point_Velodyne
        """

    def __init__(self):
        self.p0 = []
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.r0_rect = []
        self.velo_to_cam = []


def read_frame_calib(calib_file_path):
    """Reads the calibration file for a sample

    Args:
        calib_file_path: calibration file path

    Returns:
        frame_calib: FrameCalib frame calibration
    """

    data_file = open(calib_file_path, 'r')
    data_reader = csv.reader(data_file, delimiter=' ')
    data = []

    for row in data_reader:
        data.append(row)

    data_file.close()

    p_all = []

    for i in range(4):
        p = data[i]
        p = p[1:]
        p = [float(p[i]) for i in range(len(p))]
        p = np.reshape(p, (3, 4))
        p_all.append(p)

    frame_calib = FrameCalib()
    frame_calib.p0 = p_all[0]
    frame_calib.p1 = p_all[1]
    frame_calib.p2 = p_all[2]
    frame_calib.p3 = p_all[3]

    # Read in rectification matrix
    tr_rect = data[4]
    tr_rect = tr_rect[1:]
    tr_rect = [float(tr_rect[i]) for i in range(len(tr_rect))]
    frame_calib.r0_rect = np.reshape(tr_rect, (3, 3))

    # Read in velodyne to cam matrix
    tr_v2c = data[5]
    tr_v2c = tr_v2c[1:]
    tr_v2c = [float(tr_v2c[i]) for i in range(len(tr_v2c))]
    frame_calib.velo_to_cam = np.reshape(tr_v2c, (3, 4))

    return frame_calib


class StereoCalib:
    """Stereo Calibration

    Fields:
        baseline: distance between the two camera centers
        f: focal length
        k: (3, 3) intrinsic calibration matrix
        p: (3, 4) camera projection matrix
        center_u: camera origin u coordinate
        center_v: camera origin v coordinate
        """

    def __init__(self):
        self.baseline = 0.0
        self.f = 0.0
        self.k = []
        self.center_u = 0.0
        self.center_v = 0.0


def krt_from_p(p, fsign=1):
    """Factorize the projection matrix P as P=K*[R;t]
    and enforce the sign of the focal length to be fsign.


    Keyword Arguments:
    ------------------
    p : 3x4 list
        Camera Matrix.

    fsign : int
            Sign of the focal length.


    Returns:
    --------
    k : 3x3 list
        Intrinsic calibration matrix.

    r : 3x3 list
        Extrinsic rotation matrix.

    t : 1x3 list
        Extrinsic translation.
    """
    s = p[0:3, 3]
    q = np.linalg.inv(p[0:3, 0:3])
    u, b = np.linalg.qr(q)
    sgn = np.sign(b[2, 2])
    b = b * sgn
    s = s * sgn

    # If the focal length has wrong sign, change it
    # and change rotation matrix accordingly.
    if fsign * b[0, 0] < 0:
        e = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    if fsign * b[2, 2] < 0:
        e = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    # If u is not a rotation matrix, fix it by flipping the sign.
    if np.linalg.det(u) < 0:
        u = -u
        s = -s

    r = np.matrix.transpose(u)
    t = np.matmul(b, s)
    k = np.linalg.inv(b)
    k = k / k[2, 2]

    # Sanity checks to ensure factorization is correct
    if np.linalg.det(r) < 0:
        print('Warning: R is not a rotation matrix.')

    if k[2, 2] < 0:
        print('Warning: K has a wrong sign.')

    return k, r, t


def get_stereo_calibration(left_cam_mat, right_cam_mat):
    """Extract parameters required to transform disparity image to 3D point
    cloud.

    Keyword Arguments:
    ------------------
    left_cam_mat : 3x4 list
                   Left Camera Matrix.

    right_cam_mat : 3x4 list
                   Right Camera Matrix.


    Returns:
    --------
    stereo_calibration_info : Instance of StereoCalibrationData class
                              Placeholder for stereo calibration parameters.
    """

    stereo_calib = StereoCalib()
    k_left, r_left, t_left = krt_from_p(left_cam_mat)
    _, _, t_right = krt_from_p(right_cam_mat)

    stereo_calib.baseline = abs(t_left[0] - t_right[0])
    stereo_calib.f = k_left[0, 0]
    stereo_calib.k = k_left
    stereo_calib.center_u = k_left[0, 2]
    stereo_calib.center_v = k_left[1, 2]

    return stereo_calib

    
def feature_detection(img):
    orb = cv.ORB_create(nfeatures=1000)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des

def Q1():
    ## Input directory
    left_image_dir = os.path.abspath('./test/left')
    right_image_dir = os.path.abspath('./test/right')
    ## Output directory
    result_imgs_dir_left = os.path.abspath('./results/left')
    result_imgs_dir_right = os.path.abspath('./results/right')
    

    ## Detect Keypoints
    for img_path in os.listdir(left_image_dir):
        img_path = os.path.join(left_image_dir, img_path)
        img = cv.imread(img_path, 0)
        kp_l, des_l = feature_detection(img)
        img_with_kp_l = cv.drawKeypoints(img, kp_l, None, 
                                 flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        
        cv.imwrite(os.path.join(result_imgs_dir_left, f'{img_path.replace("/", "_")}_keypoints.jpg'), img_with_kp_l)

    for img_path in os.listdir(right_image_dir):
        img_path = os.path.join(right_image_dir, img_path)
        img = cv.imread(img_path, 0)
        kp_l, des_l = feature_detection(img)
        img_with_kp_r = cv.drawKeypoints(img, kp_l, None,
                                 flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv.imwrite(os.path.join(result_imgs_dir_right, f'{img_path.replace("/", "_")}_keypoints.jpg'), img_with_kp_r)


def Q2(training = True):
    if training:
        ## Input
        left_image_dir = os.path.abspath('./training/left')
        right_image_dir = os.path.abspath('./training/right')
        calib_dir = os.path.abspath('./training/calib')
        sample_list = ['000001', '000002', '000003', '000004','000005', '000006', '000007', '000008', '000009', '000010']
    else:
        ## Input
        left_image_dir = os.path.abspath('./test/left')
        right_image_dir = os.path.abspath('./test/right')
        calib_dir = os.path.abspath('./test/calib')
        sample_list = ['000011', '000012', '000013', '000014','000015']

    ## Output
    output_file = open("P3_result.txt", "a")
    output_file.truncate(0)

    ## Main
    for sample_name in sample_list:
        left_image_path = left_image_dir +'/' + sample_name + '.png'
        right_image_path = right_image_dir +'/' + sample_name + '.png'

        img_left = cv.imread(left_image_path, 0)
        img_right = cv.imread(right_image_path, 0)

        kp_left, des_left = feature_detection(img_left)
        kp_right, des_right = feature_detection(img_right)

        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_left,des_right)

        matches = sorted(matches, key = lambda x:x.distance)      


        # Read calibration
        frame_calib = read_frame_calib(calib_dir+'/' + sample_name + '.txt')
        stereo_calib = get_stereo_calibration(frame_calib.p2, frame_calib.p3)

        # Find disparity and depth
        pixel_u_list = [] # x pixel on left image
        pixel_v_list = [] # y pixel on left image
        disparity_list = []
        depth_list = []
        for i, match in enumerate(matches):
            pt_left = kp_left[match.queryIdx].pt
            pt_right = kp_right[match.trainIdx].pt
            disparity = pt_left[0] - pt_right[0]
            pixel_u_list.append(round(pt_left[0]))
            pixel_v_list.append(round(pt_left[1]))
            disparity_list.append(disparity)
            depth_list.append(stereo_calib.f * stereo_calib.baseline / disparity)

        # Output
        for u, v, disp, depth in zip(pixel_u_list, pixel_v_list, disparity_list, depth_list):
            line = "{} {:.2f} {:.2f} {:.2f} {:.2f}".format(sample_name, u, v, disp, depth)
            output_file.write(line + '\n')

        # Draw matches
        img_with_matches = cv.drawMatches(img_left, kp_left, img_right, kp_right, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img_with_matches)
        plt.show()

        if training:
            results_imgs_dir = os.path.abspath('./q2_results/training')
            cv.imwrite(os.path.join(results_imgs_dir, f'{sample_name}_matches.jpg'), img_with_matches)
        else:
            results_imgs_dir = os.path.abspath('./q2_results/test')
            cv.imwrite(os.path.join(results_imgs_dir, f'{sample_name}_matches.jpg'), img_with_matches)


def Q2_hor(training = True):
    if training:
        ## Input
        left_image_dir = os.path.abspath('./training/left')
        right_image_dir = os.path.abspath('./training/right')
        calib_dir = os.path.abspath('./training/calib')
        sample_list = ['000001', '000002', '000003', '000004','000005', '000006', '000007', '000008', '000009', '000010']
    else:
        ## Input
        left_image_dir = os.path.abspath('./test/left')
        right_image_dir = os.path.abspath('./test/right')
        calib_dir = os.path.abspath('./test/calib')
        sample_list = ['000011', '000012', '000013', '000014','000015']

    ## Output
    output_file = open("P3_result.txt", "a")
    output_file.truncate(0)

    ## Main
    for sample_name in sample_list:
        left_image_path = left_image_dir +'/' + sample_name + '.png'
        right_image_path = right_image_dir +'/' + sample_name + '.png'

        img_left = cv.imread(left_image_path, 0)
        img_right = cv.imread(right_image_path, 0)

        kp_left, des_left = feature_detection(img_left)
        kp_right, des_right = feature_detection(img_right)

        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_left,des_right)

        matches = sorted(matches, key = lambda x:x.distance)
        bestMatches = int(len(matches) * 0.8)
        matches = matches[:bestMatches]

        horizontal_matches = matches.copy()

        mat = 0

        while mat < len(horizontal_matches):
            if (kp_left[horizontal_matches[mat].queryIdx].pt)[1] != (kp_right[horizontal_matches[mat].trainIdx].pt)[1]:
                del horizontal_matches[mat]
            else:
                mat += 1

        # Read calibration
        frame_calib = read_frame_calib(calib_dir+'/' + sample_name + '.txt')
        stereo_calib = get_stereo_calibration(frame_calib.p2, frame_calib.p3)

        # Find disparity and depth
        pixel_u_list = [] # x pixel on left image
        pixel_v_list = [] # y pixel on left image
        disparity_list = []
        depth_list = []
        for i, match in enumerate(horizontal_matches):
            pt_left = kp_left[match.queryIdx].pt
            pt_right = kp_right[match.trainIdx].pt
            disparity = pt_left[0] - pt_right[0]
            if disparity == 0:
                del horizontal_matches[i]
            else:
                pixel_u_list.append(round(pt_left[0]))
                pixel_v_list.append(round(pt_left[1]))
                disparity_list.append(disparity)
                depth_list.append(stereo_calib.f * stereo_calib.baseline / disparity)

        # Output
        for u, v, disp, depth in zip(pixel_u_list, pixel_v_list, disparity_list, depth_list):
            line = "{} {:.2f} {:.2f} {:.2f} {:.2f}".format(sample_name, u, v, disp, depth)
            output_file.write(line + '\n')

        # Draw matches
        img_with_matches = cv.drawMatches(img_left, kp_left, img_right, kp_right, horizontal_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img_with_matches)
        plt.show()

        if training:
            results_imgs_dir = os.path.abspath('./q2_hor_results/training')
            cv.imwrite(os.path.join(results_imgs_dir, f'{sample_name}_matches.jpg'), img_with_matches)
        else:
            results_imgs_dir = os.path.abspath('./q2_hor_results/test')
            cv.imwrite(os.path.join(results_imgs_dir, f'{sample_name}_matches.jpg'), img_with_matches)


    output_file.close()

def Q3(training = True):
    if training:
        ## Input
        left_image_dir = os.path.abspath('./training/left')
        right_image_dir = os.path.abspath('./training/right')
        calib_dir = os.path.abspath('./training/calib')
        sample_list = ['000001', '000002', '000003', '000004','000005', '000006', '000007', '000008', '000009', '000010']
    else:
        ## Input
        left_image_dir = os.path.abspath('./test/left')
        right_image_dir = os.path.abspath('./test/right')
        calib_dir = os.path.abspath('./test/calib')
        sample_list = ['000011', '000012', '000013', '000014','000015']

    ## Output
    output_file = open("P3_result.txt", "a")
    output_file.truncate(0)

    ## Main
    for sample_name in sample_list:
        left_image_path = left_image_dir +'/' + sample_name + '.png'
        right_image_path = right_image_dir +'/' + sample_name + '.png'

        img_left = cv.imread(left_image_path, 0)
        img_right = cv.imread(right_image_path, 0)

        kp_left, des_left = feature_detection(img_left)
        kp_right, des_right = feature_detection(img_right)

        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_left,des_right)

        matches = sorted(matches, key = lambda x:x.distance)
        bestMatches = int(len(matches) * 0.8)
        matches = matches[:bestMatches]
        
        # Vectorized Data Preparation
        points1 = np.float32([kp_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        points2 = np.float32([kp_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # RANSAC Homography Estimation
        h, status = cv.findHomography(points1, points2, cv.RANSAC, 
                                        ransacReprojThreshold=5.0, maxIters=1000, confidence=0.99) 

        # RANSAC Result Analysis with NumPy
        inliers = status.ravel() == 1
        outliers = len(matches) - np.sum(inliers)

        # Visualization
        draw_params = dict(singlePointColor=None,
                        matchesMask=inliers.tolist(),  # draw only inliers
                        flags=2)

        ransac_matches = [matches[i] for i in range(len(matches)) if inliers[i]]



        # Read calibration
        frame_calib = read_frame_calib(calib_dir+'/' + sample_name + '.txt')
        stereo_calib = get_stereo_calibration(frame_calib.p2, frame_calib.p3)

        # Find disparity and depth
        pixel_u_list = [] # x pixel on left image
        pixel_v_list = [] # y pixel on left image
        disparity_list = []
        depth_list = []

        for match in ransac_matches:
                pt_left = kp_left[match.queryIdx].pt
                pt_right = kp_right[match.trainIdx].pt
                disparity = pt_left[0] - pt_right[0]
                if disparity != 0:
                    pixel_u_list.append(round(pt_left[0]))
                    pixel_v_list.append(round(pt_left[1]))
                    disparity_list.append(disparity)
                    depth_list.append(stereo_calib.f * stereo_calib.baseline / disparity)

        # Output
        for u, v, disp, depth in zip(pixel_u_list, pixel_v_list, disparity_list, depth_list):
            if depth<80:
                line = "{} {:.2f} {:.2f} {:.2f} {:.2f}".format(sample_name, u, v, disp, depth)
                output_file.write(line + '\n')

        # Draw matches
        img_with_matches = cv.drawMatches(img_left, kp_left, img_right, kp_right, ransac_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img_with_matches)
        plt.show()

        if training:
            results_imgs_dir = os.path.abspath('./q3/training')
            cv.imwrite(os.path.join(results_imgs_dir, f'{sample_name}_matches.jpg'), img_with_matches)
        else:
            results_imgs_dir = os.path.abspath('./q3/test')
            cv.imwrite(os.path.join(results_imgs_dir, f'{sample_name}_matches.jpg'), img_with_matches)


    output_file.close()


def main():
    Q1()
    Q2_hor(training = True)
    Q3(True)

if __name__ == "__main__":
    main()