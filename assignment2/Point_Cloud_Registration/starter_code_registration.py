import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def load_point_cloud(path):
    # Load the point cloud data (do NOT change this function!)
    data = pd.read_csv(path, header=None)
    point_cloud = data.to_numpy()
    return point_cloud


def nearest_search(pcd_source, pcd_target):
    # corr_target = []
    # corr_source = []
    # ec_dist_mean = 0

    # for i in range(len(pcd_source)):
    #     min_distance = float('inf') 
    #     nearest_point_index = -1

    #     # Brute-force search for the nearest neighbor in the target point cloud
    #     for j in range(len(pcd_target)):
    #         distance = np.linalg.norm(np.asarray(pcd_source[i]) - np.asarray(pcd_target[j]))

    #         if distance < min_distance:
    #             min_distance = distance
    #             nearest_point_index = j

    #     corr_target.append(pcd_target[nearest_point_index])
    #     corr_source.append(pcd_source[i])

    #     # Accumulate distances for mean calculation
    #     ec_dist_mean += min_distance

    # # Compute the mean nearest Euclidean distance
    # ec_dist_mean /= len(pcd_source)

    nearestN = NearestNeighbors(n_neighbors=1).fit(pcd_target)
    dist, ind = nearestN.kneighbors(pcd_source)
    ec_dist_mean = np.mean(dist)

    return ind, ec_dist_mean



def estimate_pose(corr_source, corr_target):
    centroid_source = np.mean(corr_source, axis=0)
    centroid_target = np.mean(corr_target, axis=0)

    centered_source = corr_source - centroid_source
    centered_target = corr_target - centroid_target

    H = corr_target.T @ corr_source / len(corr_source)

    V, _, U_T = np.linalg.svd(H,full_matrices = False)

    C = np.identity(3)
    C[2, 2] = np.linalg.det(V) * np.linalg.det(U_T.T)

    R = V @ C @ U_T

    t = centroid_target - R @ centroid_source

    pose = np.identity(4)
    pose[:3, :3] = R
    pose[:3, 3] = t

    translation_x, translation_y, translation_z = pose[:3, 3]

    return pose, translation_x, translation_y, translation_z


def icp(pcd_source, pcd_target, max_iterations=30, plot_results=False):
    mean_distances = []
    translations = [] 
    pcd_source_copy = pcd_source.copy()
    pts = np.vstack([np.transpose(pcd_source), np.ones(len(pcd_source))])

    for iteration in range(max_iterations):
        ind, dist = nearest_search(pcd_source_copy, pcd_target)
        corr_source = pcd_source
        corr_target = pcd_target[ind[:, 0]]

        pose, translation_x, translation_y, translation_z = estimate_pose(corr_source, corr_target)

        pcd_source_copy = np.matmul(pose, pts).T[:, 0:3]
 

        translations.append([translation_x, translation_y, translation_z])

        mean_distances.append(dist)


        if plot_results:
            plt.figure(figsize=(12, 4))

            # Plot mean Euclidean distance
            plt.subplot(1, 2, 1)
            plt.plot(mean_distances, marker='o')
            plt.title('Mean Euclidean Distance')
            plt.xlabel('Iteration')
            plt.ylabel('Distance')
            plt.title(f'ICP Iterations: {iteration + 1}')  # Add title with iteration number

            # Plot pose translation for each axis
            plt.subplot(1, 2, 2)
            translations_array = np.array(translations)
            plt.plot(translations_array[:, 0], label='X', marker='o')
            plt.plot(translations_array[:, 1], label='Y', marker='o')
            plt.plot(translations_array[:, 2], label='Z', marker='o')
            plt.title('Pose Translation')
            plt.xlabel('Iteration')
            plt.ylabel('Translation')
            plt.legend()

            plt.tight_layout()
            plt.show()

    return pose




def main():
    # Dataset and ground truth poses
    #########################################################################################
    # Training and test data (3 pairs in total)
    train_file = ['bunny', 'dragon']
    test_file = ['armadillo']

    # Ground truth pose (from training data only, used for validating your implementation)
    GT_poses = []
    gt_pose = [0.8738,-0.1128,-0.4731,24.7571,
            0.1099,0.9934,-0.0339,4.5644,
            0.4738,-0.0224,0.8804,10.8654,
            0.0,0.0,0.0,1.0]
    gt_pose = np.array(gt_pose).reshape([4,4])
    GT_poses.append(gt_pose)
    gt_pose = [0.7095,-0.3180,0.6289,46.3636,
               0.3194,0.9406,0.1153,3.3165,
               -0.6282,0.1191,0.7689,-6.4642,
               0.0,0.0,0.0,1.0]
    gt_pose = np.array(gt_pose).reshape([4,4])
    GT_poses.append(gt_pose)
    #########################################################################################



    # Training (validate your algorithm)
    ##########################################################################################################
    for i in range(2):
        # Load data
        path_source = './training/' + train_file[i] + '_source.csv'
        path_target = './training/' + train_file[i] + '_target.csv'
        pcd_source = load_point_cloud(path_source)
        pcd_target = load_point_cloud(path_target)
        gt_pose_i = GT_poses[i]

        # Visualize the point clouds before the registration
        ax = plt.axes(projection='3d')
        ax.scatter3D(pcd_source[:,0], pcd_source[:,1], pcd_source[:,2], cmap='Greens')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], cmap='Reds')
        plt.legend(["Source Point Cloud" , "Target Point Cloud"])
        ax.set_title('Point Clouds Before Registration: Training')
        plt.show()



        # TODO: Use your implemented ICP algorithm to get the estimated 6D pose (from source to target point cloud)
        pose = icp(pcd_source, pcd_target)
        print("pose")
        print(pose)
        # Transform the point cloud
        # TODO: Replace the ground truth pose with your computed pose and transform the source point cloud
        pts = np.vstack([np.transpose(pcd_source), np.ones(len(pcd_source))])
        cloud_registered = np.matmul(pose, pts)
        cloud_registered = np.transpose(cloud_registered[0:3, :])

        # TODO: Evaluate the rotation and translation error of your estimated 6D pose with the ground truth pose
        print("rotational error matrix")
        rotational_matrix_err = np.matmul(pose[0:3, 0:3], (gt_pose_i[0:3, 0:3]).T)
        print(rotational_matrix_err)
        rotation_error = np.arccos((np.trace(np.matmul(pose[0:3, 0:3], (gt_pose_i[0:3, 0:3]).T)) - 1) / 2)
        translation_error = np.abs(pose[0:3, 3] - gt_pose_i[0:3, 3])
        print("rotatinoal error")

        print(rotation_error)
        print("translational Error")

        print(translation_error)

        # Visualize the point clouds after the registration
        ax = plt.axes(projection='3d')
        ax.scatter3D(cloud_registered[:,0], cloud_registered[:,1], cloud_registered[:,2], cmap='Greens')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], cmap='Reds')
        plt.legend(["Transformed Source Point Cloud", "Target Point Cloud"])
        ax.set_title('Point Clouds After Registration:Training')
        plt.show()
    ##########################################################################################################



    # Test
    ####################################################################################
    for i in range(1):
        # Load data
        path_source = './test/' + test_file[i] + '_source.csv'
        path_target = './test/' + test_file[i] + '_target.csv'
        pcd_source = load_point_cloud(path_source)
        pcd_target = load_point_cloud(path_target)

        # Visualize the point clouds before the registration
        ax = plt.axes(projection='3d')
        ax.scatter3D(pcd_source[:,0], pcd_source[:,1], pcd_source[:,2], cmap='Greens')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], cmap='Reds')
        plt.legend(["Source Point Cloud" , "Target Point Cloud"])
        ax.set_title('Point Clouds Before Registration:Test')
        plt.show()

        # TODO: Use your implemented ICP algorithm to get the estimated 6D pose (from source to target point cloud)
        pose = icp(pcd_source, pcd_target)
        print("test pose")
        print(pose)
        
        # TODO: Show your outputs in the report
        pts = np.vstack([np.transpose(pcd_source), np.ones(len(pcd_source))])
        cloud_registered = np.matmul(pose, pts)
        cloud_registered = np.transpose(cloud_registered[0:3, :])

        # Visualize the point clouds after the registration
        ax = plt.axes(projection='3d')
        ax.scatter3D(cloud_registered[:,0], cloud_registered[:,1], cloud_registered[:,2], cmap='Greens')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], cmap='Reds')
        plt.legend(["Transformed Source Point Cloud", "Target Point Cloud"])
        ax.set_title('Point Clouds After Registration:Test')
        plt.show()


if __name__ == '__main__':
    main()