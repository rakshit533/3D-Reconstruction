import cv2
import numpy as np
import os

# Directory containing the pre-captured images
IMAGE_DIR = r"C:\Users\DELL\OneDrive\Desktop\code\3D_Reconstruction\SceauxCastle"

# Parameters
MIN_MATCH_COUNT = 10  # Minimum number of matches for feature matching

def load_images(image_dir):
    """Load images from the specified directory."""
    images = []
    filenames = sorted(os.listdir(image_dir))
    for filename in filenames:
        if filename.endswith(('.JPG', '.png', '.jpeg')):
            print(filename)
            filepath = os.path.join(image_dir, filename)
            img = cv2.imread(filepath)
            images.append(img)
    return images

def feature_matching(img1, img2):
    """Perform feature matching between two images using ORB."""
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    return kp1, kp2, matches

def reconstruct_3d(images):
    """Reconstruct 3D points from a set of images."""
    if len(images) < 2:
        raise ValueError("At least two images are required for 3D reconstruction.")

    # Convert images to grayscale
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    # Initialize a list to store the 3D points
    points_3d = []

    # Assuming camera intrinsic matrix (replace with calibrated values if available)
    K = np.array([[1000, 0, 500], 
                  [0, 1000, 500], 
                  [0, 0, 1]])

    for i in range(len(gray_images) - 1):
        img1 = gray_images[i]
        img2 = gray_images[i + 1]

        print(f"Processing image pair {i + 1} and {i + 2}...")

        # Match features between two consecutive images
        kp1, kp2, matches = feature_matching(img1, img2)

        if len(matches) < MIN_MATCH_COUNT:
            print(f"Not enough matches are found - {len(matches)}/{MIN_MATCH_COUNT}")
            continue

        # Extract points from matches
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Compute the fundamental matrix
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

        # Select inlier points
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

        # Compute the essential matrix
        E = K.T @ F @ K

        # Decompose essential matrix to retrieve relative pose
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

        # Triangulate points to get 3D coordinates
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R, t))
        points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_4d = points_4d_hom / points_4d_hom[3]  # Convert to non-homogeneous coordinates

        # Append 3D points
        points_3d.append(points_4d[:3].T)

    return points_3d

def visualize_3d(points_3d):
    """Visualize the 3D points using Open3D."""
    import open3d as o3d

    # Flatten the list of 3D points
    all_points = np.vstack(points_3d)

    print(f"Shape of all_points: {all_points.shape}")
    
    # Remove non-finite values
    if not np.all(np.isfinite(all_points)):
        print("Non-finite values found in all_points!")
        all_points = all_points[np.isfinite(all_points).all(axis=1)]

    # Downsample if too many points
    if all_points.shape[0] > 100000:
        all_points = all_points[::10]
        print(f"Downsampled points to: {all_points.shape[0]} points")

    # Check for empty points
    if all_points.shape[0] == 0:
        print("No points to visualize!")
        return

    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    print("reached here")
    pcd.points = o3d.utility.Vector3dVector(all_points)
    print("reached here")

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

# Main script
if __name__ == "__main__":
    print("Loading images...")
    images = load_images(IMAGE_DIR)
    print(images)

    print("Reconstructing 3D points...")
    points_3d = reconstruct_3d(images)

    print("Visualizing 3D points...")
    visualize_3d(points_3d)
