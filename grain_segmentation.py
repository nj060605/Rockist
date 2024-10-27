import numpy as np
import cv2
import sys
import json
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial import distance
from PIL import Image, ImageEnhance

def adjust_image_properties(image, brightness=1.2, contrast=1.2, sharpness=1.5, saturation=1.3):
    """
    Adjust brightness, contrast, sharpness, and saturation of an image.
    """
    pil_img = Image.fromarray(image)
    pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(sharpness)
    pil_img = ImageEnhance.Color(pil_img).enhance(saturation)
    return np.array(pil_img)

def find_optimal_clusters(image_lab, max_clusters=10):
    """
    Finds the optimal number of clusters based on the Elbow method.
    """
    distortions = []
    K_range = range(2, max_clusters + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(image_lab)
        distortions.append(kmeans.inertia_)  # Sum of squared distances to cluster centers

    # Plot to visually inspect the Elbow
    # plt.figure(figsize=(8, 4))
    # plt.plot(K_range, distortions, 'bx-')
    # plt.xlabel('Number of Clusters (K)')
    # plt.ylabel('Distortion')
    # plt.title('Elbow Method for Optimal K')
    # plt.show()

    # Select the elbow point as the optimal number of clusters
    optimal_k = np.argmin(np.diff(distortions, 2)) + 2  # Second derivative for the elbow point
    return optimal_k

def segment_and_group_grains(image_path, max_clusters=10, color_distance_threshold=15, area_scale=1.0):
    # Load and adjust the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    adjusted_image_rgb = adjust_image_properties(image_rgb)

    # Convert to LAB color space for better color segmentation
    image_lab = cv2.cvtColor(adjusted_image_rgb, cv2.COLOR_RGB2LAB)
    reshaped_lab = image_lab.reshape((-1, 3))

    # Find the optimal number of clusters
    num_clusters = find_optimal_clusters(reshaped_lab, max_clusters)

    # Apply k-means clustering with optimal clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(reshaped_lab)
    cluster_centers_lab = kmeans.cluster_centers_  # Get LAB color centers for each cluster

    # Convert cluster centers to RGB for visualization
    cluster_centers_rgb = cv2.cvtColor(cluster_centers_lab.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_LAB2RGB).reshape(-1, 3)

    # Reshape the labels to match the original image shape
    segmented_image = labels.reshape((image_lab.shape[:2]))

    # Dictionary to store total area per color cluster
    grain_areas_by_color = {}
    for i in range(num_clusters):
        # Create a binary mask where the cluster equals the current label
        mask = np.uint8(segmented_image == i) * 255

        # Find contours to detect each grain in the cluster
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt) * area_scale  # Scale area to desired units
            if area > 10 * area_scale:  # Filter out small contours
                # Add the area to the corresponding cluster's list
                if i not in grain_areas_by_color:
                    grain_areas_by_color[i] = []
                grain_areas_by_color[i].append(area)

                # Draw contours on the image for visualization
                cv2.drawContours(adjusted_image_rgb, [cnt], -1, (255, 0, 0), 2)

    # Group clusters with similar colors
    grouped_areas = {}
    grouped_colors = {}
    for i, center_i in enumerate(cluster_centers_lab):
        if i in grouped_areas:
            continue  # Skip if already grouped

        grouped_areas[i] = sum(grain_areas_by_color[i])
        grouped_colors[i] = cluster_centers_rgb[i]  # Assign RGB color to the group

        for j, center_j in enumerate(cluster_centers_lab):
            if i != j and j not in grouped_areas:
                color_distance = distance.euclidean(center_i, center_j)

                # Group areas if color distance is within threshold
                if color_distance < color_distance_threshold:
                    grouped_areas[i] += sum(grain_areas_by_color[j])
                    grouped_colors[i] = cluster_centers_rgb[i]
                    grouped_areas[j] = 0  # Mark as grouped

    # Filter out empty groups and prepare the final result
    final_grouped_areas = {k: v for k, v in grouped_areas.items() if v > 0}
    final_grouped_colors = [grouped_colors[k] / 255.0 for k in final_grouped_areas.keys()] # Normalize colors for plotting

    # Display the segmented image with contours
    # cv2.imwrite("finalmask.png", adjusted_image_rgb)
    # plt.imshow(adjusted_image_rgb)
    # plt.axis('off')
    # plt.show()

    # Plotting the pie chart
    # plt.figure(figsize=(8, 6))
    # plt.pie(final_grouped_areas.values(), labels=[f"Cluster {i}" for i in final_grouped_areas.keys()],
    #         colors=final_grouped_colors, autopct='%1.1f%%')
    # plt.title("Grain Areas Grouped by Similar Color")
    # plt.show()
    final_grouped_colors = [i.tolist() for i in final_grouped_colors]
    s = 0
    for i in final_grouped_areas.values():
        s += i
    per = [(i/s)*100 for i in final_grouped_areas.values()]
    # out_file = open("new.json", "w")
    # final = {
    #     "percentages": per,
    #     "pixels": final_grouped_areas.values(),
    #     "colors": final_grouped_colors
    # }
    # out_file.close()
    # print(final)
    return [per, list(final_grouped_areas.values()), final_grouped_colors]

# Path to your microscopic image
# image_path = 'mineral2.jpeg'
# final_areas = segment_and_group_grains(image_path, area_scale=0.01)

# Print the final grouped areas based on similar colors
# print("Final grouped areas based on similar colors:", final_areas)


if __name__ == "__main__":
    # Get command-line arguments
    image_path = sys.argv[1]  # First argument: Image path
    remove_background = sys.argv[2].lower() == 'true'  # Second argument: Background removal
    area_scale = float(sys.argv[3])  # Third argument: Area scale

    # Call the function and return the result
    final = segment_and_group_grains(image_path, area_scale=area_scale)
    print(final) 