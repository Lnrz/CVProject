import numpy as np
import pycolmap as col
from scipy.spatial import KDTree

# TODO to improve performance parallellize the processing
# TODO to improve performance ignore points hidden by already processed points

def main():
    print("Loading data...")

    # load reconstructions
    rec2_path = "../../ColmapWorkspace/CVProject/roomCornerNaturalWarmLightSparseReconstruction"
    rec1 = col.Reconstruction()
    rec2 = col.Reconstruction(rec2_path)

    # load the similiarity from disk
    rec2_from_rec1 = col.Sim3d(
        translation=np.load("local/translation.npy"),
        rotation=col.Rotation3d(np.load("local/rotation.npy")),
        scale=np.load("local/scale.npy")
    )
    
    # load the points of the dense reconstruction
    rec1_ply_path = "../../ColmapWorkspace/CVProject/roomCornerNaturalLightDenseReconstruction/fused.ply"
    rec1.import_PLY(rec1_ply_path)
    
    # put the points of the dense reconstruction in a numpy array and transform them to rec2 space
    points = np.array([rec2_from_rec1 * point.xyz for point in rec1.points3D.values()], dtype=np.float64)

    # free some memory
    # removing 3d points from rec2 is a step for when we will use rec2 to write the ply file 
    del rec1
    for point_id in rec2.points3D.keys():
        rec2.delete_point3D(point_id)

    # data structure for keeping colors of every point
    colors_per_point = [[] for i in range(points.size)]

    # PARAMETERS
    neighbor_distance = 15 # in pixel
    max_depth_difference = 1 # in the unit of measure of the 3D model

    print("Processing images...")
    image_index = 0
    for image in rec2.images.values():
        image: col.Image

        # check if image is of a certain type
        # if not skip it
        image_type = "Warm"
        if image_type not in image.name:
            continue

        image_index += 1
        print("    Processing " + str(image_index) + "th image...")
        print("    Image is", image.name)

        # get image size
        width = image.camera.width
        height = image.camera.height

        # load image in memory
        # the array has size [height, width, 3]
        image_data = col.Bitmap.read("../../ColmapWorkspace/CVProject/images/" + image.name, True).to_array()

        # project point on image (discarding points behind the camera)
        projected_points = [(point_index, image.project_point(point))
                                for point_index, point in enumerate(points)
                                if image.project_point(point) is not None]
        projected_points_info = np.array([[point_index, point[0], point[1]]
                                          for point_index, point in projected_points],
                                    dtype=np.float64)
        del projected_points

        # put them in a 2dtree
        points_tree = KDTree(projected_points_info[:,1:3])
        
        for projected_point_info in projected_points_info:
            projected_point_index = int(projected_point_info[0])
            projected_point = projected_point_info[1:3]
            
            # get the point depth in 3D
            point_depth = points[projected_point_index][2]
            
            # if the point is projected outside the visible part of the image ignore it
            if projected_point[0] < 0 or projected_point[1] < 0 or projected_point[0] > width or projected_point[1] > height:
                continue 
            
            # get the indices of the neighbors
            neighbor_indices = points_tree.query_ball_point(projected_point, neighbor_distance)
            
            # get the neighbor depths
            neighbor_depths = points[projected_points_info[neighbor_indices][:,0].astype(np.int64)][:,2]
            
            # get the depth of the neighbor nearest to the camera
            lowest_depth = np.min(neighbor_depths, initial=np.inf)

            # if it can find a neighbor closer to the camera by a certain difference
            # assume that this point is not actually seen in the image
            if (lowest_depth != np.inf) and (point_depth - lowest_depth > max_depth_difference):
                continue
            
            # get color from image
            color = image_data[int(np.floor(projected_point[1])), int(np.floor(projected_point[0]))] # TODO do bilinear or trilinear interpolation

            # add color to point
            colors_per_point[projected_point_index].append(color)

    # calculate the color of the points
    colors = np.array(  [np.mean(colors, axis=0).astype(np.uint8)
                            if len(colors) > 0
                            else np.array([0, 0, 0], dtype=np.uint8) # TODO manage not colored points
                            for colors in colors_per_point])
    
    # add points to reconstruction
    print("Coloring...")
    dummy_track = col.Track()
    for index in range(points.shape[0]):
        rec2.add_point3D(points[index], dummy_track, colors[index])

    # TODO need a way to store the index of not colored point
    # TODO to color not colored point (if they are a low % otherwise don't do it) median filter on every channel of HSV

    # write ply file
    print("Writing ply...")
    ply_out_path = "out.ply"
    rec2.export_PLY(ply_out_path)

if __name__ == "__main__":
    main()