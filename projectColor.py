import numpy as np
import pycolmap as col
from scipy.spatial import KDTree

# TODO to improve performance parallellize the processing
# TODO to improve performance ignore points hidden by already processed points

def main():
    print("Loading data...")

    # load reconstruction and model
    rec_path = "../../ColmapWorkspace/CVProject/roomCornerNaturalWarmLightSparseReconstruction"
    ply_in_path = "../../ColmapWorkspace/CVProject/roomCornerNaturalLightDenseReconstruction/fused.ply"
    rec = col.Reconstruction(rec_path)
    ply = col.Reconstruction()
    ply.import_PLY(ply_in_path)

    # load the similiarity from disk
    rec_similiarity = col.Sim3d(
        translation=np.load("local/translation.npy"),
        rotation=col.Rotation3d(np.load("local/rotation.npy")),
        scale=np.load("local/scale.npy")
    )
    
    # apply similiarity to model's points
    ply.transform(rec_similiarity)

    # extract points from model
    points = np.array([point.xyz for point in ply.points3D.values()], dtype=np.float64)

    # unload model
    del ply
    
    # data structure for keeping the colors of every point
    colors_per_point = [[] for i in range(points.size)]

    # PARAMETERS
    neighbor_distance = 15 # in pixel
    max_depth_difference = 1 # in the unit of measure of the 3D model
    
    # Filter images
    images_id = []
    image_filter = "Warm"
    for image in rec.images.values():
        if image_filter in image.name:
            images_id.append(image.image_id)

    # Process filtered images
    print("Processing images...")
    for image_index, image_id in enumerate(images_id):
        image = rec.image(image_id)
        
        print("    Processing " + str(image_index + 1) + "th image...")
        print("    Image is", image.name)

        # get image camera
        camera = image.camera

        # get image size
        width = image.camera.width
        height = image.camera.height

        # load image in memory
        # the array has size [height, width, 3]
        image_data = col.Bitmap.read("../../ColmapWorkspace/CVProject/images/" + image.name, True).to_array()

        # project points on image
        projected_points = camera.img_from_cam(image.cam_from_world() * points)

        # calculate boolean array checking for nans (i.e. points behind the camera)
        is_not_behind_camera = np.logical_not(np.isnan(projected_points[:,0]))

        # put valid points in a 2dtree
        points_tree = KDTree(projected_points[is_not_behind_camera])
        
        for projected_point_index, projected_point in enumerate(projected_points):
            # check if point is behind the camera, if it is skip it
            if not is_not_behind_camera[projected_point_index]:
                continue

            # get the point depth in 3D
            point_depth = points[projected_point_index][2]
            
            # if the point is projected outside the visible part of the image ignore it
            if projected_point[0] < 0 or projected_point[1] < 0 or projected_point[0] > width or projected_point[1] > height:
                continue
            
            # get the indices of the neighbors
            neighbor_indices = points_tree.query_ball_point(projected_point, neighbor_distance)
            
            # get the neighbor depths
            neighbor_depths = points[neighbor_indices][:,2]
            
            # get the depth of the neighbor nearest to the camera
            lowest_depth = np.min(neighbor_depths, initial=np.inf)

            # if it can find a neighbor closer to the camera by a certain difference
            # assume that this point is not actually seen in the image
            if (lowest_depth != np.inf) and (point_depth - lowest_depth > max_depth_difference):
                continue
            
            # get color from image
            color = image_data[np.floor(projected_point[1]).astype(np.int32), np.floor(projected_point[0]).astype(np.int32)] # TODO do bilinear or trilinear interpolation

            # add color to point
            colors_per_point[projected_point_index].append(color)

    # calculate the color of the points
    colors = np.array(  [np.mean(colors, axis=0).astype(np.uint8)
                            if len(colors) > 0
                            else np.array([0, 0, 0], dtype=np.uint8) # TODO manage not colored points
                            for colors in colors_per_point])
    
    # add points to reconstruction
    print("Coloring...")
    ply = col.Reconstruction()
    dummy_track = col.Track()
    for index in range(points.shape[0]):
        ply.add_point3D(points[index], dummy_track, colors[index])

    # TODO need a way to store the index of not colored point
    # TODO to color not colored point (if they are a low % otherwise don't do it) median filter on every channel of HSV

    # write ply file
    print("Writing ply...")
    ply_out_path = "out.ply"
    ply.export_PLY(ply_out_path)

if __name__ == "__main__":
    main()