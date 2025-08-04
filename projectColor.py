import numpy as np
import pycolmap as col
from scipy.spatial import KDTree

class Image:
    def __init__(self, path, image: col.Image):
        self.path = path
        self.width = image.camera.width
        self.height = image.camera.height
        self.__image = image
        self.__camera = image.camera
    
    def project_points(self, points):
        return self.__camera.img_from_cam(self.__image.cam_from_world() * points)


def project_colors(points, images, neighbor_distance, max_depth_difference):
    # data structure for keeping the colors of every point
    colors_per_point = [[] for i in range(points.shape[0])]
    
    # projecting colors to points
    print("Processing images...")
    for image_index, image in enumerate(images):
        image: Image
        print("    Processing " + str(image_index + 1) + "th image...")
        print("    Image is", image.path)
        
        # load image in memory
        # the array has size [height, width, 3]
        image_data = col.Bitmap.read(image.path, True).to_array()

        # project points on image
        projected_points = image.project_points(points)

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
            if (projected_point[0] < 0 or projected_point[1] < 0  or
                projected_point[0] > image.width or projected_point[1] > image.height):
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

    # TODO to color not colored point (if they are a low % otherwise don't do it) median filter on every channel of HSV

    # calculate the color of the points
    return np.array([np.mean(colors, axis=0).astype(np.uint8)
                            if len(colors) > 0
                            else np.array([0, 0, 0], dtype=np.uint8) # color not colored points black
                            for colors in colors_per_point])    


def write_ply(out_path, points, colors):
    # add points to model
    print("Coloring...")
    ply = col.Reconstruction()
    dummy_track = col.Track()
    for index in range(points.shape[0]):
        ply.add_point3D(points[index], dummy_track, colors[index])

    # write ply file
    print("Writing ply...")
    ply.export_PLY(out_path)


def main():
    print("Loading data...")

    # load reconstruction
    rec_path = "../../ColmapWorkspace/CVProject/roomCornerNaturalWarmLightSparseReconstruction"
    rec = col.Reconstruction(rec_path)

    # load model
    ply_in_path = "../../ColmapWorkspace/CVProject/roomCornerNaturalLightDenseReconstruction/fused.ply"
    ply = col.Reconstruction()
    ply.import_PLY(ply_in_path)

    # load similiarity model->rec from disk
    rec_similiarity = col.Sim3d(
        translation=np.load("local/translation.npy"),
        rotation=col.Rotation3d(np.load("local/rotation.npy")),
        scale=np.load("local/scale.npy")
    )
    
    # apply similiarity to model points
    ply.transform(rec_similiarity)

    # extract points from model
    points = np.array([point.xyz for point in ply.points3D.values()], dtype=np.float64)

    # unload model
    del ply
    
    # Filter images
    images = []
    image_filter = "Warm"
    for image in rec.images.values():
        image: col.Image
        if image_filter in image.name:
            images.append(Image("../../ColmapWorkspace/CVProject/images/" + image.name, image))
    
    colors = project_colors(points, images, 15, 1)
    write_ply("out.ply", points, colors)

if __name__ == "__main__":
    main()