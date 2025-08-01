import numpy as np
import pycolmap as col

def main():
    rec1_path = "../../ColmapWorkspace/CVProject/roomCornerNaturalLightSparseReconstruction"
    rec2_path = "../../ColmapWorkspace/CVProject/roomCornerNaturalWarmLightSparseReconstruction"
    rec1 = col.Reconstruction(rec1_path)
    rec2 = col.Reconstruction(rec2_path)

    # get the similiarity between the two
    rec2_from_rec1 = col.align_reconstructions_via_points(rec1, rec2)

    # save the similiarity to disk
    np.save("local/translation.npy", rec2_from_rec1.translation)
    np.save("local/rotation.npy", rec2_from_rec1.rotation.quat)
    np.save("local/scale.npy", rec2_from_rec1.scale)

if __name__ == "__main__":
    main()