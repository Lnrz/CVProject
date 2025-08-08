import configparser as config
from configparser import UNNAMED_SECTION
from pathlib import Path

def change_project_paths():
    project_ini_in_path = "ins/reconstructions/myReconstruction/projectToBeChanged.ini"
    project_config = config.ConfigParser(allow_unnamed_section=True)
    project_config.read(project_ini_in_path)
    
    new_database_path = Path("ins/reconstructions/myReconstruction/database.db").absolute()
    new_image_path = Path("ins/reconstructions/myReconstruction/images").absolute()
    project_config[UNNAMED_SECTION]["database_path"] = str(new_database_path)
    project_config[UNNAMED_SECTION]["image_path"] = str(new_image_path)
    
    project_ini_out_path = "ins/reconstructions/myReconstruction/roomCornerNaturalWarmLightSparseReconstruction/project.ini"
    with open(project_ini_out_path, "w") as f:
        project_config.write(f, space_around_delimiters=False)

def main():
    change_project_paths()

if __name__ == "__main__":
    main()