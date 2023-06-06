import shutil
import os


def copy_files(source_folder, destination_folder):
    # Get the list of files in the source folder
    files = os.listdir(source_folder)

    # Iterate over each file and copy it to the destination folder
    for file_name in files:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.copy2(source_path, destination_path)
        print(f"Copied {file_name} to {destination_folder}")


# Usage example
source_folder = '/home/akmal/APIIT/FYP Code/Housegan-data-reader/sample_output'
destination_folder = '/home/akmal/APIIT/FYP Code/house_diffusion/datasets/rplan'

copy_files(source_folder, destination_folder)
