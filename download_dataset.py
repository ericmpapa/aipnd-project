# Download the dataset
# This cell has to run only once.
# NO need to run every time you arrive on this notebook.

import requests
import tarfile
import os
import shutil
import argparse

def download(data_dir):
    # Define the URL and folder paths
    url = "https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz"
    folder_name = data_dir
    file_name = "flower_data.tar.gz"
    file_path = os.path.join(folder_name, file_name)

    # Remove the folder or symbolic link if it already exists (equivalent to `rm -rf flowers`)
    try:
        if os.path.islink(folder_name) or os.path.isfile(folder_name):
            os.remove(folder_name)  # Remove the symbolic link or file
        elif os.path.isdir(folder_name):
            shutil.rmtree(folder_name)  # Remove the directory
        print(f"Removed existing {folder_name} folder/file/soft link, if any.")
    except FileNotFoundError:
        pass  # If the file or directory does not exist, do nothing

    # Create the folder
    os.makedirs(folder_name)
    print(f"Created folder: {folder_name}")

    # Download the file
    response = requests.get(url, stream=True)

    # Save the file in the 'flowers' folder
    with open(file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

    print(f"Downloaded {file_name} to {folder_name}")

    # Extract the file in the 'flowers' folder
    if file_path.endswith("tar.gz"):
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=folder_name)
            print(f"Extracted {file_name} to {folder_name}")

    # Clean up by removing the tar.gz file after extraction
    os.remove(file_path)
    print(f"Removed the downloaded tar.gz file: {file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='download', description='Download the dataset')
    parser.add_argument('--data_dir', default="", help="the dataset directory")
    args = parser.parse_args()
    data_dir = args.data_dir
    download(data_dir)