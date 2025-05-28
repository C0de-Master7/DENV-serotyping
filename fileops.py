import os
import random
import shutil


def file2list(filepath: str):
    temp_list = []
    with open(filepath, 'r') as file:
        content = file.readlines()
    for line in content:
        line = line.strip()
        temp_list.append(line)
    return temp_list


def list2file(inlist: list, filepath: str):
    if os.path.exists(filepath):
        print("Filename already exists")
        return None
    with open(filepath, 'w') as file:
        for item in inlist:
            file.write(f"{item}\n")
    print(f"Content written to {filepath} successfully")
    return None


def split_files(source_dir, num_files, target_dir1, target_dir2):
    """
    Randomly selects a given number of files from the source directory,
    copies them to target_dir1, and copies the remaining files to target_dir2.

    :param source_dir: Path to the source directory
    :param num_files: Number of files to select randomly
    :param target_dir1: Path to the first target directory
    :param target_dir2: Path to the second target directory
    """
    # Create target directories if they don't exist
    os.makedirs(target_dir1, exist_ok=True)
    os.makedirs(target_dir2, exist_ok=True)

    # Get the list of files in the source directory
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # Check if the number of files is sufficient
    if num_files > len(files):
        raise ValueError(f"Source directory contains only {len(files)} files, but {num_files} were requested.")

    # Randomly select the specified number of files
    selected_files = random.sample(files, num_files)

    # Copy selected files to target_dir1
    for file in selected_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(target_dir1, file))

    # Copy remaining files to target_dir2
    remaining_files = set(files) - set(selected_files)
    for file in remaining_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(target_dir2, file))

    print(f"Copied {len(selected_files)} files to {target_dir1}")
    print(f"Copied {len(remaining_files)} files to {target_dir2}")


def random_split(source_dir, num_files, seed=7):
    # Get the list of files in the source directory
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # Check if the number of files is sufficient
    if num_files > len(files):
        raise ValueError(f"Source directory contains only {len(files)} files, but {num_files} were requested.")

    # Setting seed value for reproducibility
    random.seed(seed)

    # Randomly select the specified number of files
    selected_files = random.sample(files, num_files)
    remaining_files = list(set(files) - set(selected_files))

    return selected_files, remaining_files


def main():
    print("This is module for file operations.")


if __name__ == "__main__":
    main()
