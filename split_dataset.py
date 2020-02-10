import os
import argparse
from sklearn.model_selection import train_test_split
import shutil
def main(args):
    path = args.path
    output_path = args.output
    paths = []
    for folder in os.listdir(path):
        for img_file in os.listdir(os.path.join(path, folder)):
            paths.append((folder, img_file))
    train_paths, valid_paths = train_test_split(paths, test_size=0.15)
    if not  os.path.exists(output_path):
        os.mkdir(output_path)

    for folder, img_File in valid_paths:
        shutil.move(os.path.join(path, folder, img_File), os.path.join(output_path, folder, img_File))


    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()
    main(args)