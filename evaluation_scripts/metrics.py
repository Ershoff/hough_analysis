import os
from math import degrees

import numpy as np
import csv
import argparse
import cv2

def read_from_csv(path):
    answers = dict()

    with open(path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None) # helps skip header
        for row in reader:
            name, x, y, _ = row[0].split(';')
            answers[name] = tuple((int(x), int(y)))

    return answers

def get_shapes(path):
    names = os.listdir(path)
    return {name: cv2.imread(os.path.join(path, name)).shape[0:2] for name in names if not name.endswith('.csv')}


def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def calculate_angle(ans, gt, shape):
    n,m = shape
    
    O = np.array((n/2, m/2, ((n/2)**2 + (m/2)**2)**0.5))
    A = np.array((ans[0], ans[1], 0))
    GT = np.array((gt[0], gt[1], 0))

    return angle_between(A - O, GT - O)


def calc_metrics(path_to_answers, path_to_gt, path_to_imgs):
    answers = read_from_csv(path_to_answers)
    gt = read_from_csv(path_to_gt)
    shapes = get_shapes(path_to_imgs)

    if set(answers.keys()) != set(gt.keys()) or set(answers.keys()) != set(shapes.keys()):
        raise ValueError('Names of files in gt, answers or in folder does not match!\n')

    angles = []
    for name in gt:
        angle = calculate_angle(answers[name], gt[name], shapes[name])
        angles.append(angle)

    angles = np.array(angles)

    return np.mean(angles), np.median(angles)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read two files from command line.')
    parser.add_argument('--gt', help='path to file with ground truth')
    parser.add_argument('--ans', help='path to file with answers')
    parser.add_argument('--imgs', help='path to images')

    args = parser.parse_args()

    mean, median = calc_metrics(args.ans, args.gt, args.imgs)
    print(f"Mean: {mean}, median: {median}")