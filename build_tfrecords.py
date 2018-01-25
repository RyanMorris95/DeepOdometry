import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import glob

from transforms3d.quaternions import quat2mat, mat2quat
from lie_algebra import se3, relative_se3


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def to_transform_mtx(pose):
    trans = pose[0:3]
    quat = pose[3:7]
    rot = quat2mat(quat)
    return se3(rot, trans)


def to_quatpose_mtx(se3_mtx):
    trans = se3_mtx[:3, 3]
    rot = mat2quat(se3_mtx[:3, :3])
    pose = np.zeros(7)
    pose[0:3] = trans
    pose[3:7] = rot
    return pose


def build_college_tfrecord(directory, dataset_txt, total_length=50, type='train'):
    dataset_txt = directory + dataset_txt
    print ("Creating tf record for " + dataset_txt)
    writer = tf.python_io.TFRecordWriter(directory+type+'.tfrecord')
    count = 0
    with open(dataset_txt) as f:
        se3_prev = None
        for i, line in enumerate(f):
            if i == total_length and total_length:
                break
            if i % 500 == 0:
                print (i)

            fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
            p0 = float(p0)
            p1 = float(p1)
            p2 = float(p2)
            p3 = float(p3)
            p4 = float(p4)
            p5 = float(p5)
            p6 = float(p6)
            image = cv2.imread(directory+fname)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.imencode('.jpg', image)[1].tostring()

            pose_abs = np.array([p0, p1, p2, p3, p4, p5, p6])

            if i == 0:
                se3_prev = to_transform_mtx(pose_abs)
            else:
                se3_curr = to_transform_mtx(pose_abs)
                se3_rel = relative_se3(se3_prev, se3_curr)
                pose_rel = to_quatpose_mtx(se3_rel)

                example = tf.train.Example(features=tf.train.Features(feature={ # SequenceExample for seuqnce example
                    "pose_abs": _bytes_feature(pose_abs.tostring()),
                    "pose_rel": _bytes_feature(pose_rel.tostring()),
                    "image": _bytes_feature(image)
                }))
                writer.write(example.SerializeToString())
                se3_prev = se3_curr
                count += 1
    print ("Total Size: ", str(count))
    writer.close()

def convert_to_logq(pose):
    trans = pose[0:3]
    quat = pose[3:7]
    u, v = quat[0], quat[1:4]
    if np.linalg.norm(v) == 0:
        return np.zeros(6)
    log_q = np.dot(v / np.linalg.norm(v), np.arccos(u))

    new_pose = np.zeros(6)
    new_pose[0:3] = trans
    new_pose[3:6] = log_q

    return new_pose


import os
def build_7scenes_tfrecord(directory, sets_dict, type='train'):
    writer = tf.python_io.TFRecordWriter(directory+type+'_7scenes.tfrecord')
    count = 0
    for scene, seq_nums in sets_dict.items():
        for seq in seq_nums:
            print ("On sequence " + str(seq) + " of " + scene + " scene.")
            img_search = directory+scene+'/seq-'+'%02d'%seq+'/*color.png'
            img_files = sorted(glob.glob(img_search))
            pose_search = directory+scene+'/seq-'+'%02d'%seq+'/*.txt'
            pose_files = sorted(glob.glob(pose_search))

            se3_prev = None
            i = 0
            for img_file, pose_file in zip(img_files, pose_files):
                if count % 500 == 0:
                    print (count)

                image = cv2.imread(img_file)

                image_raw = cv2.imencode('.jpg', image)[1].tostring()

                se3_abs = np.loadtxt(pose_file)
                if i == 0:
                    se3_prev = se3_abs
                    i += 1
                else:
                    pose_abs = to_quatpose_mtx(se3_abs)

                    se3_rel = relative_se3(se3_prev, se3_abs)
                    pose_rel = to_quatpose_mtx(se3_rel)

                    pose_abs = convert_to_logq(pose_abs)
                    pose_rel = convert_to_logq(pose_rel)


                    example = tf.train.Example(
                        features=tf.train.Features(feature={  # SequenceExample for seuqnce example
                            "pose_abs": _bytes_feature(pose_abs.tostring()),
                            "pose_rel": _bytes_feature(pose_rel.tostring()),
                            "image": _bytes_feature(image_raw)
                        }))
                    writer.write(example.SerializeToString())
                    se3_prev = se3_abs
                    count += 1
                    i += 1

    print ("Total Size: ", str(count))
    writer.close()


if __name__ == '__main__':
    # build_tfrecord('/media/ryan/E4DE46CCDE4696A8/KingsCollege/', 'dataset_train_ordered.txt',
    #                total_length=None)
    # build_tfrecord('/media/ryan/E4DE46CCDE4696A8/KingsCollege/', 'dataset_test_ordered.txt',
    #                total_length=None, type='test')

    # build 7 scenes tfrecord
    dir = '/media/ryan/E4DE46CCDE4696A8/7-scenes/'
    train_sets = {'chess': [1, 2, 4, 6], 'fire': [1, 2], 'heads': [2], 'office': [1, 3, 4, 5, 8, 10],
                  'pumpkin': [2, 3, 6, 8], 'redkitchen': [1, 2, 5, 7, 8, 11, 13], 'stairs': [2, 3, 5, 6]}
    test_sets = {'chess': [3, 5], 'fire': [3, 4], 'heads': [1], 'office': [2, 6, 7, 9], 'pumpkin': [1, 7],
                 'redkitchen': [3, 4, 6, 12, 14], 'stairs': [1, 4]}
    build_7scenes_tfrecord(dir, train_sets, type='train')
    build_7scenes_tfrecord(dir, test_sets, type='test')
