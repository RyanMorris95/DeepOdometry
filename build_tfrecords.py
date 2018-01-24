import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

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


def build_tfrecord(directory, dataset_txt, total_length=50, type='train'):
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

if __name__ == '__main__':
    build_tfrecord('/media/ryan/E4DE46CCDE4696A8/KingsCollege/', 'dataset_train_ordered.txt',
                   total_length=None)
    build_tfrecord('/media/ryan/E4DE46CCDE4696A8/KingsCollege/', 'dataset_test_ordered.txt',
                   total_length=None, type='test')
