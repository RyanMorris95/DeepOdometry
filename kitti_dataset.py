import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import numpy as np
import scipy.linalg as sl
from transforms3d.euler import mat2euler
from transforms3d.quaternions import mat2quat

HEIGHT = 384
WIDTH = 1280
DEPTH = 1
LABELS_MEAN = [ 1.68654529e-01, -3.36497239e-01,  1.85786224e+01,  5.75098885e-04,
  2.16883295e-02,  1.72457594e-03] 
LABELS_STD = [2.66941123,  0.51481087, 14.32043476,  0.01867875,  0.32007611,  0.0254444]


def se3(r=np.eye(3), t=np.array([0, 0, 0])):
    """
    :param r: SO(3) rotation matrix
    :param t: 3x1 translation vector
    :return: SE(3) transformation matrix
    """
    se3 = np.eye(4)
    se3[:3, :3] = r
    se3[:3, 3] = t
    return se3


def se3_inverse(p):
    """
    :param p: absolute SE(3) pose
    :return: the inverted pose
    """
    r_inv = p[:3, :3].transpose()
    t_inv = -r_inv.dot(p[:3, 3])
    return se3(r_inv, t_inv)


def abs_se3_to_rel_eul(p1, p2):
    """
    :param p1: absolute SE(3) pose at timestamp 1
    :param p2: absolute SE(3) pose at timestamp 2
    :return: the relative pose p1^{⁻1} * p2
    """
    p1 = np.reshape(p1, (4, 4))
    p2 = np.reshape(p2, (4, 4))
    rel_se3 = np.dot(se3_inverse(p1), p2)
    rel_trans = rel_se3[:3, 3]
    ax, ay, az = mat2euler(rel_se3)
    #quat = mat2quat(rel_se3[:3,:3])
    rel_eul = np.zeros(6)
    rel_eul[0:3] = rel_trans
    rel_eul[3:6] = np.array([ax, ay, az])
    return rel_eul


def tf_euler2mat(z, y, x):
    """Converts euler angles to rotation matrix
    TODO: remove the dimension for 'N' (deprecated for converting all source
         poses altogether)
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
      z: rotation angle along z axis (in radians) -- size = [B, N]
      y: rotation angle along y axis (in radians) -- size = [B, N]
      x: rotation angle along x axis (in radians) -- size = [B, N]
    Returns:
      Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
    """
    B = tf.shape(z)[0]
    N = 1
    z = tf.clip_by_value(z, -np.pi, np.pi)
    y = tf.clip_by_value(y, -np.pi, np.pi)
    x = tf.clip_by_value(x, -np.pi, np.pi)

    # Expand to B x N x 1 x 1
    z = tf.expand_dims(tf.expand_dims(z, -1), -1)
    y = tf.expand_dims(tf.expand_dims(y, -1), -1)
    x = tf.expand_dims(tf.expand_dims(x, -1), -1)

    zeros = tf.zeros([B, N, 1, 1], dtype=tf.float64)
    ones  = tf.ones([B, N, 1, 1], dtype=tf.float64)

    cosz = tf.cos(z)
    sinz = tf.sin(z)
    rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
    rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
    rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
    zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

    cosy = tf.cos(y)
    siny = tf.sin(y)
    roty_1 = tf.concat([cosy, zeros, siny], axis=3)
    roty_2 = tf.concat([zeros, ones, zeros], axis=3)
    roty_3 = tf.concat([-siny,zeros, cosy], axis=3)
    ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

    cosx = tf.cos(x)
    sinx = tf.sin(x)
    rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
    rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
    rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
    xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

    rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
    return rotMat

def tf_pose_vec2mat(vec, seq_len):
    """Converts 6DoF parameters to transformation matrix
    Args:
      vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
      A transformation matrix -- [B, 4, 4]
    """
    batch_size = seq_len
    translation = tf.slice(vec, [0, 0], [-1, 3])
    translation = tf.expand_dims(translation, -1)
    rx = tf.slice(vec, [0, 3], [-1, 1])
    ry = tf.slice(vec, [0, 4], [-1, 1])
    rz = tf.slice(vec, [0, 5], [-1, 1])
    rot_mat = tf_euler2mat(rz, ry, rx)
    rot_mat = tf.squeeze(rot_mat, axis=[1])
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4], dtype=tf.float64)
    filler = tf.tile(filler, [batch_size, 1, 1])
    transform_mat = tf.concat([rot_mat, translation], axis=2)
    transform_mat = tf.concat([transform_mat, filler], axis=1)
    return transform_mat


from transforms3d.euler import euler2mat, mat2euler
def compose_poses(rel_poses):
    abs_poses = np.zeros_like(rel_poses)
    abs_pose = np.identity(4)
    for i in range(rel_poses.shape[1]):
        rel_pose = rel_poses[0,i,:]
        trans, eul = rel_pose[0:3], rel_pose[3:6]
        rot = euler2mat(eul[0], eul[1], eul[2])
        rel_pose = se3(rot, trans)
        abs_pose = np.dot(abs_pose, rel_pose)
        abs_trans = abs_pose[:3,3]
        ax, ay, az = mat2euler(abs_pose)

        abs_poses[0,i,:] = np.array([trans[0], trans[1], trans[2],
            ax, ay, az])

    return abs_poses.astype('float64')


def eul_less(r11, r12, r13, r23, r33, cy):
    z = tf.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
    y = tf.atan2(r13,  cy) # atan2(sin(y), cy)
    x = tf.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    return tf.concat([x, y, z], axis=1)

# def eul_greater(r13, r21, r22, cy, seq_len):
#     # so r21 -> sin(z), r22 -> cos(z) and
#     z = tf.atan2(r21,  r22)
#     y = tf.atan2(r13,  cy) # atan2(sin(y), cy)
#     x = tf.zeros((seq_len, 1, 1), dtype=tf.float64)
#     print (x, y, z)
#     return tf.concat([x, y, z], axis=1)

def tf_mat2euler(M, seq_len):
    print (M)
    M = tf.reshape(M, (seq_len, 9, 1))
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = tf.split(M, 9, axis=1)
    cy = tf.sqrt(r33*r33 + r23*r23)
    #return tf.cond(cy < 1e-4, lambda: eul_less(r11, r12, r13, r23, r33, cy), lambda: eul_greater(r13, r21, r22, cy))
    return eul_less(r11, r12, r13, r23, r33, cy)


def compose_poses_tf(rel_poses, seq_len):
    abs_pose = tf.eye(4, dtype=tf.float64)
    rel_poses = tf_pose_vec2mat(rel_poses, seq_len)
    rel_poses = tf.reshape(rel_poses, (seq_len, 4, 4))
    for i in range(seq_len):
        rel_pose = tf.gather(rel_poses, i)
        print (rel_pose)
        abs_pose = tf.reshape(abs_pose, (4, 4))
        abs_pose = tf.matmul(rel_pose, abs_pose)
        abs_pose = tf.reshape(abs_pose, (1, 4, 4))
        if i == 0:
            abs_poses = abs_pose
        else:
            abs_poses = tf.concat([abs_poses, abs_pose], axis=0)
            
    trans = tf.slice(abs_poses, (0, 0, 3), (seq_len, 3, 1))
    trans = tf.reshape(trans, (seq_len, 3, 1))
    rot = tf.slice(abs_poses, (0, 0, 0), (seq_len, 3, 3))
    eul = tf_mat2euler(rot, seq_len)
    new_poses = tf.concat([trans, eul], axis=1)
    return new_poses


class KittiDataset(object):
    def __init__(self, data_dir, target_height=HEIGHT, target_width=WIDTH, mean=0.5, std=0.5,
                 subset='train', use_distortion=True):
        self.data_dir = data_dir
        self.subset = subset
        self.use_distortion = use_distortion
        self.target_height = target_height
        self.target_width = target_width
        self.mean = mean
        self.std = std

    def get_filenames(self):
        if self.subset in ['train', 'validation', 'test']:
            return [os.path.join(self.data_dir, self.subset + '.tfrecord')]
        else:
            raise ValueError('Invalid data subset "%s"' % self.subset)

    def parser(self, record):
        keys_to_features = {
            "SE3": tf.FixedLenFeature((), tf.string, default_value=""),
            "img_raw": tf.FixedLenFeature((), tf.string, default_value=""),
            "img_raw_prev": tf.FixedLenFeature((), tf.string, default_value=""),
        }

        parsed = tf.parse_single_example(record, keys_to_features)

        image = tf.image.decode_jpeg(parsed["img_raw"])
        image = tf.reshape(image, [HEIGHT, WIDTH, DEPTH])
        image = tf.image.convert_image_dtype(image, tf.float32)  # IMAGE BETWEEN 1 and 0

        image_prev = tf.image.decode_jpeg(parsed["img_raw_prev"])
        image_prev = tf.reshape(image_prev, [HEIGHT, WIDTH, DEPTH])
        image_prev = tf.image.convert_image_dtype(image_prev, tf.float32)  # IMAGE BETWEEN 1 and 0

        label_abs = tf.decode_raw(parsed["SE3"], tf.float64)
        label_abs = tf.reshape(label_abs, [4, 4])

        return image, image_prev, label_abs

    def make_batch(self, batch_size, sequence_length=2):
        filenames = self.get_filenames()
        print (filenames)

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.parser, num_parallel_calls=sequence_length)
        dataset = dataset.repeat()
        dataset = dataset.batch(sequence_length)
        iterator = dataset.make_one_shot_iterator()
        image_batch, image_prev_batch, labels_abs = iterator.get_next()
        image_batch = self.preprocess(image_batch)
        image_prev_batch = self.preprocess(image_prev_batch)
        image_concat_seq = tf.concat([image_batch, image_prev_batch], axis=3)

        with tf.variable_scope('convert_labels'):
            start_abs = tf.gather(labels_abs, 0)
            for i in range(sequence_length): 
                curr_abs = tf.gather(labels_abs, i)
                if i == 0:
                    labels_rel = tf.py_func(abs_se3_to_rel_eul, [start_abs, curr_abs], [tf.float64])
                else:
                    label = tf.py_func(abs_se3_to_rel_eul, [start_abs, curr_abs], [tf.float64])
                    labels_rel = tf.concat([labels_rel, label], axis=0)

        # image_concat_seq = None
        # labels_seq = None

        # image_concat_seq = tf.concat([image_batch, image_prev_batch], axis=3)
        # labels_seq = labels_abs
        ### DON'T REALLY NEED
        # for i in range(sequence_length+1):
        #     if i == 0:
        #         image_batch, image_prev_batch, labels_abs = iterator.get_next()
        #         image_batch = self.preprocess(image_batch)
        #         image_prev_batch = self.preprocess(image_prev_batch)
        #         ref_labels = labels_abs
        #         image_concat_seq = tf.concat([image_batch, image_prev_batch], axis=3)
        #     elif i == 1:
        #         image_batch, image_prev_batch, labels_abs = iterator.get_next()
        #         labels_rel = tf.py_func(abs_se3_to_rel_eul, [ref_labels, labels_abs], [tf.float64])
        #     else:
        #         image_batch, image_prev_batch, labels_abs = iterator.get_next()
        #         labels_rel = tf.py_func(abs_se3_to_rel_eul, [ref_labels, labels_abs], [tf.float64])
        #
        #         image_batch = self.preprocess(image_batch)
        #         image_prev_batch = self.preprocess(image_prev_batch)
        #
        #         image_batch = tf.concat([image_batch, image_prev_batch], axis=3)
        #         image_concat_seq = tf.concat([image_concat_seq, image_batch], axis=0)

        labels_rel = tf.divide(tf.subtract(labels_rel, LABELS_MEAN), LABELS_STD)
        #label_rel = tf.subtract(LABESL_MEAN)
        return image_concat_seq, labels_rel

    def _image_augmentation(self, image, seed=42):
        # Perform additional preprocessing on the parsed data.
        # hue_delta = tf.random_uniform([1], minval=0.9, maxval=1.1, seed=seed)
        # sat_delta = tf.random_uniform([1], minval=0.9, maxval=1.1, seed=seed)
        # gam_delta = tf.random_uniform([1], minval=0.9, maxval=1.1, seed=seed)
        #

        # image = tf.image.random_brightness(image, max_delta=32. / 255.)
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        # image = tf.image.random_hue(image, max_delta=0.2)
        # image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        # image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    def preprocess(self, image):
        if self.subset in 'train':
            image = self._image_augmentation(image, seed=42)

        image = tf.subtract(image, self.mean)
        image = tf.image.resize_area(image, (self.target_height, self.target_width))

        return image


if __name__ == '__main__':
    import cv2

    MEAN = [0, 0, 0]
    STD = [0, 0, 0]

    scenes_dataset = KittiDataset('/media/ryan/E4DE46CCDE4696A8/kitti_tfrecords/', subset='train')
    image_batch, image_labels = scenes_dataset.make_batch(1, sequence_length=20)
    means = []
    stds = []
    # from autograd import grad
    # import tangent
    # with tf.Session() as sess:
    #     for i in range(18500):
    #         images, labels = sess.run([image_batch, image_labels])
    #         print (images.shape)
    #         img = images[0,:,:,0].reshape((images[0].shape[0], images[0].shape[1])).copy()

    #         import _pickle as p
    #         labels = np.array(labels)
            
    #         labels = np.reshape(labels, (1, labels.shape[0], labels.shape[1]))
    #         print (labels)
    #         print (labels.shape)
    #         abs_poses = compose_poses(labels.copy())
    #         print (abs_poses.shape)
    #         print ("Abs Poses")
    #         print (abs_poses[0,-1,:])
    #         print ("tf Poses")

    #         plt.figure(1)
    #         plt.imshow(img)
    #         plt.title('Last image')
    #         plt.figure(2)
    #         img = images[-1,:,:,0].reshape((images[0].shape[0], images[0].shape[1])).copy()
    #         plt.imshow(img)
    #         plt.title('First image')
    #         plt.show()

    ### FOR FINDING THE MEAN AND STD OF TRAINING SET
    labels_list = []
    with tf.Session() as sess:
        for i in range(int(15960/20)):
            if i % 100 == 0:
                print (i)
            images, labels = sess.run([image_batch, image_labels])
            img = images[0,:,:,0].reshape((images[0].shape[0], images[0].shape[1], 1)).copy()
            mean = img.reshape(-1, img.shape[-1]).mean(0)
            _std = img.reshape(-1, img.shape[-1]).std(0)
            labels_list.append(labels[-1].tolist())
    
            means.append(mean.copy())
            stds.append(_std.copy())
    
    labels_arr = np.array(labels_list)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(labels_arr)
    print (scaler.mean_, scaler.scale_, scaler.var_)
    
    means = np.array(means)
    stds = np.array(stds)
    
    import pickle as p
    with open('tmp.p', 'wb') as fp:
        p.dump([means, stds], fp)
    
    final_mean = means.reshape(-1, means.shape[-1]).mean(0)
    final_stds = stds.reshape(-1, stds.shape[-1]).mean(0)
    print (final_mean, final_stds)
