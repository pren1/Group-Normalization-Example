import tensorflow as tf
import numpy as np
import pdb

def GroupNorm(x, gamma, beta, G, eps=1e-5):
    # x: input features with shape [N,C,H,W]
    # gamma, beta: scale and offset, with shape [1,C,1,1]
    # G: number of groups for GN
    N, C, H, W = x.shape
    # Before reshape
    print("Input size before reshape: \n\
    Number of images: {},\n\
    Number of Channels: {},\n\
    Image Height: {},\n\
    Image Width: {}".format(N, C, H, W));
    # // represents the Integer division
    x = tf.reshape(x, [N, G, C // G, H, W])
    print("Input size after reshape: \n\
    Number of images: {},\n\
    Group Size: {}, \n\
    Number of Groups: {},\n\
    Image Height: {},\n\
    Image Width: {}".format(N, G, C // G, H, W));
    # Try to see what keep_dims does:
    tutorial_mean, tutorial_var = tf.nn.moments(x, [2, 3, 4], keep_dims=False);
    print("keep_dims_False: \n\
    mean: {},\n\
    var: {}".format(tutorial_mean, tutorial_var));
    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
    print("keep_dims_True: \n\
    mean: {},\n\
    var: {}".format(mean, var));
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [N, C, H, W])
    print("Output size after reshape: \n\
    Number of images: {},\n\
    Number of Channels: {},\n\
    Image Height: {},\n\
    Image Width: {}".format(N, C, H, W));
    return x * gamma + beta

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    # Number of images=16
    # Channels=64
    # Image width = 256
    # Image Height = 256
    shape = (16, 64, 256, 256)
    feature_array = np.random.uniform(size=shape)
    # These parameters are set randomly.
    gamma = 10;
    beta = 10;
    G = 32;
    sess = tf.Session()
    with sess.graph.as_default():
        features = tf.placeholder(tf.float32, (16, 64, 256, 256), "Input")
        res = GroupNorm(features, gamma, beta, G);
        # initialize global variables
        sess.run(tf.global_variables_initializer())
        train_op = tf.no_op()
        # run one step
        _ = sess.run([train_op],
                               feed_dict={features: feature_array})
    sess.close()
