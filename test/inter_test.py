import tensorflow as tf
import numpy as np

h = 2
w = 2
A = 2
c = A*A*2
n = h*w*c
a = np.linspace(1, n, n, dtype=np.int)
a = np.reshape(a, [h, w, c])
p = np.split(a, c//A//A, axis=-1)
rs = []
for tt in p:
    t = np.split(tt, A, axis=-1)
    l = []
    for s in t:
        s = s.reshape(h, w * A)
        l.append(s)
    r = np.concatenate(l, -1)
    r = np.reshape(r, [h*A, w*A, 1])
    rs.append(r)

np_res = np.concatenate(rs, -1)

def tf_re(x, A):

    x = tf.convert_to_tensor(x)
    _, h, w, c = x.get_shape().as_list()
    p = tf.split(x, c//A//A, axis=-1)
    rs = []
    for tt in p:
        t = tf.split(tt, A, axis=-1)
        l = []
        for s in t:
            s = tf.reshape(s, (-1, h, w * A))
            l.append(s)
        r = tf.concat(l, -1)
        r = tf.reshape(r, [-1, h * A, w * A, 1])
        rs.append(r)
    res = tf.concat(rs, -1)
    return res

tf_res = tf_re(a, A)
with tf.Session() as sess:
    tf_res = sess.run(tf_res)

print(tf_res - np_res)