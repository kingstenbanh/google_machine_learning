import tensorflow as tf

g = tf.Graph()

with g.as_default():
    x = tf.constant(8, name="x_const")
    y = tf.constant(5, name="y_const")
    my_sum = tf.add(x, y, name="x_y_sum")

    z = tf.constant(4, name="z_const")
    new_sum = tf.add(my_sum, z, name="x_y_z_sum")

    with tf.Session() as sess:
        print new_sum.eval()

