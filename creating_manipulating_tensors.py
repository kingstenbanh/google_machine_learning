import tensor as tf

# Vector Addition
with tf.Graph().as_default():
    primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)
    ones = tf.ones([6], dtype=tf.int32)

    just_beyond_primes = tf.add(primes, ones)

    with tf.Session() as sess:
        print just_beyond_primes.eval()


# Tensor Shapes
with tf.Graph().as_default():
    scalar = tf.zeros([])

    vector = tf.zeros([3])

    matrix = tf.zeros([2, 3])

    with tf.Session() as sess:
        print 'scalar has shape', scalar.get_shape(), 'and value:\n', scalar.eval()
        print 'vector has shape', vector.get_shape(), 'and value:\n', vector.eval()
        print 'matrix has shape', matrix.get_shape(), 'and value:\n', matrix.eval()


# Broadcasting
with tf.Graph().as_default():
    primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)

    ones = tf.constant(1, dtype=tf.int32)

    just_beyond_primes = tf.add(primes, ones)

    with tf.Session() as sess:
        print just_beyond_primes.eval()


# Matrix Multiplication
with tf.Graph().as_default():
    x = tf.constant([5, 2, 4, 3], [5, 1, 6, -2], [-1, 3, -1, -2], dtype=tf.int32)

    y = tf.constant([2, 2], [3, 5], [4, 5], [1, 6], dtype=tf.int32)

    matrix_multiply_result = tf.matmul(x, y)

    with tf.Session() as sess:
        print matrix_multiply_result.eval();


# Tensor Reshaping
with tf.Graph().as_default():
    matrix = tf.constant([
        [1, 2], [3, 4], [5, 6], [7, 8],
        [9, 10], [11, 12], [13, 14], [15, 16]
    ], dtype=tf.int32)

    reshaped_2x8_matrix = tf.reshape(matrix, [2, 8])

    reshaped_4x4_matrix = tf.reshape(matrix, [4, 4])

    reshaped_2x2x4_tensor = tf.reshape(matrix, [2, 2, 4])

    one_dimensional_vector = tf.reshape(matrix, [16])

    with tf.Session() as sess:
        print 'Original matrix (8x2):'
        print matrix.eval()
        print 'Reshaped matrix (2x8):'
        print reshaped_2x8_matrix.eval()
        print 'Reshaped matrix (4x4):'
        print reshaped_4x4_matrix.eval()
        print 'Reshaped 3-D tensor (2x2x4):'
        print reshaped_2x2x4_tensor.eval()
        print '1-D vector:'
        print one_dimensional_vector.eval()

# Exercise #1: Reshape two tensors in order to multiply them
with tf.Graph().as_default(), tf.Session() as sess:
    a = tf.constant([5, 3, 2, 7, 1, 4])
    b = tf.constant([4, 6, 3])

    reshaped_a = tf.reshape(a, [2, 3])
    reshaped_b = tf.reshape(b, [3, 1])

    c = tf.matmul(reshaped_a, reshaped_b)
    print(c.eval())


# Variables, Initialization and Assignment
g = tf.Graph()
with g.as_default():
    v = tf.Variable([3])
    w = tf.Variable(tf.random_normal([1], mean=1.0, stddev=0.35))


with g.as_default():
    with tf.Session() as sess:
        initialization = tf.global_variables_initializer()
        sess.run(initialization)
        print v.eval()
        print w.eval()


with g.as_default():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print v.eval()

        assignment = tf.assign(v, [7])
        print v.eval()

        sess.run(assignment)
        print v.eval()


# Exercise #2: Simulate 10 rolls of two dice
with tf.Graph().as_default():
    with tf.Session() as sess:
        dice1 = tf.Variable(tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32))
        dice2 = tf.Variable(tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32))
        sum = tf.add(dice1, dice2)

        resulting_matrix = tf.concat(values=[a, b], axis=1)

        sess.run(tf.global_variables_initializer())

        print(resulting_matrix.eval())

        




