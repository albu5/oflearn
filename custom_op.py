import numpy as np
import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.python.framework import ops


def np_mod(x, y):
  return (x * y).astype(np.float32)

def modgrad(op, grad):
  x = op.inputs[0]
  y = op.inputs[1]
  return grad * y, grad * x

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
  rnd_name = 'PyFuncGrad' + str(np.random.randint(0,1e+8))

  tf.RegisterGradient(rnd_name)(grad)
  g = tf.get_default_graph()
  with g.gradient_override_map({"PyFunc":rnd_name}):
    return tf.py_func(func, inp, Tout, stateful=True, name=name)


def tf_mod(x,y, name=None):

    with ops.name_scope(name, "mod", [x,y]) as name:
        z = py_func(np_mod,
                        [x,y],
                        [tf.float32],
                        name=name,
                        grad=modgrad)  # <-- here's the call to the gradient
        return z[0]

with tf.Session() as sess:

    x = tf.constant([0.3,0.7,1.2,1.7])
    y = tf.constant([0.2,0.5,1.0,2.9])
    z = tf_mod(x,y)
    gr = tf.gradients(z, [x,y])
    tf.global_variables_initializer().run()

    print(x.eval(), y.eval(),z.eval(), gr[0].eval(), gr[1].eval())