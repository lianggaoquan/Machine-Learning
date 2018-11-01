import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

'''
SVM tensorflow implementation

'''

iris = datasets.load_iris()
x_vals = np.array([[x[0],x[3]] for x in iris.data])
y_vals = np.array([1 if y == 0 else -1 for y in iris.target])

train_indices = np.random.choice(len(x_vals),int(len(x_vals)*0.8),replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

batch_size = 64

x_data = tf.placeholder(shape=[None,2],dtype=tf.float32)
y_target = tf.placeholder(shape=[None,1],dtype=tf.float32)

W = tf.Variable(tf.random_normal(shape=[2,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# y = Wx + b
model_output = tf.add(tf.matmul(x_data,W),b)

l2_norm = tf.reduce_sum(tf.square(W))

# hinge_loss = max(0,1-y*pred) + 0.5*l2_norm
loss = tf.reduce_mean(tf.maximum(0.0,1-tf.multiply(y_target,model_output)))

loss += 0.5*l2_norm

train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    loss_all = []

    for i in range(10000):
        rand_index = np.random.choice(len(x_vals_train),size=batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])
        per_loss, _ = sess.run([loss,train_step],feed_dict={x_data:rand_x,y_target:rand_y})

        if i % 10 == 0:
            loss_all.append(per_loss)

    W_star = sess.run(W)
    b_star = sess.run(b)
    sess.close()

print('W = ',W_star)
print('b = ',b_star)

plt.plot(loss_all)
plt.show()