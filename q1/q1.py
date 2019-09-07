import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
from scipy.linalg import null_space

F_mat = [[-1.29750186e-06,  8.07894025e-07,  1.84071967e-03], \
		[3.54098411e-06,  1.05620725e-06, -8.90168709e-03], \
		[-3.29878312e-03,  5.14822628e-03,  1.00000000e+00]]


F_mat = np.array(F_mat)
img1_points = np.array([[381, 402], [452, 497], [671, 538], [501, 254], [506, 381], [474, 440], [471, 537], [498, 364], [706, 319], [635, 367]])
img2_points = np.array([[390, 346], [439, 412], [651, 417], [477, 194], [482, 300], [456, 359], [454, 444], [475, 287], [686, 185], [606, 253]])

newrow = [1] * img1_points.shape[0]
newrow = np.array([newrow])
img1_points = np.concatenate((img1_points, newrow.T), axis=1)
img2_points = np.concatenate((img2_points, newrow.T), axis=1)

print('Image 1 points:')
print(img1_points)
print('Image 2 points:')
print(img2_points)

eqs1 = np.matmul(F_mat.T, img2_points.T).T
eqs2 = np.matmul(F_mat, img1_points.T).T

x = np.linspace(0, 1280, 1281)

print('Image 1 equations:')
print(eqs1)
print('Image 2 equations:')
print(eqs2)

plt.subplot(1, 2, 1)
img1 = mpimg.imread('img1.jpg')

plt.imshow(img1) 
plt.scatter(x=img1_points[:, 0], y=img1_points[:, 1], c='r', s=10, marker='x', zorder=2)

for line in eqs1:
	y = (-line[2] - line[0] * x)/line[1]
	plt.plot(x, y, c='g', linewidth=1)
	y = -(line[2] + line[0] * x)/line[1]
	plt.plot(x, y, c='g', linewidth=2, zorder=1)

plt.subplot(1, 2, 2)
img2 = mpimg.imread('img2.jpg')

plt.imshow(img2)
plt.scatter(x=img2_points[:, 0],y=img2_points[:, 1], c='r', s=10, marker='x')

for line in eqs2:
	y = (-line[2] - line[0] * x)/line[1]
	plt.plot(x, y, c='g', linewidth=1)
plt.imshow(img2) 
plt.scatter(x=img2_points[:, 0], y=img2_points[:, 1], c='r', s=10, marker='x', zorder=2)

for line in eqs2:
	y = -(line[2] + line[0] * x)/line[1]
	plt.plot(x, y, c='g', linewidth=2, zorder=1)

mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.show()

# To check whether the epipolar equation is satisfied.
# for i in range(10):
# 	print(np.matmul(img2_points[i], np.matmul(F_mat, img1_points[i].T)))

u,s,v = np.linalg.svd(F_mat, full_matrices=True)
s[2] = 0
F_mat = u@np.diag(s)@v

e1 = null_space(F_mat)
e1[0, :] /= e1[2, :]
e1[1, :] /= e1[2, :]
e1[2, :] /= e1[2, :]
print("epipole1 ", e1.T)

e2 = null_space(F_mat.T)
e2[0, :] /= e2[2, :]
e2[1, :] /= e2[2, :]
e2[2, :] /= e2[2, :]
print("epipole2 ", e2.T)

print(v.T)