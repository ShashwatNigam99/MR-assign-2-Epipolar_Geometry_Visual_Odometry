import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
import cv2
import random


def get_features_flow(img1, img2, pts1):
	pts2, status, _ = cv2.calcOpticalFlowPyrLK(img1,img2,pts1, None)
	status = status.reshape(status.shape[0])
	pts1 = pts1[status == 1]
	pts2 = pts2[status == 1]

	return pts1, pts2

def get_features_sift(img1):
	sift = cv2.xfeatures2d.SIFT_create()
	key_points = sift.detect(img1)
	pts1 = np.array([x.pt for x in key_points],dtype=np.float32)

	return pts1

def get_fundamental_matrix(pts1, pts2):
	
	# print(pts1)
	# print(pts2)

	A = np.zeros([8, 9])

	for i in range(8):
		A[i] = np.kron(pts2[i], pts1[i]) 
	
	# print(A)

	u, s, vh = np.linalg.svd(A, full_matrices=False)

	F_mat = vh[-1].reshape(3,3)
	F_mat = np.array(F_mat)
	
	u, s, vh = np.linalg.svd(F_mat, full_matrices=False)

	F_mat = u @ np.diag([s[0], s[1], 0]) @ vh
	F = cv2.findFundamentalMat(pts1, pts2)

	# print(F[0])
	# print(F_mat)

	# quit()

	return F[0]

def RANSAC(pts1, pts2):


	ptsPerItr = 8
	iterations = 1000
	maxInliers = 0
	errThreshold = 0.05


	F_matrix = np.zeros([3,3])

	newrow = [1] * pts1.shape[0]
	newrow = np.array([newrow])
	pts1 = np.concatenate((pts1, newrow.T), axis=1)
	pts2 = np.concatenate((pts2, newrow.T), axis=1)

	xa = pts1
	xb = pts2

	total_points = pts1.shape[0]

	for i in range(iterations):
 		idx = random.sample(range(0, total_points), ptsPerItr)

 		# for i in range(8):
 		# 	print(pts1[i])
 		# 	print(pts2[i])

 		select1 = pts1[idx]
 		select2 = pts2[idx]

 		F_mat = get_fundamental_matrix(select1, select2)

 		err = np.sum(np.multiply(xb, (F_mat @ xa.T).T), axis = 1)
 		# print(err)

 		currentInliers = 0
 		for i in range(len(xa)):
 			if abs(err[i]) < errThreshold:
 				currentInliers += 1

 		# print(currentInliers)
 		if currentInliers > maxInliers:
 		   F_matrix = F_mat 
 		   maxInliers = currentInliers

	# print(F_matrix)
	print(maxInliers)
	return F_matrix

def get_E_mat(F_mat, K_mat):
	return K_mat.T @ F_mat @ K_mat

cnt = 0
total_points = 0
K_mat = np.array([[7.215377000000e+02,0.000000000000e+00,6.095593000000e+02], \
			  [0.000000000000e+00,7.215377000000e+02,1.728540000000e+02], \
			  [0.000000000000e+00,0.000000000000e+00,1.000000000000e+00]])

f = open('output.txt', 'w')

while cnt < 801:
	s1 = str(cnt).zfill(6)
	img1 = cv2.imread('./images/' + s1 + '.png')
	
	if total_points < 2500:
		pts1 = get_features_sift(img1)
		# print("sift: ", cnt)
		total_points = 2500

		if cnt == 0:
			cnt = 1
	else:
		pts1, pts2 = get_features_flow(img1, img2, pts2) 
		total_points = len(pts1)
		print("flow: ", cnt)

		# print(pts1.shape)
		# print(pts2.shape)

		# plt.subplot(1, 2, 1)
		# plt.imshow(img2) 
		# plt.scatter(x=pts2[:, 0], y=pts2[:, 1], c='r', s=10, marker='x', zorder=2)
		
		# plt.subplot(1, 2, 2)
		# plt.imshow(img1) 
		# plt.scatter(x=pts1[:, 0], y=pts1[:, 1], c='r', s=10, marker='x', zorder=2)

		# mng = plt.get_current_fig_manager()
		# mng.full_screen_toggle()
		# plt.show()

		F_mat = RANSAC(pts1, pts2)

		# E, _ = cv2.findEssentialMat(pts1, pts2)
		# print(E)
		E_mat = get_E_mat(F_mat, K_mat)
		print(F_mat)
		# print(E_mat)

		points, R, t, mask = cv2.recoverPose(E_mat, pts1, pts2)

		# print(R)
		# print(t) 

		T = np.concatenate((R, t), axis=1)
		T = T.reshape(1, 12)
		# print(T)

		
		for i in range(len(T[0])):
			item = T[0][i]
			if i != 11:
				f.write("%s " % item)
			else:
				f.write("%s" % item)
		f.write("\n")

		cnt += 1
		

	# print(total_points)
	img2 = img1
	pts2 = pts1

# plt.imshow(img1) 
# plt.scatter(x=pts1[:, 0], y=pts1[:, 1], c='r', s=10, marker='x', zorder=2)
# plt.show()

# plt.imshow(img2) 
# plt.scatter(x=pts2[:, 0], y=pts2[:, 1], c='r', s=10, marker='x', zorder=2)
# plt.show()