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
	
	print(pts1)
	print(pts2)

	A = np.zeros([8, 9])

	for i in range(8):
		A[i] = np.kron(pts2[i], pts1[i]) 
	
	# print(A)

	u, s, vh = np.linalg.svd(A, full_matrices=False)

	F_mat = vh[-1].reshape(3,3)
	F_mat = np.array(F_mat)
	
	u, s, vh = np.linalg.svd(F_mat, full_matrices=False)

	F_mat = u @ np.diag([s[0], s[1], 0]) @ vh.T
	
	print(F_mat)

	for i in range(8):
		print(pts1[i] @ F_mat @ pts2[i].T)

	quit()
	return F_mat

	# 	function [ F_matrix ] = fundamental_matrix_estimation(matchCoords1, matchCoords2)

#	 for n=1:size(matchCoords1,1)
#		 A(n,:) = kron(matchCoords2(n,:), matchCoords1(n,:));
#	 end
	
#	 [U, D, V] = svd(A);
   
#	 Fa = reshape(V(:,9), 3, 3)';
	
#	 % Enforcing Rank = 2 %
#	 [Ua, Da, Va] = svd(Fa);
#	 F_matrix = Ua * diag([Da(1,1), Da(2,2), 0]) * Va';
	
# end

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

 		print(idx)
 		select1 = pts1[idx, :]
 		select2 = pts2[idx, :]

 		F_mat = get_fundamental_matrix(select1, select2)

 		# print(xa.shape, F_mat.shape, xa.T.shape)

 		err = np.sum(np.multiply(xb, (F_mat @ xa.T).T))

 		# print(F_mat)
 		# print(err)
 		# quit()
	print(F_mat)
 	
 #	% RANSAC Loop
 #	for i=1:iterations
 #		idx = randi(size(matchCoords1,1), [ptsPerItr,1]);

 #		select1 = matchCoords1(idx,:);
 #		select2 = matchCoords2(idx,:);

 #		F_mat = fundamental_matrix_estimation(select1, select2);
 #		err = sum((xb .* (F_mat * xa')'),2);

 #		currentInliers = size( find(abs(err) <= errThreshold) , 1);
 #		if (currentInliers > maxInliers)
 #		   F_matrix = F_mat; 
 #		   maxInliers = currentInliers;
 #		end

 #	end

cnt = 0
total_points = 0

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

		print(pts1.shape)
		print(pts2.shape)

		plt.subplot(1, 2, 1)
		plt.imshow(img2) 
		plt.scatter(x=pts2[:, 0], y=pts2[:, 1], c='r', s=10, marker='x', zorder=2)
		
		plt.subplot(1, 2, 2)
		plt.imshow(img1) 
		plt.scatter(x=pts1[:, 0], y=pts1[:, 1], c='r', s=10, marker='x', zorder=2)

		mng = plt.get_current_fig_manager()
		mng.full_screen_toggle()
		plt.show()

		RANSAC(pts1, pts2)
		cnt += 1
		

	print(total_points)
	img2 = img1
	pts2 = pts1

# plt.imshow(img1) 
# plt.scatter(x=pts1[:, 0], y=pts1[:, 1], c='r', s=10, marker='x', zorder=2)
# plt.show()

# plt.imshow(img2) 
# plt.scatter(x=pts2[:, 0], y=pts2[:, 1], c='r', s=10, marker='x', zorder=2)
# plt.show()