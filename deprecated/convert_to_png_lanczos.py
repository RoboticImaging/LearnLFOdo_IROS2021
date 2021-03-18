import epi_io
import os
import cv2
import shutil
import sys

for _dummy in [1]:

	sequences = sys.argv[1:]

	for sequence in sequences:
		print(sequence)
		lfs = os.listdir(sequence)
		lfs = [int(lf.split(".")[-2]) for lf in lfs if lf.split(".")[-1] == "epirect"]
		lfs = sorted(lfs)
		lfs = ["{:06d}.epirect".format(lf) for lf in lfs]
		lfs = [os.path.join(sequence, lf) for lf in lfs]

		output_directory = os.path.join("./module1-3-png", sequence.split("/")[-1])
		
		print(output_directory)
				
		for i, lf in enumerate(lfs):
			print(f"{i}/{len(lfs)}", end="\r")
			lf_img = epi_io.read_rectified(lf)
			for j in range(0, 17):
				img = lf_img[j, :, :, :] 
				output_directory_numbered = os.path.join(output_directory, str(j))
				
				if not os.path.exists(output_directory_numbered):
					os.makedirs(output_directory_numbered)
					
				output_filename = os.path.join(output_directory_numbered, "{:010d}.png".format(i))
				img = cv2.resize(img, (256, 192), interpolation=cv2.INTER_LANCZOS4)
				cv2.imwrite(output_filename, img)
				
		print("")
		shutil.copyfile(os.path.join(sequence, "poses_gt_absolute.npy"), os.path.join(output_directory, "poses_gt_absolute.npy"))
		shutil.copyfile(os.path.join(sequence, "poses_gt_relative.npy"), os.path.join(output_directory, "poses_gt_relative.npy"))	
		
	
		

	
	

