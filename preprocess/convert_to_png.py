import epi_io
import os
import cv2
import shutil
import argparse


if __name__ == "__main__":
	"""
	This script converts epirect images to sets of png images.
	Usage: 
	python3 convert_to_png \
	--input-folder "/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/data/sequences" \
	--input-seq "seq16" \
	--interpolation "area" \
	--output-folder "/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/data/module-1-1/module1-2-png"
	"""

	parser = argparse.ArgumentParser(description='Convert epirect images to png using inter_area interpolation')
	parser.add_argument("--input-folder", type=str, default=None, required=True,
						help="Folder where the folder containing the epirect images is present.")
	parser.add_argument("--input-seq", type=str, default=None, required=True,
						help="Name of the folder containing the epirect images.")
	parser.add_argument("--output-folder", type=str, default=None, required=True,
						help="Folder where the output png will be stored.")
	parser.add_argument("--interpolation", type=str, default="area", choices=["area", "linear", "lanczos"],
						help="Interpolation mode to use. Choices are area, linear or lanczos. Default is area.")

	args = parser.parse_args()

	epirect_dir = os.path.join(args.input_folder, args.input_seq)
	output_dir = os.path.join(args.output_folder, args.input_seq)
	print("epirect directory: {}".format(epirect_dir))
	print("output directory: {}".format(output_dir))

	# collect all the epirect files in the folder
	lfs = os.listdir(epirect_dir)
	lfs = [int(lf.split(".")[-2]) for lf in lfs if lf.split(".")[-1] == "epirect"]
	lfs = sorted(lfs)
	lfs = ["{:06d}.epirect".format(lf) for lf in lfs]
	lfs = [os.path.join(epirect_dir, lf) for lf in lfs]

	# set downsampling scheme
	if args.interpolation == "area":
		interpolation = cv2.INTER_AREA
	elif args.interplolation == "linear":
		interpolation = cv2.INTER_LINEAR
	elif args.interpolation == "lanczos":
		interpolation = cv2.INTER_LANCZOS4
	else:
		raise ValueError("Incorrect interpolation mode chosen")

	# downsample and convert to pngs using the chosen downsampling scheme
	for i, lf in enumerate(lfs):
		print(f"{i}/{len(lfs)}", end="\r")
		lf_img = epi_io.read_rectified(lf)
		for j in range(0, 17):
			img = lf_img[j, :, :, :]
			output_directory_numbered = os.path.join(output_dir, str(j))

			if not os.path.exists(output_directory_numbered):
				os.makedirs(output_directory_numbered)

			output_filename = os.path.join(output_directory_numbered, "{:010d}.png".format(i))
			img = cv2.resize(img, (256, 192), interpolation=interpolation)
			cv2.imwrite(output_filename, img)

	if os.path.exists(os.path.join(epirect_dir, "poses_gt_absolute.npy")):
		print("Copying GT absolute poses")
		shutil.copyfile(os.path.join(epirect_dir, "poses_gt_absolute.npy"),
						os.path.join(output_dir, "poses_gt_absolute.npy"))
	else:
		print("GT absolute poses do not exist")

	if os.path.exists(os.path.join(epirect_dir, "poses_gt_relative.npy")):
		print("Copying GT relative poses")
		shutil.copyfile(os.path.join(epirect_dir, "poses_gt_relative.npy"),
						os.path.join(output_dir, "poses_gt_relative.npy"))
	else:
		print("GT relative poses do not exist")
