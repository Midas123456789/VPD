import os
import imageio

jpg_files = [f for f in os.listdir(".") if f.endswith(".jpg")]
for jpg_file in jpg_files:
	png_file = jpg_file.replace(".jpg", ".png")
	imageio.imwrite(png_file, imageio.imread(jpg_file))