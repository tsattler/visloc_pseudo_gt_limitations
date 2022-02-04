import os
import zipfile

def mkdir(directory):
	"""Checks whether the directory exists and creates it if necessacy."""
	if not os.path.exists(directory):
		os.makedirs(directory)

# name of the folder where we download the original 12scenes dataset to
# we restructure the dataset by creating symbolic links to that folder
src_folder = '12scenes_source'

# download the original 12 scenes dataset for calibration, poses and images
mkdir(src_folder)
os.chdir(src_folder)

for ds in ['apt1', 'apt2', 'office1', 'office2']:

	print("=== Downloading 12scenes Data:", ds, "===============================")

	os.system('wget http://graphics.stanford.edu/projects/reloc/data/' + ds + '.zip')

	# unpack and delete zip file
	f = zipfile.PyZipFile(ds + '.zip')
	f.extractall()

	os.system('rm ' + ds + '.zip')