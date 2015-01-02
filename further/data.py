
def list_with_one_in(size, idx_one):
	l = [0] * size
	l[idx_one] = 1
	return l

from struct import unpack
def get_mnist(filename_images, filename_labels, take=2000):

	images = []

	images_file = open(filename_images, "r")
	images_file.read(4) # header
	nb_images, nb_rows, nb_cols = unpack('>iii', images_file.read(12))
	for image in xrange(min(take, nb_images)):
		image_content = map(ord, images_file.read(nb_rows * nb_cols))
		image_content = map(float, image_content)
		images.append(image_content)
	images_file.close()

	labels = []

	labels_file = open(filename_labels, "r")
	labels_file.read(4)
	nb_labels, = unpack('>i', labels_file.read(4))

	assert nb_labels == nb_images
	labels = map(ord, labels_file.read(min(take, nb_labels)))

	return images, labels, (nb_rows, nb_cols)

