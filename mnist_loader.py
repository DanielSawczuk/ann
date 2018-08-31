import numpy as np


def load_helper(images_path, labels_path):
    with open(images_path, "rb") as f:
        f.read(4)  # magic number
        num_of_images = int.from_bytes(f.read(4), byteorder="big")
        num_of_rows = int.from_bytes(f.read(4), byteorder="big")
        num_of_columns = int.from_bytes(f.read(4), byteorder="big")
        image_size = num_of_rows * num_of_columns

        data_images = np.zeros((image_size, num_of_images), dtype=np.float32)

        # reads byte by byte so byteorder is not significant
        data_in_bytes = list(f.read(image_size*num_of_images))

        for c in range(num_of_images):
            curr_index = c * image_size
            data_images[:, c] = data_in_bytes[curr_index:curr_index+image_size]

        data_images /= 255

    with open(labels_path, "rb") as f:
        f.read(4)  # magic number
        num_of_labels = int.from_bytes(f.read(4), byteorder="big")

        data_labels = np.zeros((10, num_of_labels), dtype=np.float32)

        # reads byte by byte so byteorder is not significant
        data_in_bytes = list(f.read(image_size*num_of_images))

        for c in range(num_of_labels):
            data_labels[data_in_bytes[c], c] = 1

    return data_images, data_labels


def load_mnist():
    training_data_images_path = "./train-images.idx3-ubyte"
    training_data_labels_path = "./train-labels.idx1-ubyte"

    test_data_images_path = "./t10k-images.idx3-ubyte"
    test_data_labels_path = "./t10k-labels.idx1-ubyte"

    training_images, training_labels = load_helper(training_data_images_path, training_data_labels_path)
    test_images, test_labels = load_helper(test_data_images_path, test_data_labels_path)

    training_data = (training_images, training_labels)
    test_data = (test_images, test_labels)

    return training_data, test_data