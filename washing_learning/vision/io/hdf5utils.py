# Standard libraries


import os

# Third-party libraries
import h5py
import numpy as np
from keras.utils import np_utils


class HDF5DatasetWriter:
    def __init__(
        self, dims, output_path: str, data_key="images", buf_size=1000
    ) -> None:
        if os.path.exists(output_path):
            raise ValueError(
                "The supplied outputPath already exists and cannot be overwritten. Manually delete the file \
                before continuing",
                output_path,
            )

        self.db = h5py.File(output_path, "w")
        self.data = self.db.create_dataset(data_key, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")
        self.bufSize = buf_size
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels) -> None:
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self) -> None:
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx : i] = self.buffer["data"]
        self.labels[self.idx : i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def store_class_labels(self, class_labels) -> None:
        dt = h5py.special_dtype(vlen=str)
        label_set = self.db.create_dataset(
            "label_names", (len(class_labels),), dtype=dt
        )
        label_set[:] = class_labels

    def close(self) -> None:
        if len(self.buffer["data"]) > 0:
            self.flush()

        self.db.close()


class HDF5DatasetGenerator:
    def __init__(
        self,
        db_path,
        batch_size,
        preprocessors=None,
        aug=None,
        binarize=True,
        classes=2,
    ) -> None:
        self.batchSize = batch_size
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        self.db = h5py.File(db_path)
        self.numImages = self.db["labels"].shape[0]

    def generator(self, passes=np.inf) -> None:
        epochs = 0
        while epochs < passes:
            for i in np.arange(0, self.numImages, self.batchSize):
                images = self.db["images"][i : i + self.batchSize]
                labels = self.db["labels"][i : i + self.batchSize]
                if self.binarize:
                    labels = np_utils.to_categorical(labels, self.classes)
                if self.preprocessors is not None:
                    proc_images = []
                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        proc_images.append(image)
                    images = np.array(proc_images)
                if self.aug is not None:
                    (images, labels) = next(
                        self.aug.flow(images, labels, batch_size=self.batch_size)
                    )

                yield (images, labels)

            epochs += 1

    def close(self) -> None:
        self.db.close()
