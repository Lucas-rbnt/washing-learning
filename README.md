<h1 align="center">mlutils : The Machine Learning Toolbox </h1>
<p align="center">

</p>

## Quickstart

This repository is not yet released on PyPI but you can install it quite simply.

```
git clone git@github.com:Lucas-rbnt/mlutils.git
cd mlutils/
pip install .
```

If you want to contribute to development, you can also install extras packages such as pre-commit and black to format 
the code properly.

```
pip install .[dev]
```

From now on you can use this toolbox simply, you'll find a code snippet below

```python
from mlutils.vision.datasets.dataloaders import SimpleDatasetLoader
from mlutils.vision.preprocessing.preprocessors import SimpleScaler, SimplePreprocessor, ImageToArrayPreprocessor
import os

DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets")

sp = SimplePreprocessor(128, 128)
iap = ImageToArrayPreprocessor()
sc = SimpleScaler()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap, sc])

X, y = sdl.load(DATASET_PATH, grayscale=True, verbose=10000)
```

## The aim

The aim is to develop a Machine Learning Toolbox to preserve a consequent amount of time and energy.
There is still lots of things to do such as include typing everywhere, add some PyTorch compatibility methods,
generate the docs ...
Feel free to contact me to contribute.

## Documentation

The full package documentation is available [not done](https://localhost:80) for detailed specifications. The documentation was built with [Sphinx](https://www.sphinx-doc.org) using a [theme](https://github.com/readthedocs/sphinx_rtd_theme) provided by [Read the Docs](https://readthedocs.org).

## Contributing 

Please refer to the [`CONTRIBUTING`](./CONTRIBUTING.md) guide if you wish to contribute to this project.

## Credits

This project is developed and maintained by volunteers.