![alt text](washinglearning.png "Washing Learning")
<h1 align="center">  Washing Learning : The Machine Learning Toolbox </h1>
<p align="center">

</p>

## Quickstart

This repository is not yet released on PyPI but you can install it quite simply.

```
git clone git@github.com:Lucas-rbnt/washing-learning.git
cd washing-learning/
pip install .
```

If you want to contribute to development, you can also install extras packages such as pre-commit and black to format
the code properly.

```
pip install .[dev]
```

From now on you can use this toolbox simply, you'll find a code snippet below

```python
# Standard libraries
import os
from typing import List

# Third-party libraries
from washing_learning.loggers.time_loggers import chronometer
from washing_learning.vision.preprocessing.preprocessors as preprocessors

DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets")
Dataset = ...

@chronometer
def function(path: str, preprocessors: List[preprocessors]) -> Dataset:
    data = dataloader(path)
    for preprocessor in preprocessors:
        data = preprocessor.preprocess(data)
    
    return data

def dataloader(path: str):
    raise NotImplementedError

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
