# PyCUDACov - A PyCuda Covariance Matrix Parallel Implementation

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)

## Usage and Installation

Requires CUDA enviroment.

### Installation:

```sh
$ pip install pycudacov
```

### Basic Usage

```python
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
import numpy as np
from pycudacov import get_cov

# Generate test dataset
rows, cols = 16384, 1024 # samples, features
X, y = make_blobs(n_samples = rows, centers = 2, n_features = cols)
X_std = StandardScaler().fit_transform(X) # Optional
df = DataFrame(X_std)
df = df.astype(np.float32)

# Call to PyCUDA Kernel
covariance_matrix = get_cov(df.values)

```

## Limitations

-The maximum number of _features_ or columns of the data matrix is up to 1024

## License

[MIT](https://choosealicense.com/licenses/mit/)
