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
rows, cols = 2048, 2048 # samples, features
X, y = make_blobs(n_samples = rows, centers = 2, n_features = cols)
X_std = StandardScaler().fit_transform(X) # Optional
df = DataFrame(X_std)
df = df.astype(np.float32)


blocks = 512	# Size of kernel blocks
threads = 256	# Size of threads per block

# Call to PyCUDA Kernel, return the cov. matrix and
# GPU execution time in milliseconds
covariance_matrix, gpu_exec_time = get_cov(df.values, blocks, threads)

```

## License

[MIT](https://choosealicense.com/licenses/mit/)
