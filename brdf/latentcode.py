import torch.nn as nn
import torch
import numpy as np
class LatentCode():
    """Latent code to be optimized, as in Generative Latent Optimization.
    """
    def __init__(self, normalize=False):
        latent_code = np.load('./brdf/latent_code.npy')
        self._z = torch.from_numpy(latent_code)
        self.normalize = normalize

    @property
    def z(self):
        """The exposed interface for retrieving the current latent codes.
        """
        if self.normalize:
            return nn.functional.normalize(self._z, p=2, dim=1, eps=1e-6)

        return self._z

    @z.setter
    def z(self, value):
        self._z.data[0] = value

    def call(self, ind):
        """When you need only some slices of z.
        """
        # 0D to 1D
        if len(torch.shape(ind)) == 0: # pylint: disable=len-as-condition
            ind = torch.reshape(ind, (1,))
        # Take values by index
        z = torch.gather(self.z, ind[:, None])
        return z