import numpy as np

class ChangeChannels:

    """
    Copies the data while changing the data type to the provided one

    :param channel_to_keep: Channel to keep
    :type dtype: numpy.dtype
    :param key: The keys for reading from and writing to the state dict.
    :type key: str
    """

    def __init__(self, channel_to_keep, key="X"):
        assert isinstance(key, str)
        self.key = key
        self.channel_to_keep = channel_to_keep

    def __call__(self, state_dict):
        x, metadata = state_dict[self.key]

        x = np.expand_dims(x[self.channel_to_keep,:], axis=0)
        state_dict[self.key] = (x, metadata)

    def __str__(self):
        return f"ChangeChannels (channel_to_keep={self.channel_to_keep}, key={self.key})"
