from dataset import *


class TaskDataset(Dataset):
    """Dataset for task-level learning.
    """

    def __init__(self,
                 size,
                 amp_range=amp_range,
                 phase_range=phase_range,
                 input_range=input_range):
        # size of all dataset
        self.size = size

        # sample amp and phase uniformly
        self.amp = np.random.uniform(*amp_range)
        self.phase = np.random.uniform(*phase_range)

        # input range
        self.input_range = input_range

    # get size of dataset
    def __len__(self):
        return self.size

    # get a sample
    def __getitem__(self, idx):
        # sample uniformly input and calculate output
        x = np.random.uniform(*self.input_range, [1])
        y = self.amp * np.sin(x - self.phase)

        # cast x and y to numpy array of float32
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # print("amp: {}, phase: {}, x: {}, y: {}".format(self.amp, self.phase, x, y))

        return x, y
