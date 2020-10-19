from dataset import *


class MetaDataset(Dataset):
    """Dataset for meta-level learning.
    Generates tasks from distribution p(tau).
    """

    def __init__(self,
                 meta_model,
                 meta_batch_size=25,
                 task_batch_size=5,
                 model_class=None):

        # metamodel
        self.meta_model = meta_model

        # size of all meta dataset
        self.meta_batch_size = meta_batch_size

        # batch size for task learning (K for K-shot learning)
        self.task_batch_size = task_batch_size

        # define architecture of model
        self.model_class = Regressor if model_class is None else model_class

    # get size of dataset
    def __len__(self):
        return self.meta_batch_size

    # get a sample of task (model, dataset, dataloaders)
    def __getitem__(self, idx):
        # make new task model and initialize it w/ metamodel weights
        task_model = self.model_class(dim_input=dim_input,
                                      dim_output=dim_output,
                                      dim_hidden=dim_hidden)
        task_model.load_state_dict(self.meta_model.state_dict())

        # make dataset and dataloaders for K-shot learning
        task_dataset = TaskDataset(self.task_batch_size)
        task_train_dl = DataLoader(task_dataset, batch_size=self.task_batch_size, shuffle=True)
        task_test_dl = DataLoader(task_dataset, batch_size=self.task_batch_size, shuffle=False)
        return task_model, task_dataset, task_train_dl, task_test_dl
