import os
from torch.utils.tensorboard import SummaryWriter
from torch import save, rand, zeros
from config import *
from model import *
from dataset import *
from trainer import *
from utils import *


def run_training_wo_metatesting(nepoch, logname, seed=None):
    # fix random seed for reproducibility
    if seed is not None:
        set_seed(seed)

    # create folder for saving models and log
    folder = os.path.join(os.getcwd(), "experiments", logname)
    os.makedirs(folder, exist_ok=False)

    # loggers
    writer = SummaryWriter('runs/{}'.format(logname))  # writer for TensorBoard
    logger = Logger(os.path.join(folder, "{}.log".format(logname)))  # custom logger for visualizations

    # create meta-model
    meta_model = RegressorBN(dim_input=dim_input,
                             dim_output=dim_output,
                             dim_hidden=dim_hidden)
    # initialize weights
    meta_model.apply(weights_initialization)
    print(meta_model)

    # create zero gradients (initially they are None)
    for p in meta_model.parameters():
        p.grad = zeros(p.size())

    # loss function
    lf = MetaLearningMSELoss(reduction='mean', verbose=False)

    # define the optimization
    optimizer = Adam(meta_model.parameters(), lr=meta_lr)
    # optimizer = SGD(task_model.parameters(), lr=meta_lr)

    # LR scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer,
                                            T_0=nepoch+1,
                                            T_mult=1,
                                            eta_min=1e-5)

    # create meta-dataset
    meta_dataset = MetaDataset(meta_model=meta_model,
                               meta_batch_size=meta_batch_size,
                               task_batch_size=task_batch_size,
                               model_class=RegressorBN)

    # save init model
    model_path = os.path.join(folder, "epo{}.pth".format(str(0).zfill(len(str(nepoch)))))
    save(meta_model.state_dict(), model_path)

    with Profiler() as p:
        for epoch in range(nepoch + 1):
            # make one iteration of meta-learning
            # train_loss = train_meta_model_grads(meta_model,
            train_loss = train_meta_model(meta_model,
                                          meta_dataset,
                                          optimizer,
                                          loss_func=lf,
                                          verbose=False)
            # logging
            writer.add_scalar('train_loss', train_loss, epoch + 1)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)
            logger.write(",".join(map(str, [epoch + 1, train_loss])))

            # save model
            if epoch % saving_freq == 0:
                model_path = os.path.join(folder, "epo{}.pth".format(str(epoch).zfill(len(str(nepoch)))))
                save(meta_model.state_dict(), model_path)

            # update LR
            scheduler.step()

    return meta_model


def run_training_preatrain_model(nepoch, logname, seed=None):
    # fix random seed for reproducibility
    if seed is not None:
        set_seed(seed)

    # create folder for saving models and log
    folder = os.path.join(os.getcwd(), "experiments", logname)
    os.makedirs(folder, exist_ok=False)

    # loggers
    writer = SummaryWriter('runs/{}'.format(logname))  # writer for TensorBoard
    logger = Logger(os.path.join(folder, "{}.log".format(logname)))  # custom logger for visualizations

    # create meta-model
    pretrain_model = RegressorBN(dim_input=dim_input,
                                 dim_output=dim_output,
                                 dim_hidden=dim_hidden)
    # initialize weights
    pretrain_model.apply(weights_initialization)
    print(pretrain_model)

    # create zero gradients (initially they are None)
    for p in pretrain_model.parameters():
        p.grad = zeros(p.size())

    # loss function
    lf = MSELoss(reduction='mean')

    # define the optimization
    optimizer = Adam(pretrain_model.parameters(), lr=meta_lr)
    # optimizer = SGD(task_model.parameters(), lr=meta_lr)

    # LR scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer,
                                            T_0=nepoch+1,
                                            T_mult=1,
                                            eta_min=1e-5)

    # save init model
    model_path = os.path.join(folder, "epo{}.pth".format(str(0).zfill(len(str(nepoch)))))
    save(pretrain_model.state_dict(), model_path)

    with Profiler() as p:
        for epoch in range(nepoch + 1):
            # create pretrain dataset
            pretrain_dataset = TaskDataset(meta_batch_size*task_batch_size)
            pretrain_dl = DataLoader(pretrain_dataset, batch_size=task_batch_size, shuffle=True)

            # make one iteration of training of pretrain baseline model
            train_loss = train_pretrain_baseline_model(pretrain_model,
                                                       pretrain_dl,
                                                       optimizer,
                                                       loss_func=lf,
                                                       verbose=False)
            # logging
            writer.add_scalar('train_loss', train_loss, epoch + 1)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)
            logger.write(",".join(map(str, [epoch + 1, train_loss])))

            # save model
            if epoch % saving_freq == 0:
                model_path = os.path.join(folder, "epo{}.pth".format(str(epoch).zfill(len(str(nepoch)))))
                save(pretrain_model.state_dict(), model_path)

            # update LR
            scheduler.step()

    return pretrain_model
