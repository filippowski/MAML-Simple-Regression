from trainer import *


def train_task_model(task_model,
                     task_train_dl,
                     lr=0.001,
                     verbose=False):
    # train mode
    task_model.train()

    # loss function
    lf = MSELoss(reduction='mean')

    # define the optimization
    optimizer = Adam(task_model.parameters(), lr=lr)
    # optimizer = SGD(task_model.parameters(), lr=lr)

    # losses
    losses = []

    # run minibatches of dataset (= one training epoch)
    for i, (inputs, targets) in enumerate(task_train_dl):
        # clear the gradients
        optimizer.zero_grad()
        # compute the model output
        outputs = task_model(inputs)
        # calculate train loss
        loss = lf(outputs, targets)
        # credit assignment
        loss.backward()
        # update model weights
        optimizer.step()
        # update losses list
        losses.append(loss.item())

        if verbose:
            print("inputs: {}, \npredicts: {}, \ntargets: {}, \nloss: {}"
                  .format(inputs.data, outputs.data, targets.data, loss.item()))

    return np.mean(losses)
