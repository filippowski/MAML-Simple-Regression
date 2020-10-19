from trainer import *


def train_pretrain_baseline_model(pretrain_model,
                                  pretrain_dl,
                                  optimizer,
                                  loss_func,
                                  verbose=False):
    # train mode
    pretrain_model.train()

    # losses
    losses = []

    # run minibatches of dataset (= one training epoch)
    for i, (inputs, targets) in enumerate(pretrain_dl):
        # clear the gradients
        optimizer.zero_grad()
        # compute the model output
        outputs = pretrain_model(inputs)
        # calculate train loss
        loss = loss_func(outputs, targets)
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
