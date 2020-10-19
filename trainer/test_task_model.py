from trainer import *


def test_task_model(task_model,
                    task_test_dl,
                    verbose=False):
    # test mode
    task_model.eval()

    # loss function
    lf = MSELoss(reduction='mean')

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(task_test_dl):
            # evaluate the model on the test set
            outputs = task_model(inputs)
            # calculate test loss
            loss = lf(outputs, targets)

            if verbose:
                print("inputs: {}, \npredicts: {}, \ntargets: {}, \nloss: {}"
                      .format(inputs.data, outputs.data, targets.data, loss.item()))

    return inputs, targets, outputs, loss


def test_task_model_grads(task_model,
                          task_test_dl,
                          verbose=False):
    # test mode
    task_model.train()

    # loss function
    lf = MSELoss(reduction='mean')

    for i, (inputs, targets) in enumerate(task_test_dl):
        # evaluate the model on the test set
        outputs = task_model(inputs)
        # calculate test loss
        loss = lf(outputs, targets)

        if verbose:
            print("inputs: {}, \npredicts: {}, \ntargets: {}, \nloss: {}"
                  .format(inputs.data, outputs.data, targets.data, loss.item()))

        # calculate gradients w/o updating of task model weights
        # these gradients will be used for weights updating
        loss.backward()

    return inputs, targets, outputs, loss
