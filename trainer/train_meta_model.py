from trainer import *


def train_meta_model(meta_model,
                     meta_dataset,
                     optimizer,
                     loss_func,
                     verbose=False,
                     seed=None):

    # fix random seed if performance evaluation by fine-tuning
    if seed is not None:
        set_seed(seed)

    # train mode
    meta_model.train()

    # train losses
    losses = []

    # run meta-learning on meta-dataset (= one meta-training epoch)
    for i in range(meta_batch_size):

        # generate model, dataset and dataloaders for a task
        task_model, task_dataset, task_train_dl, task_test_dl = meta_dataset[i]

        if verbose:
            print("iter: {}, task_model weights: {}, task_model grads: {} \n"
                  "iter: {}, meta_model weights: {}, meta_model grads: {}"
                  .format(i,
                          task_model.layers[0][0].weight.data.numpy().ravel().tolist(),
                          None if task_model.layers[0][0].weight.grad is None else
                          task_model.layers[0][0].weight.grad.data.numpy().ravel().tolist(),
                          i,
                          meta_model.layers[0][0].weight.data.numpy().ravel().tolist(),
                          None if meta_model.layers[0][0].weight.grad is None else
                          meta_model.layers[0][0].weight.grad.data.numpy().ravel().tolist(),
                          ))

        # train task model using K samples in K-shot learning setting
        train_task_model(task_model,
                         task_train_dl,
                         lr=task_lr,
                         verbose=False)

        # test task model using K samples and get test loss for metatraining training update
        task_inputs, task_targets, task_outputs, task_loss = test_task_model(task_model,
                                                                             task_test_dl,
                                                                             verbose=False)

        # compute the meta model output
        meta_outputs = meta_model(task_inputs)

        # calculate meta loss and check that it is calculated correctly
        loss = loss_func(meta_outputs, task_targets, task_outputs)
        assert loss.item() == task_loss.item(), "Meta learning MSE loss calculated incorrectly."

        # save loss
        losses.append(loss.item())

        # calculate and accumulate gradients
        loss.backward()

        if verbose:
            print("inputs: {}, \npredicts: {}, \ntargets: {}, \nloss: {}"
                  .format(task_inputs.data, meta_outputs.data, task_targets.data, loss.item()))

    # clip gradients to [-10...10]
    # clip_grad_value_(meta_model.parameters(), clip_value=10)
    # update model weights by accumulated gradients
    optimizer.step()
    # clear the gradients
    optimizer.zero_grad()

    return np.mean(losses)


def train_meta_model_grads(meta_model,
                           meta_dataset,
                           lr=0.001,
                           verbose=False,
                           seed=None):

    # fix random seed if performance evaluation by fine-tuning
    if seed is not None:
        set_seed(seed)

    # train mode
    meta_model.train()

    # define the optimization
    optimizer = Adam(meta_model.parameters(), lr=lr)

    # train losses
    losses = []

    # run meta-learning on meta-dataset (= one meta-training epoch)
    for i in range(meta_batch_size):

        # generate model, dataset and dataloaders for a task
        task_model, task_dataset, task_train_dl, task_test_dl = meta_dataset[i]

        if verbose:
            print("iter: {}, task_model weights: {}, task_model grads: {} \n"
                  "iter: {}, meta_model weights: {}, meta_model grads: {}"
                  .format(i,
                          task_model.layers[0][0].weight.data.numpy().ravel().tolist(),
                          None if task_model.layers[0][0].weight.grad is None else
                          task_model.layers[0][0].weight.grad.data.numpy().ravel().tolist(),
                          i,
                          meta_model.layers[0][0].weight.data.numpy().ravel().tolist(),
                          None if meta_model.layers[0][0].weight.grad is None else
                          meta_model.layers[0][0].weight.grad.data.numpy().ravel().tolist(),
                          ))

        # train task model using K samples in K-shot learning setting
        train_task_model(task_model,
                         task_train_dl,
                         lr=task_lr,
                         verbose=False)

        # test task model using K samples and get test loss for metatraining training update
        task_inputs, task_targets, task_outputs, task_loss = test_task_model_grads(task_model,
                                                                                   task_test_dl,
                                                                                   verbose=False)

        # check that gradients of task_model after testing are not equal zero
        for p in task_model.parameters():
            assert p.grad is not None, "It seems that gradients of task-model are not calculated, " \
                                       "check them out: {}".format(p.grad)

        # check that gradients of meta_model after testing are not equal zero
        for p in meta_model.parameters():
            assert p.grad is not None, "It seems that gradients of meta-model are not calculated, " \
                                       "check them out: {}".format(p.grad)

        # replace gradients of meta model by gradients of task model after testing
        for p_task,p_meta in zip(task_model.parameters(), meta_model.parameters()):
            # p_meta.grad.data = p_task.grad.data
            p_meta.grad.data = p_meta.grad.data + p_task.grad.data  # gradients accumulation

        # save loss
        losses.append(task_loss.item())

        if verbose:
            print("inputs: {}, \npredicts: {}, \ntargets: {}, \nloss: {}"
                  .format(task_inputs.data, meta_outputs.data, task_targets.data, loss.item()))

    # clip gradients to [-10...10]
    # clip_grad_value_(meta_model.parameters(), clip_value=10)
    # update model weights by accumulated gradients
    optimizer.step()
    # clear the gradients
    optimizer.zero_grad()

    return np.mean(losses)
