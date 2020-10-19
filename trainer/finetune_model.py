from trainer import *


def finetune_model(modelclass,
                   pretrain_path,
                   lr,
                   K,
                   nepoch=10,
                   xnum=100,
                   seed=42,
                   verbose=False):

    # qualitative results
    results = {}

    # xs for results datapoints
    xs = np.linspace(*input_range, xnum)
    xs = xs[:, np.newaxis]
    xs = torch.tensor(xs).float()

    # create model
    model = modelclass(dim_input=dim_input,
                       dim_output=dim_output,
                       dim_hidden=dim_hidden)

    # load pretrained model
    model.load_state_dict(torch.load(pretrain_path))

    # add pre-update
    results["pre-update"] = model(xs).data.numpy().ravel()

    # make dataset and dataloaders for K-shot fine tuning
    set_seed(seed=seed)
    dataset = TaskDataset(size=K)
    train_dl = DataLoader(dataset, batch_size=K, shuffle=True)

    # loss function
    lf = MSELoss(reduction='mean')
    losses = []

    # pre-update loss
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(train_dl):
            # compute the model output
            outputs = model(inputs)
            # calculate train loss
            loss = lf(outputs, targets)
            # save loss
            losses.append(loss.item())

        if verbose: print("Epoch: {}, loss: {}".format(0, losses[0]))

    # define the optimization
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(nepoch):
        # fix seed so that have the same samples
        set_seed(seed=seed, verbose=verbose)
        # run minibatches of dataset (= one training epoch)
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            outputs = model(inputs)
            # calculate train loss
            loss = lf(outputs, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            # save loss
            losses.append(loss.item())

        if verbose: print("Epoch: {}, loss: {}".format(epoch + 1, loss))
        # add qualitative results
        results[epoch + 1] = model(xs).data.numpy().ravel()

    # xs for qualitative results
    xs = xs.data.numpy().ravel()

    # add ground truth
    amp = dataset.amp
    phase = dataset.phase
    results["ground truth"] = amp * np.sin(xs - phase)

    # datapoints xs and ys
    inputs = inputs.data.numpy().ravel()
    outputs = amp * np.sin(inputs - phase)

    return losses, results, inputs, outputs, xs
