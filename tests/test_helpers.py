import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD


class VariablesChangeException(Exception):
    pass


"""Set random seed for torch"""
torch.manual_seed(42)


def _train_step(model, loss_fn, optim, batch, **kwargs):
    """Run a training step on model for a given batch of data
    Parameters of the model accumulate gradients and the optimizer performs
    a gradient update on the parameters
    Parameters
    ----------
    model : torch.nn.Module
    torch model, an instance of torch.nn.Module
    loss_fn : function
    a loss function from torch.nn.functional
    optim : torch.optim.Optimizer
    an optimizer instance
    batch : list
    a 2 element list of inputs and labels, to be fed to the model
    """

    # put model in train mode
    model.train()

    #  run one forward + backward step
    # clear gradient
    optim.zero_grad()
    # forward
    logits, labels = model(batch, **kwargs)
    # calc loss
    loss = loss_fn(logits.squeeze(), labels.float())
    # backward
    loss.backward()
    # optimization step
    optim.step()


def _forward_step(model, batch, **kwargs):
    """Run a forward step of model for a given batch of data
    Parameters
    ----------
    model : torch.nn.Module
    torch model, an instance of torch.nn.Module
    batch : list
    a 2 element list of inputs and labels, to be fed to the model
    Returns
    -------
    torch.tensor
    output of model's forward function
    """

    # put model in evaluate mode
    model.eval()
    batch_size = batch[0].shape[0]
    with torch.no_grad():
        # inputs and targets <- forward pass
        logits, labels = model(batch, **kwargs)
    assert logits.shape == torch.Size([batch_size, 1])
    assert labels.shape == torch.Size([batch_size])


def _var_change_helper(vars_change, model, batch, params=None, **kwargs):
    """Check if given variables (params) change or not during training
    If parameters (params) aren't provided, check all parameters.
    Parameters
    ----------
    vars_change : bool
    a flag which controls the check for change or not change
    model : torch.nn.Module
    torch model, an instance of torch.nn.Module
    batch : tuple
    a 3 element tuple of inputs, labels and lengths, to be fed to the model
    params : list, optional
    list of parameters of form (name, variable)
    Raises
    ------
    VariablesChangeException
    if vars_change is True and params DO NOT change during training
    if vars_change is False and params DO change during training
    """
    loss_fn = BCEWithLogitsLoss()
    optim = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1.0)

    if params is None:
        # get a list of params that are allowed to change
        params = [np for np in model.named_parameters() if np[1].requires_grad]

    # take a copy
    initial_params = [(name, p.clone()) for (name, p) in params]

    # run a training step
    _train_step(model, loss_fn, optim, batch, **kwargs)

    # check if variables have changed
    for (_, p0), (name, p1) in zip(initial_params, params):
        try:
            if vars_change:
                assert not torch.equal(p0, p1)
            else:
                assert torch.equal(p0, p1)
        except AssertionError:
            raise VariablesChangeException(  # error message
                "{var_name} {msg}".format(
                    var_name=name,
                    msg='did not change!' if vars_change else 'changed!'
                )
            )
