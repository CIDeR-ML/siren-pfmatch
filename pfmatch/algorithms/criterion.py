import torch

class PoissonMatchLoss(torch.nn.Module):
    """
    Poisson NLL Loss for gradient-based optimization model
    """
    def __init__(self):
        super(PoissonMatchLoss, self).__init__()
        self.poisson_nll = torch.nn.PoissonNLLLoss(log_input=False,
            full=True,
            reduction="none")

    def forward(self, input, target, weight=1., axis=-1):
        H = torch.clamp(input, min=0.01)
        O = torch.clamp(target, min=0.01)
        loss = self.poisson_nll(H, O) - torch.log(H) / 2
        return torch.nanmean(weight*loss, axis=axis)

class Chi2Loss(torch.nn.Module):
    '''
    Chi2 loss w/ additional (constant) error terms.
    '''
    def __init__(self, eps=0.01):
        super().__init__()
        self.register_buffer(
            'eps', torch.as_tensor(eps, dtype=torch.float32)
        )
        
    def forward(self, input, target, axis=-1):
        H = input.clamp(min=0)
        O = target.clamp(min=0)
        return torch.nanmean((H - O)**2 / (O + self.eps), axis=axis)

    def __str__(self):
        return f'Chi2Loss(eps={self.eps})'
