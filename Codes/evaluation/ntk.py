import torch
from torch.func import vmap, jacrev, functional_call


def get_ntk(model, x1, x2, compute='full'):
    model.eval() 

    params = dict(model.named_parameters())

    def fnet_single(params, x):
        return functional_call(model, params, (x.unsqueeze(0))).squeeze(0)

    # Compute J(x1) and J(x2)
    jac1 = vmap(jacrev(fnet_single, argnums=1), (None, 0))(params, x1)
    jac2 = vmap(jacrev(fnet_single, argnums=1), (None, 0))(params, x2)

    jac1 = jac1.flatten(start_dim=2)
    jac2 = jac2.flatten(start_dim=2)
    
    # Compute J(x1) @ J(x2).T
    if compute == 'full':
        ntk = torch.einsum('Naf,Mbf->NMab', jac1, jac2)  # Full NTK with classes
    elif compute == 'trace':
        ntk = torch.einsum('ncp,mcp->nm', jac1, jac2)  # Sum over params and classes
    elif compute == 'diagonal':
        ntk = torch.einsum('ncp,mcp->nmp', jac1, jac2)  # Keep param dimension
    else:
        raise ValueError(f"Unknown compute mode: {compute}")

    return ntk
