import torch
from torch.func import vmap, jacrev, functional_call


def get_ntk_from_logits(model, x1, x2, compute='full'):
    model.eval() 

    params = dict(model.named_parameters())

    def fnet_single(params, x):
        return functional_call(model, params, (x.unsqueeze(0))).squeeze(0)

    # Compute J(x1) and J(x2)
    jac1 = vmap(jacrev(fnet_single, argnums=0), (None, 0))(params, x1)
    for key in jac1:
        jac1[key] = jac1[key].detach().cpu()

    jac2 = vmap(jacrev(fnet_single, argnums=0), (None, 0))(params, x2)
    for key in jac2:
        jac2[key] = jac1[key].detach().cpu()

    # Vectorize J(x1) and J(x2)
    jac1 = torch.cat([t.flatten(start_dim=2) for t in jac1.values()], dim=2).cuda()
    jac2 = torch.cat([t.flatten(start_dim=2) for t in jac2.values()], dim=2).cuda()
    
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


def get_ntk_from_loss(model, x1, x2, y1, y2):
    model.eval() 

    params = dict(model.named_parameters())

    def compute_loss(params, inputs, targets):
        prediction = functional_call(model, params, (inputs.unsqueeze(0))).squeeze(0)
        return torch.nn.functional.cross_entropy(prediction, targets)

    # Compute J(x1) and J(x2)
    jac1 = vmap(jacrev(compute_loss, argnums=0), (None, 0, 0))(params, x1, y1)
    jac2 = vmap(jacrev(compute_loss, argnums=0), (None, 0, 0))(params, x2, y2)

    # Vectorize J(x1) and J(x2)
    jac1 = torch.cat([t.flatten(start_dim=1) for t in jac1.values()], dim=1)
    jac2 = torch.cat([t.flatten(start_dim=1) for t in jac2.values()], dim=1)

    return jac1 @ jac2.T