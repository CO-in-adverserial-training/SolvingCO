import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.animation import PillowWriter, FuncAnimation
import os
from tqdm import tqdm


def collect_pgd_grads(model, x, y, upper_limit, lower_limit, epsilon: float = 8/255, alpha: float = 0.25, attack_iters: int = 10, k: float = 0.0, clip: bool = True, device: str = 'cuda'):
    grads = []

    # Initialize random step
    delta = torch.zeros_like(x).to(device)
    if k != 0:
        for j in range(len(epsilon)):
            delta[:, j, :, :].uniform_(-k * epsilon[j][0][0].item(), k * epsilon[j][0][0].item())
        delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    delta.requires_grad = True

    for _ in range(attack_iters):
        output = model(x + delta)
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()
        grads.append(grad.clone().view(grad.size(0), -1))
        delta.data = delta + alpha * epsilon * torch.sign(grad)
        if clip:
            delta.data = torch.clamp(delta, -epsilon, epsilon)
        delta.data = torch.clamp(delta, lower_limit - x, upper_limit - x)
        delta.grad.zero_()
    delta = delta.detach()

    grads = torch.stack(grads, dim=1)
    return grads

def cosine_similarity_matrix(grads):  # grads: (batch, steps, dim)
    batch_size, steps, dim = grads.shape
    sims = []
    for b in range(batch_size):
        g = grads[b]  # (steps, dim)
        g = F.normalize(g, dim=1)
        sim = torch.matmul(g, g.T).cpu().numpy()  # (steps x steps)
        sims.append(sim)
    return np.mean(sims, axis=0)  # average over batch


def pgd_gradient_similarity_gif(architecture, saving_directory, loader, lower_limit, upper_limit, std, epsilon=8/255, 
                                alpha=0.25, pgd_steps=10, num_classes=10, num_iters=10, n_epochs=30, device='cuda'):
    model = architecture(num_classes=num_classes).to(device)

    similarity_matrices = []

    for epoch in range(n_epochs+1):
        model.load_state_dict(torch.load(f'{saving_directory}/weights_{str(epoch).zfill(2)}.pth', weights_only=False))
        model.eval()

        grads_all = []

        iters = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            grads = collect_pgd_grads(model, x, y, upper_limit, lower_limit, epsilon=epsilon/std, alpha=alpha, attack_iters=pgd_steps, device=device)
            grads_all.append(grads)
            iters += 1
            if iters == num_iters:
                break

        grads_all = torch.cat(grads_all, dim=0)  # (N * B, steps, dim)
        sim_matrix = cosine_similarity_matrix(grads_all)
        print(f'Epcoh {epoch} | Min {np.min(sim_matrix):.2f} | Max {np.max(sim_matrix):.2f}')
        similarity_matrices.append(sim_matrix)

    # Create the GIF
    fig, ax = plt.subplots(dpi=300)
    fig.tight_layout()
    cax = ax.imshow(similarity_matrices[0], cmap='viridis', vmin=-1.0, vmax=1.0)
    fig.colorbar(cax)
    title = ax.set_title('Epoch 0')

    def update(frame):
        cax.set_data(similarity_matrices[frame])
        title.set_text(f"Epoch {frame}")
        return [cax, title]

    anim = FuncAnimation(fig, update, frames=n_epochs+1, blit=False)
    gif_path=f'{saving_directory}/pgd_cosine_{pgd_steps}.gif'
    anim.save(gif_path, writer=PillowWriter(fps=2))
    plt.close()
    
    return gif_path