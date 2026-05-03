#!/usr/bin/env python3
"""Generate formula images for PPT slides."""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

OUTDIR = "/Users/clouseren/klaus_ai/mu-muzero/assets/formulas/"

formulas = {
    "pomdp": r"$\mathbf{s}_{t+1} = f(\mathbf{s}_t, \mathbf{a}_t) + \mathbf{w}_t$",
    "observation": r"$\mathbf{o}_t = g(\mathbf{s}_t) + \mathbf{\epsilon}_t$ ",
    "stochastic_dynamics": r"$\mathbf{s}' \sim \mathcal{N}\left(\mathbf{\mu}_g(\mathbf{s},\mathbf{a}), \mathbf{\sigma}_g(\mathbf{s},\mathbf{a})\right)$ ",
    "ucb": r"$U(s,a) = Q(s,a) + c_{\mathrm{puct}} P(a|s) \frac{\sqrt{N(s)}}{1+N(s,a)} \sigma_{\mathrm{obs}}(s)$",
    "langevin": r"$\gamma \dot{\mathbf{x}} = \mathbf{F}_{\mathrm{optical}} + \mathbf{F}_{\mathrm{hydro}} + \mathbf{F}_{\mathrm{Brownian}}$",
    "factored_action": r"$\pi(a|s) = \pi_{\mathrm{trap}}(k|s) \cdot \pi_{\mathrm{prim}}(m|k,s)$",
    "reward": r"$R = R_{\mathrm{task}} + R_{\mathrm{efficiency}} + R_{\mathrm{safety}}$",
    "value_backup": r"$V(s) = \max_a \left[ R(s,a) + \gamma \mathbb{E}[V(s')] \right]$",
    "depth_rmse": r"$\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (z_i - \hat{z}_i)^2}$",
    "brownian": r"$\langle \mathbf{F}_B(t) \mathbf{F}_B(t')^\top \rangle = 2\gamma k_B T \delta(t-t') \mathbf{I}$",
}

for name, formula in formulas.items():
    fig, ax = plt.subplots(figsize=(8, 1.5))
    ax.text(0.5, 0.5, formula, ha='center', va='center', fontsize=28)
    ax.axis('off')
    plt.savefig(f"{OUTDIR}{name}.png", dpi=200, bbox_inches='tight', pad_inches=0.1, transparent=True)
    plt.close()
    print(f"Generated {name}.png")

print("All formulas generated!")
