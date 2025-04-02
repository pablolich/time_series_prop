import time

import numpy as np

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import optax  # https://github.com/deepmind/optax

import sys
import os

# Get the absolute path of the parent directory (code/)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import data

path_name = "../../data/glv_generic"
file_list = os.listdir(path_name)
file_list = [os.path.join(path_name, file_name) for file_name in file_list]

class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.mlp(y)


class NeuralODE(eqx.Module):
    func: Func

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(data_size, width_size, depth, key=key)

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys

def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            #batch_perm = perm[start:end]
            batch_indices = indices[start:dataset_size]
            yield tuple(array[batch_indices] for array in arrays)#before, [batch_perm]
            start = end
            end = start + batch_size

schedule = optax.exponential_decay(init_value = 0.001,
        transition_steps = 100,
        decay_rate= 0.99,
        end_value = 0.00001)

def main(
    batch_size=1,
    lr_strategy=(0.001, schedule),
    steps_strategy=(500, 500000),
    length_strategy=(1, 1),
    width_size=16,
    depth=2,
    seed=2,
    plot=True,
    print_every=100,
):
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key = jr.split(key, 3)

    data_glv4 = data.Data(file_list)
    indices = [0, 1, 2, 3, 4,  5, 6, 7]
    data_prop = [data_glv4.proportions[i] for i in indices]
    data_abund = [data_glv4.abundances[i] for i in indices]
    ys = jnp.array(data_prop)
    xs = jnp.array(data_abund)
    ts = jnp.array(data_glv4.times[0])

    _, length_size, data_size = ys.shape

    model = NeuralODE(data_size, width_size, depth, key=model_key)

    # Training loop like normal.
    #
    # Only thing to notice is that up until step 500 we train on only the first 10% of
    # each time series. This is a standard trick to avoid getting caught in a local
    # minimum.

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
        T = y_pred.sum(axis = -1, keepdims = True) #get predicted totals
        y_pred_normalized = y_pred / T
        #errors can only be big if the totals are small (assumed sampling errors)
        return jnp.mean((yi - y_pred_normalized) ** 2)

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy):
        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        _ts = ts[: int(length_size * length)]
        _ys = ys[:, : int(length_size * length)]
        for step, (yi,) in zip(
            range(steps), dataloader((_ys,), batch_size, key=loader_key)
        ):
            start = time.time()
            loss, model, opt_state = make_step(_ts, yi, model, opt_state)
            end = time.time()
            if (step % print_every) == 0 or step == steps - 1:
                print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}")
    nexp = 4
    nrep = 2
    if plot:

        # Define a colormap for species differentiation
        base_colors = list(mcolors.TABLEAU_COLORS.values())  # Base colors for species
        num_species = ys[0].shape[1]

        # Generate two shades for each species (one for each replicate)
        replicate_colors = [
            (mcolors.to_rgba(base, alpha=0.8), mcolors.to_rgba(base, alpha=0.5))  # Brighter and darker shade
            for base in base_colors[:num_species]
        ]

        fig, axes = plt.subplots(2, nexp, figsize=(16, 8))  # 2 rows, 4 columns

        # Loop through the four experiments
        for exp in range(nexp):
            for rep in range(nrep):  # Two replicates per experiment
                idx = 2 * exp + rep  # Data index
                proportions = ys[idx]
                abundances = xs[idx]
                
                # Model predictions (normalize for proportions)
                model_y = model(ts, proportions[0, :])  
                model_y_normalized = model_y / model_y.sum(axis=-1, keepdims=True)

                # Top row: Proportions
                ax1 = axes[0, exp]
                for j in range(num_species):
                    real_color, model_color = replicate_colors[j]  # Shades for species

                    ax1.plot(ts, proportions[:, j], linestyle='-', color=real_color, label=f"Real S{j+1} (Rep {rep+1})")
                    ax1.plot(ts, model_y_normalized[:, j], linestyle='--', color=model_color, label=f"Model S{j+1} (Rep {rep+1})")
                
                # Bottom row: Abundances (log scale)
                ax2 = axes[1, exp]
                model_y = model(ts, abundances[0, :])  # Model forward propagation

                for j in range(num_species):
                    real_color, model_color = replicate_colors[j]

                    ax2.plot(ts, abundances[:, j], linestyle='-', color=real_color, label=f"Real S{j+1} (Rep {rep+1})")
                    ax2.plot(ts, model_y[:, j], linestyle='--', color=model_color, label=f"Model S{j+1} (Rep {rep+1})")

                ax2.set_xlabel("Time")

                # Formatting
                axes[0, exp].set_title(f"Experiment {exp+1}")
                axes[1, exp].set_xlabel("Time")
                axes[0, exp].set_ylabel("Proportion")
                axes[1, exp].set_ylabel("Abundance (log scale)")

        # Adjust layout, add legends
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.15, 1))

        plt.tight_layout()
        plt.savefig("neural_ode_comparison.png")
        plt.show()

    return ts, ys, model

ts, ys, model = main()
