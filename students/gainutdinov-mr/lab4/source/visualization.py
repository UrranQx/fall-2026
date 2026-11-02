"""Визуализация экспериментов лабораторной работы № 4."""

import matplotlib.pyplot as plt


def plot_reconstruction_error(errors, output_path):
    """Сохранить график ошибки восстановления в зависимости от размерности."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    try:
        ax.plot(range(1, len(errors) + 1), errors, marker="o", label=r"$E_m$")
        ax.set_xlabel("m")
        ax.set_ylabel(r"$E_m$")
        ax.grid(True)
        ax.legend()
        fig.savefig(output_path, bbox_inches="tight")
    finally:
        plt.close(fig)

