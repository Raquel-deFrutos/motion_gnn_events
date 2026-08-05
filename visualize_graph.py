import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--graph",
    type=str,
    required=True,
    help="Ruta al archivo .npz"
)

args = parser.parse_args()

graph_path = Path(args.graph)

data = np.load(graph_path)

coords = data["coords"]          # (N, 3)
edge_index = data["edge_index"]  # (E, 2)

print(f"Nodos: {len(coords)}")
print(f"Aristas: {len(edge_index)}")

if len(coords) == 0:
    print("El grafo está vacío.")
    exit()

# Convertir de [-1, 1] a píxeles
W = 346
H = 260

x = (coords[:, 0] + 1) * W / 2
y = (coords[:, 1] + 1) * H / 2

plt.figure(figsize=(10, 8))

# Dibujar aristas
for src, dst in edge_index:
    plt.plot(
        [x[src], x[dst]],
        [y[src], y[dst]],
        color="lightgray",
        linewidth=0.5,
        alpha=0.5
    )

# Dibujar nodos
plt.scatter(
    x,
    y,
    c=coords[:, 2],   # color según el tiempo
    cmap="plasma",
    s=40
)

plt.colorbar(label="Tiempo normalizado")
plt.gca().invert_yaxis()

plt.title(graph_path.name)
plt.xlabel("x")
plt.ylabel("y")

plt.show()