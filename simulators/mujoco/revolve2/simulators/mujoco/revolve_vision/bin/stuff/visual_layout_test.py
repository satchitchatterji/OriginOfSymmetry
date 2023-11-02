import random
import sys

import abrain

n = int(sys.argv[1])

genome = abrain.Genome.random(random.Random(0))

n_type = abrain.ANN.Neuron.Type
labels = {t: [] for t in [n_type.I, n_type.H, n_type.O]}

p = abrain.Point(0, 0, 0)
p1 = p
p2 = abrain.Point(0, 0, 0)

inputs, outputs = [], []
for side in [-1, 1]:
    for x in [.33, .66]:
        for z in [-.5, .5]:
            p = abrain.Point(side * x, -1, z)
            inputs.append(p)
            labels[p] = f"joint{side:+}-{x}-{z}"
            p = abrain.Point(side * x, 1, z)
            outputs.append(p)
            labels[p] = f"motor{side:+}-{x}-{z}"

for rgb in range(3):
    for i in range(n):
        for j in range(n):
            x = 2 * i / (n-1) - 1
            z = 2 * j / (n-1) - 1
            y = -1 + rgb / 3
            p = abrain.Point(x, y, z)
            inputs.append(p)
            labels[p] = f"retina{'RGB'[rgb]}-{i}-{j}"

brain = abrain.ANN.build(inputs, outputs, genome)

abrain.plotly_render(brain, labels).write_html("ann_layout_test.html")

genome.to_dot("foo", "png")
