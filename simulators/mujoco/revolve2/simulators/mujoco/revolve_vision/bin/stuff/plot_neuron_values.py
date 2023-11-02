import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import cv2

from matplotlib import rc

# activate latex text rendering
rc('text', usetex=True)

i_path = Path(sys.argv[1])
df = pd.read_csv(i_path, delimiter=' ', index_col='Step')
print(df)

# axes = df.plot(subplots=True, legend=False)#, ylim=(-1.05, 1.05))
#
# for ax, label in zip(axes, df.columns):
#     ax.yaxis.set_label_position("right")
#     label = label.split(":")
#     ax.set_ylabel(f"{label[1]}\n\\small{label[0]}")
#
# o_path = i_path.with_suffix(".pdf")
# fig = plt.gcf()
# fig.set_size_inches(5, 1 * df.shape[1])
# fig.savefig(o_path, bbox_inches='tight')

non_retina_columns = [c for c in df.columns if '[' not in c]
df = df.drop(non_retina_columns, axis=1)

df.columns = [c.split(":")[-1] for c in df.columns]

print(df.columns)
indices = []
for col in df.columns:
    ix = (
        *map(int,
            reversed(
                "".join(c for c in col if c not in "RGB[]").split(",")
            )
        ),
        "RGB".index(col[0])
    )
    indices.append(ix)

img_shape = np.max(indices, axis=0) + 1
img = np.zeros(img_shape, dtype=np.uint8)
for i in range(df.shape[0]):
    for j in range(len(df.columns)):
        ix = indices[j]
        v = round(255*df.iloc[i, j])
        if i == 0:
            print(f"img[{ix}|{df.columns[j]}] = {v}")
        img[ix] = v
    if i == 0:
        print(img)
    cv2.imwrite(f'vision_regenerated_{i:010d}.png', img)
