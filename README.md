# onnx-graphsurgeon-notes

## Install ONNX Graphsurgeon

```bash
python3 -m pip install onnx_graphsurgeon
```

## Generate Sample ONNX File

```python
import onnx_graphsurgeon as gs
import numpy as np
import onnx

X = gs.Variable(name="X", dtype=np.float32, shape=(1, 3, 5, 5))
Y = gs.Variable(name="Y", dtype=np.float32, shape=(1, 3, 1, 1))
node = gs.Node(op="GlobalLpPool", attrs={"p": 2}, inputs=[X], outputs=[Y])

graph = gs.Graph(nodes=[node], inputs=[X], outputs=[Y])
onnx.save(gs.export_onnx(graph), "test_globallppool.onnx")
```
![onnx1](https://user-images.githubusercontent.com/19248035/169281948-38654a30-11ce-47a3-be6d-4112f3cd1c09.png)

