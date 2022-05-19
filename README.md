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

Creates an ONNX model containing a single Convolution node with weights

```python
import onnx_graphsurgeon as gs
import numpy as np
import onnx

X = gs.Variable(name="X", dtype=np.float32, shape=(1, 3, 224, 224))
# Since W is a Constant, it will automatically be exported as an initializer
W = gs.Constant(name="W", values=np.ones(shape=(5, 3, 3, 3), dtype=np.float32))

Y = gs.Variable(name="Y", dtype=np.float32, shape=(1, 5, 222, 222))

node = gs.Node(op="Conv", inputs=[X, W], outputs=[Y])

# Note that initializers do not necessarily have to be graph inputs
graph = gs.Graph(nodes=[node], inputs=[X], outputs=[Y])
onnx.save(gs.export_onnx(graph), "test_conv.onnx")

```

![onnx2](https://user-images.githubusercontent.com/19248035/169283872-4cd30ffd-7f91-4e08-b860-54f950521fc2.png)
