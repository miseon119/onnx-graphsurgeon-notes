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

## Isolating A Subgraph

First, let's generate an onnx file; The generated model computes `Y = x0 + (a * x1 + b)`:
```python
import onnx_graphsurgeon as gs
import numpy as np
import onnx

# Computes Y = x0 + (a * x1 + b)

shape = (1, 3, 224, 224)
# Inputs
x0 = gs.Variable(name="x0", dtype=np.float32, shape=shape)
x1 = gs.Variable(name="x1", dtype=np.float32, shape=shape)

# Intermediate tensors
a = gs.Constant("a", values=np.ones(shape=shape, dtype=np.float32))
b = gs.Constant("b", values=np.ones(shape=shape, dtype=np.float32))
mul_out = gs.Variable(name="mul_out")
add_out = gs.Variable(name="add_out")

# Outputs
Y = gs.Variable(name="Y", dtype=np.float32, shape=shape)

nodes = [
    # mul_out = a * x1
    gs.Node(op="Mul", inputs=[a, x1], outputs=[mul_out]),
    # add_out = mul_out + b
    gs.Node(op="Add", inputs=[mul_out, b], outputs=[add_out]),
    # Y = x0 + add
    gs.Node(op="Add", inputs=[x0, add_out], outputs=[Y]),
]

graph = gs.Graph(nodes=nodes, inputs=[x0, x1], outputs=[Y])
onnx.save(gs.export_onnx(graph), "model.onnx")
```
![Screen Shot 2022-05-19 at 8 44 36 PM](https://user-images.githubusercontent.com/19248035/169285760-4756877e-db5c-4f24-974b-788c611cc3ef.png)

```python
import onnx_graphsurgeon as gs
import numpy as np
import onnx

model = onnx.load("model.onnx")
graph = gs.import_onnx(model)

tensors = graph.tensors()

graph.inputs = [tensors["x1"].to_variable(dtype=np.float32)]
graph.outputs = [tensors["add_out"].to_variable(dtype=np.float32)]

graph.cleanup()

onnx.save(gs.export_onnx(graph), "subgraph.onnx")
```

![subgraph](https://github.com/NVIDIA/TensorRT/raw/main/tools/onnx-graphsurgeon/examples/resources/03_subgraph.onnx.png)
