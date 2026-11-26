"""Latency probe graph: the code predictor's builder graph chained 15 times (teacher-forced, no sampling).

Inputs: x0 [1,2,1024] (the first two rows), emb [1,14,1024] (rows for steps 1..14), the 10 empty past tensors.
Output: logits_all [1,15,2048] (step j's head-j slice at its last position). Weights stay shared (same initializers).
"""
import sys
import onnx
from onnx import helper, numpy_helper, TensorProto
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
NOPAST = len(sys.argv) > 3 and sys.argv[3] == "nopast"
STEPS, GROUP, LAYERS = 15, 2048, 5
m = onnx.load(src, load_external_data=False)
g = m.graph
inits = {i.name for i in g.initializer}
graph_inputs = [i.name for i in g.input]
print("graph inputs", graph_inputs[:3], "...", len(graph_inputs))
past_names = [f"past_key_values.{i}.{kv}" for i in range(LAYERS) for kv in ("key", "value")]
present_names = [f"present.{i}.{kv}" for i in range(LAYERS) for kv in ("key", "value")]
assert all(p in graph_inputs for p in past_names), "unexpected past names"
for n in g.node:
    for a in n.attribute:
        assert a.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS), f"subgraph in {n.name}"

new_nodes, new_inits, step_logits = [], [], []
for k in range(STEPS):
    p = f"s{k}/"
    mapping = {}
    mapping["inputs_embeds"] = p + "inputs_embeds"
    mapping["attention_mask"] = p + "attention_mask"
    for i, name in enumerate(past_names):
        mapping[name] = name if k == 0 else f"s{k-1}/{present_names[i]}"

    def r(name):
        if name == "" or name in inits:
            return name
        return mapping.get(name, p + name)

    # the step's input rows
    if k == 0:
        new_nodes.append(helper.make_node("Identity", ["x0"], [p + "inputs_embeds"], name=p + "in"))
    else:
        for nm, val in (("starts", [k - 1]), ("ends", [k]), ("axes", [1])):
            new_inits.append(numpy_helper.from_array(np.array(val, dtype=np.int64), p + "e_" + nm))
        new_nodes.append(helper.make_node("Slice", ["emb", p + "e_starts", p + "e_ends", p + "e_axes"], [p + "inputs_embeds"], name=p + "in"))
    new_inits.append(numpy_helper.from_array(np.ones((1, 2 + k), dtype=np.int64), p + "attention_mask"))
    for n in g.node:
        ins = [r(x) for x in n.input]
        if NOPAST and k == 0 and n.op_type == "GroupQueryAttention":
            ins[3] = ""
            ins[4] = ""
        c = helper.make_node(n.op_type, ins, [r(x) if x != "" else "" for x in n.output], name=p + n.name, domain=n.domain)
        c.attribute.extend(n.attribute)
        new_nodes.append(c)
    # this step's head slice at the last position
    lg = r("logits")
    for nm, val in (("starts", [-1, k * GROUP]), ("ends", [2**62, (k + 1) * GROUP]), ("axes", [1, 2])):
        new_inits.append(numpy_helper.from_array(np.array(val, dtype=np.int64), p + "l_" + nm))
    new_nodes.append(helper.make_node("Slice", [lg, p + "l_starts", p + "l_ends", p + "l_axes"], [p + "logits_slice"], name=p + "slice"))
    step_logits.append(p + "logits_slice")

new_nodes.append(helper.make_node("Concat", step_logits, ["logits_all"], axis=1, name="concat_logits"))

x0 = helper.make_tensor_value_info("x0", TensorProto.FLOAT, [1, 2, 1024])
emb = helper.make_tensor_value_info("emb", TensorProto.FLOAT, [1, STEPS - 1, 1024])
pasts = [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 8, 0, 128]) for n in past_names]
out = helper.make_tensor_value_info("logits_all", TensorProto.FLOAT, [1, STEPS, GROUP])
# GQA's shape inference copies the past's dims to the present; chained, every step would then share one symbolic
# size and ONNX Runtime would reuse buffers across steps. Declare each step's present with its own size symbol.
infos = []
for k in range(STEPS):
    for name in present_names:
        infos.append(helper.make_tensor_value_info(f"s{k}/{name}", TensorProto.FLOAT, [1, 8, 2 + k, 128]))
ng = helper.make_graph(new_nodes, "cp_chain", [x0, emb] + ([] if NOPAST else list(pasts)), [out], initializer=list(g.initializer) + new_inits, value_info=infos)
nm = helper.make_model(ng, opset_imports=list(m.opset_import))
nm.ir_version = m.ir_version
onnx.save(nm, dst)
print("saved", dst, "nodes", len(new_nodes), "orig nodes", len(g.node))
