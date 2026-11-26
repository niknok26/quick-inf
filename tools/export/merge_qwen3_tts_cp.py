"""One-graph code predictor for Qwen3-TTS (0.6B: talker width == code-predictor width).

Inputs : hidden [1,1,H] (the talker's last hidden state), c0 [1,1] int64 (its sampled first code),
         gumbel [15,1,2048] float32 = TEMP * (-log(-log(u))), u uniform(0,1) (the sampling noise, made by the caller) [+ 10 empty past tensors unless --nopast]
Outputs: codes [1,15] float32 holding the integer codes of codebooks 1..15, embed_sum [1,1,H] (sum of the frame's 16 code embeddings, to feed the talker)
Built from the builder's code-predictor graph chained 15 times (same weight initializers), the codec-embedding lookup
(GatherBlockQuantized, one table per codebook) and top-k (50) / temperature (0.9) sampling as Gumbel-max over the
logits at or above the 50th largest (argmax(logit + TEMP*g) = argmax(logit/TEMP + g)). Everything stays in float32 or goes straight from ArgMax to the lookup: int64 Reshape/Add/GatherElements
run on the CPU in ONNX Runtime WebGPU and would cost a GPU round trip per step.
usage: merge_cp.py CP_MODEL.onnx CODEC_EMBED_Q4.onnx OUT.onnx [--nopast]
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

cp_path, emb_path, dst = sys.argv[1:4]
NOPAST = "--nopast" in sys.argv
NOTOPK = "--notopk" in sys.argv  # timing experiment only: sample from all 2048 logits
STEPS, GROUP, LAYERS, H, TOPK, TEMP, VOCAB = 15, 2048, 5, 1024, 50, 0.9, 3072

m = onnx.load(cp_path, load_external_data=False)
g = m.graph
emb = onnx.load(emb_path, load_external_data=False).graph
inits = {i.name for i in g.initializer}
past_names = [f"past_key_values.{i}.{kv}" for i in range(LAYERS) for kv in ("key", "value")]
present_names = [f"present.{i}.{kv}" for i in range(LAYERS) for kv in ("key", "value")]
for n in g.node:
    for a in n.attribute:
        assert a.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
gbq = next(n for n in emb.node if n.op_type == "GatherBlockQuantized")
assert emb.initializer[0].dims[0] == VOCAB + STEPS * GROUP, emb.initializer[0].dims

nodes, new_inits, infos = [], [], []
def const(name, arr):
    new_inits.append(numpy_helper.from_array(np.asarray(arr), name))
    return name
def node(op, ins, outs, name, **attrs):
    nodes.append(helper.make_node(op, ins, outs, name=name, **attrs))
    return outs[0]

# One quantized table per codebook (rows 0..3071 = the talker's codec embedding, then 15 x 2048), so a sampled code
# indexes its table directly and no int64 arithmetic is needed.
q4, scales = emb.initializer[0], emb.initializer[1]
assert q4.name == gbq.input[0] and scales.name == gbq.input[2], (q4.name, scales.name)
row_q4, row_sc = q4.dims[1] // 2, scales.dims[1] * 4
bounds = [(0, VOCAB)] + [(VOCAB + j * GROUP, VOCAB + (j + 1) * GROUP) for j in range(STEPS)]
tables = []
for j, (a, b) in enumerate(bounds):
    tq = onnx.TensorProto(); tq.name = f"tbl{j}_Q4"; tq.data_type = q4.data_type; tq.dims.extend([b - a, q4.dims[1]]); tq.raw_data = q4.raw_data[a * row_q4:b * row_q4]
    ts = onnx.TensorProto(); ts.name = f"tbl{j}_scales"; ts.data_type = scales.data_type; ts.dims.extend([b - a, scales.dims[1]]); ts.raw_data = scales.raw_data[a * row_sc:b * row_sc]
    new_inits.extend([tq, ts]); tables.append((tq.name, ts.name))

lookup_count = [0]
def lookup(ids, table):
    """Embedding rows of `table` (0 = c0's) for int64 ids [1,1] -> [1,1,H]."""
    lookup_count[0] += 1
    out = f"emb{lookup_count[0]}/out"
    c = helper.make_node("GatherBlockQuantized", [tables[table][0], ids, tables[table][1]], [out], name=f"emb{lookup_count[0]}", domain=gbq.domain)
    c.attribute.extend(gbq.attribute)
    nodes.append(c)
    return out

k_const = const("topk_k", np.array([TOPK], dtype=np.int64))
neg_inf = const("neg_inf", np.array(-1e30, dtype=np.float32))
arange = const("arange", np.arange(GROUP, dtype=np.float32))
axes2 = const("axes2", np.array([2], dtype=np.int64))
kth_starts = const("kth_starts", np.array([TOPK - 1], dtype=np.int64))
kth_ends = const("kth_ends", np.array([TOPK], dtype=np.int64))
kth_axes = const("kth_axes", np.array([2], dtype=np.int64))

e0 = lookup("c0", 0)
x = node("Concat", ["hidden", e0], ["x0"], "cat_x0", axis=1)
rows, toks = [e0], []
for k in range(STEPS):
    p = f"s{k}/"
    mapping = {"inputs_embeds": x, "attention_mask": p + "attention_mask"}
    for i, name in enumerate(past_names):
        mapping[name] = name if k == 0 else f"s{k-1}/{present_names[i]}"
    def r(name, p=p, mapping=mapping):
        return name if name == "" or name in inits else mapping.get(name, p + name)
    const(p + "attention_mask", np.ones((1, 2 + k), dtype=np.int64))
    for n in g.node:
        ins = [r(i) for i in n.input]
        if NOPAST and k == 0 and n.op_type == "GroupQueryAttention":
            ins[3] = ""
            ins[4] = ""
        c = helper.make_node(n.op_type, ins, [r(o) if o != "" else "" for o in n.output], name=p + n.name, domain=n.domain)
        c.attribute.extend(n.attribute)
        nodes.append(c)
    # this step's head slice at the last position: [1,1,GROUP]
    for nm, val in (("starts", [-1, k * GROUP]), ("ends", [2**62, (k + 1) * GROUP]), ("axes", [1, 2])):
        const(p + "l_" + nm, np.array(val, dtype=np.int64))
    lg = node("Slice", [r("logits"), p + "l_starts", p + "l_ends", p + "l_axes"], [p + "head"], p + "slice")
    if not NOTOPK:
        vals, idx = p + "tk_v", p + "tk_i"
        nodes.append(helper.make_node("TopK", [lg, k_const], [vals, idx], name=p + "topk", axis=2))
        kth = node("Slice", [vals, "kth_starts", "kth_ends", "kth_axes"], [p + "kth"], p + "kth_slice")
        below = node("Less", [lg, kth], [p + "below"], p + "below")
    # Gumbel-max over the logits at or above the k-th largest = sampling from softmax(top-k / T)
    for nm, val in (("n_starts", [k]), ("n_ends", [k + 1]), ("n_axes", [0])):
        const(p + nm, np.array(val, dtype=np.int64))
    gum = node("Slice", ["gumbel", p + "n_starts", p + "n_ends", p + "n_axes"], [p + "gum"], p + "noise_row")
    score = node("Add", [lg, gum], [p + "score"], p + "score")
    masked = score if NOTOPK else node("Where", [below, neg_inf, score], [p + "masked"], p + "mask")
    tok = p + "tok"
    nodes.append(helper.make_node("ArgMax", [masked], [tok], name=p + "argmax", axis=2, keepdims=0))
    # The code as a float for the output (int64 Cast/Concat/Reshape run on the CPU, one GPU round trip each): the
    # one-hot of the max, dotted with 0..2047. (Ties have probability ~0 with continuous noise.)
    nodes.append(helper.make_node("ReduceMax", [masked, "axes2"], [p + "rmax"], name=p + "rmax", keepdims=1))
    hit = node("Equal", [masked, p + "rmax"], [p + "hit"], p + "hit")
    hit_f = node("Cast", [hit], [p + "hit_f"], p + "hit_f", to=TensorProto.FLOAT)
    weighted = node("Mul", [hit_f, "arange"], [p + "weighted"], p + "weighted")
    nodes.append(helper.make_node("ReduceSum", [weighted, "axes2"], [p + "tok_f"], name=p + "tok_f", keepdims=0))
    toks.append(p + "tok_f")
    x = lookup(tok, k + 1)
    rows.append(x)
    for i, name in enumerate(present_names):
        infos.append(helper.make_tensor_value_info(f"{p}{name}", TensorProto.FLOAT, [1, 8, 2 + k, 128]))

codes = node("Concat", toks, ["codes"], "cat_codes", axis=1)
acc = rows[0]
for i, row in enumerate(rows[1:], 1):
    acc = node("Add", [acc, row], [f"sum{i}"], f"sum{i}")
nodes.append(helper.make_node("Identity", [acc], ["embed_sum"], name="embed_sum"))

graph_inputs = [
    helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, 1, H]),
    helper.make_tensor_value_info("c0", TensorProto.INT64, [1, 1]),
    helper.make_tensor_value_info("gumbel", TensorProto.FLOAT, [STEPS, 1, GROUP]),
]
if not NOPAST:
    graph_inputs += [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 8, 0, 128]) for n in past_names]
outputs = [
    helper.make_tensor_value_info("codes", TensorProto.FLOAT, [1, STEPS]),
    helper.make_tensor_value_info("embed_sum", TensorProto.FLOAT, [1, 1, H]),
]
all_inits = list(g.initializer) + new_inits
ng = helper.make_graph(nodes, "cp_frame", graph_inputs, outputs, initializer=all_inits, value_info=infos)
opsets = {o.domain: o for o in m.opset_import}
for o in emb.__class__ and onnx.load(emb_path, load_external_data=False).opset_import:
    opsets.setdefault(o.domain, o)
nm = helper.make_model(ng, opset_imports=list(opsets.values()))
nm.ir_version = max(m.ir_version, 10)
onnx.save(nm, dst)
print("saved", dst, "nodes", len(nodes), "opsets", [(d, o.version) for d, o in opsets.items()])
