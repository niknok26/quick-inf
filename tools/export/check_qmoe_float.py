"""Makes op-check cases (src/diagnostics/opcheck.ts) for QMoE with weights as small floats (weight_format "e2m1" and
"e4m3", Quick Inference's ONNX Runtime build) and the softcapped SwiGLU, from Inkling 2 Mini's own experts: layer 0
(NVFP4) and layer 20 (MXFP8), the first 16 experts, 8 chosen per token. The expected outputs are computed here in
float32 from the same codes and float16 scales. Token counts cover the fused path (1, 3) and the per-expert one (9,
40). Written to the dev server's model cache (tools/reference/.cache/micro/0/); prints the case names.

    tools/export/.venv/bin/python tools/export/check_qmoe_float.py
    node tools/bench/bench.mjs --page opcheck 'x:cases=NAMES'
"""

import numpy as np
from onnx import TensorProto, helper

from check_ggml import save
from export_inkling import E2M1, SRC, Checkpoint

EXPERTS = 16
K = 8
TOKENS = [1, 3, 9, 40]
GATE_CAP, UP_CAP = 4.0, 25.0


def layer(W: Checkpoint, L: int, kind: str):
    """(codes, float16 scales, values as the kernel decodes them [E, N, K] float32, format, bits, block)."""
    p = f'layers.{L}.mlp.experts.{kind}_weight'
    codes = W(p, native=True)[:EXPERTS].contiguous()
    scale = W(f'{p}_scale', native=True)[:EXPERTS]
    raw = codes.view(__import__('torch').uint8).numpy()
    if W.has(f'{p}_amax'):
        factor = (W(f'{p}_amax')[:EXPERTS] / W(f'{p}_outer_scale_denominator')).numpy()
        s16 = (scale.view(__import__('torch').float8_e4m3fn).float().numpy() * factor[:, None, None]).astype(np.float16)
        values = np.stack([E2M1[raw & 15], E2M1[raw >> 4]], -1).reshape(raw.shape[0], raw.shape[1], -1)
        block, fmt, bits = 16, 'e2m1', 4
    else:
        s16 = np.exp2(scale.view(__import__('torch').uint8).numpy().astype(np.float32) - 127).astype(np.float16)
        values = codes.view(__import__('torch').float8_e4m3fn).float().numpy()
        block, fmt, bits = 32, 'e4m3', 8
    w = (values.reshape(*s16.shape, block) * s16.astype(np.float32)[..., None]).reshape(values.shape)
    return raw, s16, w, fmt, bits, block


def main():
    W = Checkpoint(SRC)
    rng = np.random.default_rng(0)
    names = []
    for L in (0, 20):
        c1, s1, w1, fmt, bits, block = layer(W, L, 'w13')
        c2, s2, w2, _, _, _ = layer(W, L, 'w2')
        hidden = w1.shape[2]
        for t in TOKENS:
            name = f'qmoe_{fmt}_t{t}'
            x = (rng.standard_normal((t, hidden)) * 0.5).astype(np.float16)
            chosen = np.stack([rng.choice(EXPERTS, K, replace=False) for _ in range(t)])
            weights = rng.uniform(0.05, 1, (t, K))
            weights /= weights.sum(-1, keepdims=True)
            router = np.full((t, EXPERTS), -10000, np.float32)
            np.put_along_axis(router, chosen, np.log(weights), axis=1)
            router16 = router.astype(np.float16)
            # QMoE's softmax over the router input gives the weights back.
            probs = np.exp(router16.astype(np.float32) - router16.astype(np.float32).max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            y = np.zeros((t, hidden), np.float32)
            xf = x.astype(np.float32)
            for i in range(t):
                for e in chosen[i]:
                    h = (w1[e] @ xf[i]).astype(np.float16).astype(np.float32)
                    g, u = h[0::2], h[1::2]
                    act = GATE_CAP * np.tanh(g / GATE_CAP) / (1 + np.exp(-g)) * UP_CAP * np.tanh(u / UP_CAP)
                    out = w2[e] @ act.astype(np.float16).astype(np.float32)
                    y[i] += probs[i, e] * out.astype(np.float16).astype(np.float32)
            inits = [
                helper.make_tensor('fc1', TensorProto.UINT8, c1.shape, c1.tobytes(), raw=True),
                helper.make_tensor('fc1_scales', TensorProto.FLOAT16, s1.shape, s1.tobytes(), raw=True),
                helper.make_tensor('fc2', TensorProto.UINT8, c2.shape, c2.tobytes(), raw=True),
                helper.make_tensor('fc2_scales', TensorProto.FLOAT16, s2.shape, s2.tobytes(), raw=True),
            ]
            node = helper.make_node('QMoE', ['x', 'router', 'fc1', 'fc1_scales', '', 'fc2', 'fc2_scales'], ['y'], domain='com.microsoft',
                                    k=K, activation_type='swiglu', swiglu_fusion=1, expert_weight_bits=bits, block_size=block,
                                    weight_format=fmt, swiglu_gate_softcap=GATE_CAP, swiglu_up_softcap=UP_CAP, normalize_routing_weights=0)
            save(name, [node], inits, {'x': x, 'router': router16}, {'y': y.astype(np.float16)})
            names.append(name)
    print(','.join(names))


if __name__ == '__main__':
    main()
