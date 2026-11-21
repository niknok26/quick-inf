"""Exports the text model of Thinking Machines' Inkling 2 Mini (Inkling2LoopForConditionalGeneration, Apache 2.0) to an
ONNX decoder, from the official checkpoint. Its experts are copied bit for bit: NVFP4 (E2M1 codes, a float8 scale per
16 weights and a per-expert factor) in layers 0-19 and MXFP8 (float8 E4M3 values, a power-of-two scale per 32) in
layers 20-23, run by QMoE's weight_format "e2m1" and "e4m3" (Quick Inference's ONNX Runtime build,
https://github.com/mdanyalmalik/onnxruntime-web-kernels). The vision tower and the multi-token-prediction head are left
out.

The text model (transformers' modeling_inkling2_loop.py), 24 layers of which 6-17 run three times (48 layer applications
per token, each with its own KV cache and convolution states):
- every layer: block attention residuals (learned queries attending over the RMS-normalised block states, one block per
  6 layers, and the current hidden state) choose each sublayer's input; attention; a short causal convolution (4 taps,
  depthwise, plus the input, over sqrt(5)) after attention and after the MoE; the residual restarts at each block.
- attention: 24 query heads and 8 KV heads of 128, no rotary embedding. Queries are scaled per channel and clipped to an
  L2 norm of 16 * sqrt(2); keys go through a short convolution (static on sliding layers, per-token taps on the full
  ones) and the same clip; values through a convolution and an RMS normalisation. A position bias per head and backward
  distance (0-1023) comes from the hidden state (16 features, clipped, times a learned 16 x 1024 table), a per-token
  sink logit from another projection; the output is gated by sigmoid(gate_proj(x)). Full layers (5, 11, 17, 23) also
  scale queries and bias features by 1 + 0.1 * log(max((position + 1) / 8192, 1)); the others use a 1,024-token window.
- the MoE: 128 experts, 8 per token by sigmoid score plus a bias, weighted by the plain scores normalised and times 8;
  the activation is 4 * tanh(gate / 4) * sigmoid(gate) * 25 * tanh(up / 25).
- passes 2 and 3 of the loop start again from its input, with messages from the previous pass (learned queries over its
  last block state and output) gated into each loop layer's query, key, value and MoE inputs.
- logits are before the final soft cap (30 * tanh(logits / 30)), which the app applies.

The residual stream, norms, convolutions, residual attentions, gates and the router run in float32; activations and
the KV cache are float16. Linear layers are 4-bit MatMulNBits (block 32, round to nearest), the embeddings a 4-bit
GatherBlockQuantized, all from the checkpoint's bfloat16.

States: past_key_values.{a}.{key,value} [1, 8, past, 128] and past_conv.{a} [1, 3, 6656] (the last 3 inputs of the
key, value, post-attention and post-MoE convolutions, in that order) for each layer application a in execution order.

    tools/export/.venv/bin/python tools/export/export_inkling.py [--src DIR] [--out DIR] [--dense-experts] [--exact]
"""

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper
from safetensors import safe_open

from onnx_export import F16, F32, I64, MS, U8, Graph, Settings, matmul_nbits, sigmoid_router

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = HERE / '.cache' / 'hf' / 'Inkling-2-Mini'
REPO = 'Inkling-2-Mini-ONNX-WebGPU'
SOURCE = 'thinkingmachines/Inkling-2-Mini'
REVISION = 'ed40e7da625f594fc9c4990f5f66be24873c22d7'
P = 'model.llm.'
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)
CONV_TAPS = 4
REL_EXTENT = 1024
LOG_FLOOR = 8192
LOG_ALPHA = 0.1


class Checkpoint:
    def __init__(self, folder: Path):
        index = json.loads((folder / 'model.safetensors.index.json').read_text())['weight_map']
        self.files = {f: safe_open(str(folder / f), framework='pt') for f in set(index.values())}
        self.index = index

    def __call__(self, name: str, native=False) -> torch.Tensor:
        t = self.files[self.index[P + name]].get_tensor(P + name)
        return t if native else t.float()

    def has(self, name: str) -> bool:
        return P + name in self.index


class Builder:
    def __init__(self, g: Graph, W: Checkpoint, cfg: dict, dense_experts: bool, exact: bool):
        self.g, self.W, self.cfg = g, W, cfg
        self.dense_experts, self.exact = dense_experts, exact
        self.override: Path | None = None
        self.D = cfg['hidden_size']
        self.eps = cfg['rms_norm_eps']
        self.consts = {}

    # Basic pieces.
    def node(self, op, inputs, name, **kw):
        return self.g.node(op, inputs, name, **kw)

    def c(self, name, values, dtype=I64):
        # Constants are named by use; a name already made holds the same value (stored under the name share_identical
        # gave it).
        if name not in self.consts:
            self.consts[name] = self.g.const(name, values, dtype)
        return self.consts[name]

    def f32(self, x, name):
        return self.node('Cast', [x], name, to=F32)

    def f16(self, x, name):
        return self.node('Cast', [x], name, to=F16)

    def linear(self, x16: str, w: torch.Tensor, name: str) -> str:
        """x @ w.T, float16 in and out: 4-bit MatMulNBits, or float16 with --exact (for checking)."""
        if self.exact:
            wt = self.g.init(f'{name}.weight_T', w.float().T.contiguous().numpy().astype(np.float16), F16)
            return self.node('MatMul', [x16, wt], name)
        return matmul_nbits(self.g, x16, w, name)

    def small_linear(self, x: str, w: torch.Tensor, name: str, dtype=F16) -> str:
        """x @ w.T with w kept as it is (small projections whose outputs steer attention: sinks, position bias)."""
        np_type = np.float16 if dtype == F16 else np.float32
        wt = self.g.init(f'{name}.weight_T', w.T.contiguous().numpy().astype(np_type), dtype)
        return self.node('MatMul', [x, wt], name)

    def rms(self, x32: str, w: np.ndarray | None, name: str, eps: float | None = None) -> str:
        """RMS normalisation over the last axis in float32, with a weight or without (w None)."""
        size = self.D if w is None else w.shape[0]
        weight = np.ones(size, np.float32) if w is None else w.astype(np.float32)
        wn = self.g.init(f'{name}.weight', weight, F32)
        return self.node('SimplifiedLayerNormalization', [x32, wn], name, axis=-1, epsilon=eps or self.eps, stash_type=1)

    def head_rms(self, x32: str, heads: int, dim: int, name: str) -> str:
        """RMS normalisation (no weights) of each head of x [B, S, heads, dim]."""
        wn = self.g.init(f'{name}.weight', np.ones(dim, np.float32), F32)
        return self.node('SimplifiedLayerNormalization', [x32, wn], name, axis=-1, epsilon=self.eps, stash_type=1)

    def l2_clip(self, x32: str, threshold: float, name: str) -> str:
        """Each vector of the last axis scaled down to `threshold` when its L2 norm is larger."""
        norm = self.node('ReduceL2', [x32, self.c(f'{name}/axes', [-1])], f'{name}/ReduceL2', keepdims=1)
        denom = self.node('Add', [norm, self.c(f'{name}/eps', 1e-6, F32)], f'{name}/Add')
        factor = self.node('Div', [self.c(f'{name}/threshold', threshold, F32), denom], f'{name}/Div')
        factor = self.node('Min', [factor, self.c(f'{name}/one', 1.0, F32)], f'{name}/Min')
        return self.node('Mul', [x32, factor], f'{name}/Mul')

    # The model's parts.
    def residual_attention(self, queries: np.ndarray, states: list, name: str) -> list:
        """Learned queries [nq, D] attending over the RMS-normalised states (each [B, S, D] float32); one output per
        query. Keys and values are the normalised states, the scale 1 / sqrt(D)."""
        nq = queries.shape[0]
        stacked = [self.node('Unsqueeze', [s, self.c(f'{name}/axis', [2])], f'{name}/Unsqueeze_{i}') for i, s in enumerate(states)]
        stacked = stacked[0] if len(stacked) == 1 else self.node('Concat', stacked, f'{name}/Concat', axis=2)
        normed = self.rms(stacked, None, f'{name}/norm', 1e-6)
        if len(states) == 1:
            # One state: every query takes all of it.
            return [self.node('Squeeze', [normed, self.c(f'{name}/squeeze_axis', [2])], f'{name}/Squeeze')] * nq
        qt = self.g.init(f'{name}.query_T', (queries.T / math.sqrt(self.D)).astype(np.float32), F32)
        scores = self.node('MatMul', [normed, qt], f'{name}/scores')  # [B, S, n, nq]
        probs = self.node('Softmax', [scores], f'{name}/Softmax', axis=2)
        probs = self.node('Transpose', [probs], f'{name}/Transpose', perm=[0, 1, 3, 2])  # [B, S, nq, n]
        out = self.node('MatMul', [probs, normed], f'{name}/MatMul')  # [B, S, nq, D]
        if nq == 1:
            return [self.node('Squeeze', [out, self.c(f'{name}/squeeze_axis', [2])], f'{name}/Squeeze')]
        parts = self.node('Split', [out, self.c(f'{name}/split', [1] * nq)], f'{name}/Split', outputs=nq, axis=2)
        return [self.node('Squeeze', [p, self.c(f'{name}/squeeze_axis_{i}', [2])], f'{name}/Squeeze_{i}') for i, p in enumerate(parts)]

    def message_gate(self, L: int, kind: str, h32: str, m32: str, name: str) -> str:
        """(1 - tanh(forget)) * h + tanh(gain) * m, both from a rank-64 projection of [rms(h), rms(m)]."""
        p = f'layers.{L}.controllers.{kind}'
        both = self.node('Concat', [self.rms(h32, None, f'{name}/norm_h'), self.rms(m32, None, f'{name}/norm_m')], f'{name}/Concat', axis=-1)
        down = self.small_linear(both, self.W(f'{p}.down_weight'), f'{name}/down', F32)
        forget = self.node('Tanh', [self.small_linear(down, self.W(f'{p}.forget_weight'), f'{name}/forget', F32)], f'{name}/forget/Tanh')
        gain = self.node('Tanh', [self.small_linear(down, self.W(f'{p}.gain_weight'), f'{name}/gain', F32)], f'{name}/gain/Tanh')
        kept = self.node('Mul', [self.node('Sub', [self.c(f'{name}/one', 1.0, F32), forget], f'{name}/Sub'), h32], f'{name}/keep')
        return self.node('Add', [kept, self.node('Mul', [gain, m32], f'{name}/message')], f'{name}/Add')

    def conv(self, x32: str, state: str, taps: np.ndarray, name: str, dynamic: str | None = None, heads: int = 0) -> tuple:
        """A causal depthwise convolution of x [B, S, C] (float32) over the state's 3 earlier inputs [B, 3, C], plus x,
        over sqrt(taps + 1) (static: taps [4, C] by backward distance) or sqrt(2 * 4 + 1) (dynamic: per-token taps from
        `dynamic` [B, S, heads * 4], one set per head, added to the static ones). Returns (output, new state)."""
        ctx = self.node('Concat', [state, x32], f'{name}/Concat', axis=1)  # [B, S + 3, C]
        steps = self.node('Gather', [self.node('Shape', [x32], f'{name}/Shape'), self.c(f'{name}/one_index', 1)], f'{name}/Gather', axis=0)
        new_state = self.node('Slice', [ctx, self.c(f'{name}/minus3', [-3]), self.c(f'{name}/end', [2**62]), self.c(f'{name}/axis', [1])], f'{name}/state')
        total = None
        if dynamic is not None:
            per_head = self.node('Reshape', [dynamic, self.c(f'{name}/dyn_shape', [0, 0, heads, 1, CONV_TAPS])], f'{name}/dyn/Reshape')
            channels = taps.shape[1]
        for lag in range(CONV_TAPS):
            start = self.node('Sub', [self.c(f'{name}/start{lag}', [CONV_TAPS - 1 - lag]), self.c(f'{name}/zero{lag}', [0])], f'{name}/start_{lag}')
            stop = self.node('Add', [start, self.node('Unsqueeze', [steps, self.c(f'{name}/ax0_{lag}', [0])], f'{name}/steps_{lag}')], f'{name}/stop_{lag}')
            window = self.node('Slice', [ctx, start, stop, self.c(f'{name}/axis_{lag}', [1])], f'{name}/window_{lag}')
            coef = self.g.init(f'{name}.tap_{lag}', taps[lag].astype(np.float32), F32)
            if dynamic is not None:
                d = self.node('Gather', [per_head, self.c(f'{name}/lag{lag}', [lag])], f'{name}/dyn_{lag}', axis=4)  # [B, S, heads, 1, 1]
                d = self.node('Reshape', [d, self.c(f'{name}/dyn_shape_{lag}', [0, 0, heads, 1])], f'{name}/dyn_{lag}/Reshape')
                d = self.node('Expand', [d, self.c(f'{name}/dyn_expand_{lag}', [1, 1, heads, channels // heads])], f'{name}/dyn_{lag}/Expand')
                d = self.node('Reshape', [d, self.c(f'{name}/dyn_flat_{lag}', [0, 0, channels])], f'{name}/dyn_{lag}/Flatten')
                coef = self.node('Add', [d, coef], f'{name}/coef_{lag}')
            term = self.node('Mul', [window, coef], f'{name}/term_{lag}')
            total = term if total is None else self.node('Add', [total, term], f'{name}/sum_{lag}')
        divisor = math.sqrt(2 * CONV_TAPS + 1) if dynamic is not None else math.sqrt(CONV_TAPS + 1)
        out = self.node('Add', [total, x32], f'{name}/residual')
        out = self.node('Mul', [out, self.c(f'{name}/scale', 1 / divisor, F32)], f'{name}/scale')
        return out, new_state

    def attention(self, L: int, full: bool, qh: str, kh: str, vh: str, past: dict, state: dict, masks: dict, name: str) -> str:
        """qh, kh, vh: the normalised query, key and value inputs (float32). Returns the output (float32), and fills
        past['present'] with the new cache and state['new'] with the convolutions' new states."""
        W, cfg = self.W, self.cfg
        p = f'layers.{L}.attn'
        heads, kv_heads, dim = cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg['head_dim']
        q16, k16, v16 = self.f16(qh, f'{name}/q_in/Cast'), self.f16(kh, f'{name}/k_in/Cast'), self.f16(vh, f'{name}/v_in/Cast')

        # Queries: per-channel scale, L2 clip (and the log scale on full layers), [B, heads, S, dim].
        q = self.f32(self.node('Reshape', [self.linear(q16, W(f'{p}.wq_du.weight'), f'{name}/q_proj'), self.c(f'{name}/q_shape', [0, 0, heads, dim])], f'{name}/q/Reshape'), f'{name}/q/Cast')
        q = self.node('Mul', [q, self.g.init(f'{name}.q_scale', W(f'{p}.q_l2_clip.scale').numpy(), F32)], f'{name}/q/scale')
        q = self.l2_clip(q, cfg['qk_l2_clip_threshold'], f'{name}/q/clip')
        if full:
            q = self.node('Mul', [q, masks['log_scale']], f'{name}/q/log_scale')
        q = self.node('Transpose', [self.f16(q, f'{name}/q/Cast16')], f'{name}/q/Transpose', perm=[0, 2, 1, 3])

        # Keys and values: projection, convolution, then clip (keys) or RMS normalisation (values).
        def kv(kind: str, h16: str, h32: str, conv_state: str):
            x = self.f32(self.linear(h16, W(f'{p}.w{kind}_dv.weight'), f'{name}/{kind}_proj'), f'{name}/{kind}/Cast')
            if full:
                taps = W(f'{p}.{kind}_sconv.static_w').numpy()  # [4, C] by backward distance
                dynamic = self.small_linear(h32, W(f'{p}.{kind}_sconv.weight_proj'), f'{name}/{kind}/weight_proj', F32)
                out, new = self.conv(x, conv_state, taps, f'{name}/{kind}/conv', dynamic, kv_heads)
            else:
                w = W(f'{p}.{kind}_sconv.weight').numpy()[:, 0, :]  # [C, 4], tap j weighs the input 3 - j back
                out, new = self.conv(x, conv_state, w[:, ::-1].T.copy(), f'{name}/{kind}/conv')
            out = self.node('Reshape', [out, self.c(f'{name}/{kind}/shape', [0, 0, kv_heads, dim])], f'{name}/{kind}/Reshape')
            out = self.l2_clip(out, cfg['qk_l2_clip_threshold'], f'{name}/{kind}/clip') if kind == 'k' else self.head_rms(out, kv_heads, dim, f'{name}/{kind}/norm')
            out = self.node('Transpose', [self.f16(out, f'{name}/{kind}/Cast16')], f'{name}/{kind}/Transpose', perm=[0, 2, 1, 3])
            return out, new

        k, state['k'] = kv('k', k16, kh, state['k_past'])
        v, state['v'] = kv('v', v16, vh, state['v_past'])
        key = self.node('Concat', [past['key'], k], f'{name}/key/Concat', axis=2)
        value = self.node('Concat', [past['value'], v], f'{name}/value/Concat', axis=2)
        past['present'] = (key, value)

        # Scores [B, heads, S, T]: query groups of 3 share a KV head.
        group = heads // kv_heads
        qg = self.node('Reshape', [q, self.c(f'{name}/qg_shape', [1, kv_heads, -1, dim])], f'{name}/qg')  # [1, 8, 3S, dim]
        kt = self.node('Transpose', [key], f'{name}/key/Transpose', perm=[0, 1, 3, 2])
        scores = self.node('MatMul', [qg, kt], f'{name}/scores/MatMul')
        scores = self.node('Reshape', [scores, self.c(f'{name}/scores_shape', [1, heads, -1, 0])], f'{name}/scores/Reshape')
        scores = self.node('Mul', [self.f32(scores, f'{name}/scores/Cast'), self.c(f'{name}/scale', 1 / math.sqrt(dim), F32)], f'{name}/scores/scale')

        # Position bias: per head and backward distance, from the query input.
        rel = self.small_linear(q16, W(f'{p}.wr_du.weight'), f'{name}/rel_proj')
        rel = self.f32(self.node('Reshape', [rel, self.c(f'{name}/rel_shape', [0, 0, heads, cfg['d_rel']])], f'{name}/rel/Reshape'), f'{name}/rel/Cast')
        rel = self.node('Mul', [rel, self.g.init(f'{name}.rel_scale', W(f'{p}.rel_l2_clip.scale').numpy(), F32)], f'{name}/rel/scale')
        rel = self.l2_clip(rel, cfg['rel_l2_clip_threshold'], f'{name}/rel/clip')
        if full:
            rel = self.node('Mul', [rel, masks['log_scale']], f'{name}/rel/log_scale')
        table = self.g.init(f'{name}.rel_table', W(f'{p}.rel_logits_proj.proj').numpy(), F32)  # [16, 1024]
        rel = self.node('MatMul', [rel, table], f'{name}/rel/MatMul')  # [B, S, heads, 1024]
        rel = self.node('Transpose', [rel], f'{name}/rel/Transpose', perm=[0, 2, 1, 3])
        bias = self.node('GatherElements', [rel, masks['rel_index']], f'{name}/rel/GatherElements', axis=3)
        bias = self.node('Where', [masks['rel_valid'], bias, self.c(f'{name}/zero', 0.0, F32)], f'{name}/rel/Where')
        scores = self.node('Add', [scores, bias], f'{name}/scores/bias')
        scores = self.node('Add', [scores, masks['full' if full else 'sliding']], f'{name}/scores/mask')

        # A sink logit per head and token takes part in the softmax with no value.
        sink = self.small_linear(q16, W(f'{p}.attn_sink_proj.weight'), f'{name}/sink_proj')  # [B, S, heads]
        sink = self.node('Unsqueeze', [self.node('Transpose', [self.f32(sink, f'{name}/sink/Cast')], f'{name}/sink/Transpose', perm=[0, 2, 1]), self.c(f'{name}/sink/axis', [3])], f'{name}/sink/Unsqueeze')
        probs = self.node('Softmax', [self.node('Concat', [scores, sink], f'{name}/sink/Concat', axis=3)], f'{name}/Softmax', axis=3)
        probs = self.node('Slice', [probs, self.c(f'{name}/p_start', [0]), self.c(f'{name}/p_end', [-1]), self.c(f'{name}/p_axis', [3])], f'{name}/probs')
        probs = self.node('Reshape', [self.f16(probs, f'{name}/probs/Cast'), self.c(f'{name}/probs_shape', [1, kv_heads, -1, 0])], f'{name}/probs/Reshape')
        out = self.node('MatMul', [probs, value], f'{name}/MatMul')  # [1, 8, 3S, dim]
        out = self.node('Reshape', [out, self.c(f'{name}/out_shape', [1, heads, -1, dim])], f'{name}/out/Reshape')
        out = self.node('Transpose', [out], f'{name}/out/Transpose', perm=[0, 2, 1, 3])
        out = self.node('Reshape', [out, self.c(f'{name}/out_flat', [0, 0, heads * dim])], f'{name}/out/Flatten')
        del group

        gate = self.node('Sigmoid', [self.linear(q16, W(f'{p}.gate_attn.weight'), f'{name}/gate_proj')], f'{name}/gate/Sigmoid')
        out = self.linear(self.node('Mul', [out, gate], f'{name}/gate/Mul'), W(f'{p}.wo_ud.weight'), f'{name}/o_proj')
        return self.f32(out, f'{name}/out/Cast')

    def moe(self, L: int, x32: str, name: str) -> str:
        W = self.W
        p = f'layers.{L}.mlp'
        x16 = self.node('Reshape', [self.f16(x32, f'{name}/Cast'), self.c(f'{name}/flat', [-1, self.D])], f'{name}/Reshape')
        weight, bias = W(f'{p}.gate.weight'), W(f'{p}.gate.bias')
        experts = weight.shape[0]
        k = self.cfg['num_experts_per_tok']
        route = self.cfg['route_scale']
        probs = sigmoid_router(self.g, x16, weight, bias, k, f'{name}/router')
        if self.dense_experts:
            out = self.dense_moe(L, x16, probs, experts, route, name)
        else:
            fc1, s1, fmt, bits, block = self.expert_tensors(L, 'w13', 1.0)
            fc2, s2, _, _, _ = self.expert_tensors(L, 'w2', route)
            out = self.node('QMoE', [x16, probs, fc1, s1, '', fc2, s2], f'{name}/QMoE', domain=MS, k=k, activation_type='swiglu',
                            swiglu_fusion=1, expert_weight_bits=bits, block_size=block, weight_format=fmt,
                            swiglu_gate_softcap=self.cfg['mlp_gate_softcap'], swiglu_up_softcap=self.cfg['mlp_up_softcap'],
                            normalize_routing_weights=0)
        shape = self.node('Shape', [x32], f'{name}/Shape')
        return self.f32(self.node('Reshape', [out, shape], f'{name}/out/Reshape'), f'{name}/out/Cast')

    def expert_tensors(self, L: int, kind: str, factor: float):
        """One layer's stacked expert weights and scales as stored (the three passes of a loop layer share them: see
        share_identical)."""
        W = self.W
        p = f'layers.{L}.mlp.experts.{kind}_weight'
        codes = W(p, native=True)
        scale = W(f'{p}_scale', native=True)
        names = (f'model.layers.{L}.experts.{kind}.weight', f'model.layers.{L}.experts.{kind}.scales')
        if W.has(f'{p}_amax'):
            # NVFP4: codes [E, N, K / 2], scales float8 [E, N, K / 16] times amax / denominator per expert.
            per_expert = (W(f'{p}_amax') / W(f'{p}_outer_scale_denominator')).numpy() * factor
            s = scale.view(torch.float8_e4m3fn).float().numpy() * per_expert[:, None, None]
            fmt, bits, block = 'e2m1', 4, 16
        else:
            # MXFP8: float8 values [E, N, K], scales 2^(s - 127) per 32.
            s = np.exp2(scale.view(torch.uint8).numpy().astype(np.float32) - 127) * factor
            fmt, bits, block = 'e4m3', 8, 32
        s16 = s.astype(np.float16)
        assert np.isfinite(s16).all(), f'{p}: scales overflow float16'
        lost = np.count_nonzero((s16 == 0) & (s != 0))
        assert lost == 0, f'{p}: {lost} scales underflow float16'
        return self.g.init(names[0], codes.view(torch.uint8).numpy(), U8), self.g.init(names[1], s16, F16), fmt, bits, block

    def dense_moe(self, L: int, x16: str, probs16: str, experts: int, route: float, name: str) -> str:
        """For checking on ONNX Runtime's CPU: every expert as plain float32 MatMuls, weighted by the router's
        (softmax over its log scores) probabilities, which are 0 for unchosen experts."""
        w13, w2 = dequant_experts(self.W, L, 'w13'), dequant_experts(self.W, L, 'w2')  # [E, N, K]
        if self.override:
            # Checking a re-quantization: the gate and up experts as it decodes them (check_inkling_2bit.py).
            w13 = np.load(self.override / f'layer{L}.w13.npy')
        x = self.f32(x16, f'{name}/dense/x')
        h = self.node('MatMul', [x, self.g.init(f'{name}.w13_T', w13.transpose(0, 2, 1).copy(), F32)], f'{name}/dense/fc1')  # [E, T, 2I]
        gate = self.node('Slice', [h, self.c(f'{name}/g0', [0]), self.c(f'{name}/g1', [2**62]), self.c(f'{name}/gax', [2]), self.c(f'{name}/gst', [2])], f'{name}/dense/gate')
        up = self.node('Slice', [h, self.c(f'{name}/u0', [1]), self.c(f'{name}/u1', [2**62]), self.c(f'{name}/uax', [2]), self.c(f'{name}/ust', [2])], f'{name}/dense/up')
        gc, uc = self.cfg['mlp_gate_softcap'], self.cfg['mlp_up_softcap']
        capped = lambda t, cap, n: self.node('Mul', [self.node('Tanh', [self.node('Div', [t, self.c(f'{n}/cap', cap, F32)], f'{n}/Div')], f'{n}/Tanh'), self.c(f'{n}/capm', cap, F32)], f'{n}/Mul')
        act = self.node('Mul', [self.node('Mul', [capped(gate, gc, f'{name}/dense/gc'), self.node('Sigmoid', [gate], f'{name}/dense/sig')], f'{name}/dense/gs'), capped(up, uc, f'{name}/dense/uc')], f'{name}/dense/act')
        y = self.node('MatMul', [act, self.g.init(f'{name}.w2_T', w2.transpose(0, 2, 1).copy(), F32)], f'{name}/dense/fc2')  # [E, T, D]
        pr = self.node('Softmax', [self.f32(probs16, f'{name}/dense/p32')], f'{name}/dense/Softmax', axis=-1)  # [T, E]
        pr = self.node('Unsqueeze', [self.node('Transpose', [pr], f'{name}/dense/pT', perm=[1, 0]), self.c(f'{name}/dense/pax', [2])], f'{name}/dense/pU')
        y = self.node('ReduceSum', [self.node('Mul', [y, pr], f'{name}/dense/weigh'), self.c(f'{name}/dense/rax', [0])], f'{name}/dense/sum', keepdims=0)
        y = self.node('Mul', [y, self.c(f'{name}/dense/route', route, F32)], f'{name}/dense/route')
        return self.f16(y, f'{name}/dense/Cast')


def dequant_experts(W: Checkpoint, L: int, kind: str) -> np.ndarray:
    """A layer's experts [E, N, K] in float32, as transformers' Inkling2LoopDequantizeExperts decodes them."""
    p = f'layers.{L}.mlp.experts.{kind}_weight'
    codes = W(p, native=True).view(torch.uint8).numpy()
    scale = W(f'{p}_scale', native=True)
    if W.has(f'{p}_amax'):
        factor = (W(f'{p}_amax') / W(f'{p}_outer_scale_denominator')).numpy()
        values = np.stack([E2M1[codes & 15], E2M1[codes >> 4]], -1).reshape(codes.shape[0], codes.shape[1], -1)
        s = scale.view(torch.float8_e4m3fn).float().numpy() * factor[:, None, None]
        return (values.reshape(*s.shape, 16) * s[..., None]).reshape(values.shape)
    values = torch.from_numpy(codes).view(torch.float8_e4m3fn).float().numpy()
    s = np.exp2(scale.view(torch.uint8).numpy().astype(np.float32) - 127)
    return (values.reshape(*s.shape, 32) * s[..., None]).reshape(values.shape)


def share_identical(g: Graph):
    """Makes g.init return the existing initializer for an array identical to one already stored (type, shape and
    bytes): the looped layers' three passes then share their weights."""
    store = g.init
    seen = {}

    def init(name: str, array: np.ndarray, dtype: int) -> str:
        # (np.ascontiguousarray would make a 0-d scalar 1-d.)
        array = np.asarray(array)
        key = (dtype, array.shape, hashlib.sha1(array.tobytes()).hexdigest())
        if key not in seen:
            seen[key] = store(name, array, dtype)
        return seen[key]

    g.init = init


def layer_plan(cfg: dict) -> tuple:
    """(layer, pass) per application in execution order, and the loop's bounds."""
    start, end = cfg['loop_config']['core_start'], cfg['loop_config']['core_end']
    passes = cfg['loop_config']['num_passes']
    plan = [(L, 0) for L in range(start)]
    for p in range(passes):
        plan += [(L, p) for L in range(start, end)]
    plan += [(L, 0) for L in range(end, cfg['num_hidden_layers'])]
    return plan, start, end, passes


def masks(b: Builder, heads: int, window: int) -> dict:
    """Masks and position-bias indices from position_ids [1, S] and the attention mask's length T."""
    n = '/model/masks'
    pos = b.node('Reshape', ['position_ids', b.c(f'{n}/col', [-1, 1])], f'{n}/pos')  # [S, 1]
    total = b.node('Gather', [b.node('Shape', ['attention_mask'], f'{n}/Shape'), b.c(f'{n}/i1', 1)], f'{n}/total', axis=0)
    keys = b.node('Range', [b.c(f'{n}/r0', 0), total, b.c(f'{n}/r1', 1)], f'{n}/keys')  # [T]
    dist = b.node('Sub', [pos, keys], f'{n}/dist')  # [S, T]
    causal = b.node('GreaterOrEqual', [dist, b.c(f'{n}/zero', 0)], f'{n}/causal')
    near = b.node('Less', [dist, b.c(f'{n}/window', window)], f'{n}/near')
    in_window = b.node('And', [causal, near], f'{n}/in_window')
    neg = b.c(f'{n}/neg', -1e30, F32)
    zero = b.c(f'{n}/fzero', 0.0, F32)
    full = b.node('Where', [causal, zero, neg], f'{n}/full')
    sliding = b.node('Where', [in_window, zero, neg], f'{n}/sliding')
    near_rel = b.node('Less', [dist, b.c(f'{n}/extent', REL_EXTENT)], f'{n}/near_rel')
    rel_valid = b.node('And', [causal, near_rel], f'{n}/rel_valid')
    index = b.node('Clip', [dist, b.c(f'{n}/c0', 0), b.c(f'{n}/c1', REL_EXTENT - 1)], f'{n}/rel_clip')
    shape = b.node('Concat', [b.c(f'{n}/lead', [1, heads]), b.node('Shape', [dist], f'{n}/dist_shape')], f'{n}/index_shape', axis=0)
    index = b.node('Expand', [index, shape], f'{n}/rel_index')
    # 1 + 0.1 * log(max((position + 1) / 8192, 1)), per token: [1, S, 1, 1].
    ratio = b.node('Div', [b.node('Cast', [b.node('Add', ['position_ids', b.c(f'{n}/p1', 1)], f'{n}/p_plus')], f'{n}/p32', to=F32), b.c(f'{n}/floor', float(LOG_FLOOR), F32)], f'{n}/ratio')
    log = b.node('Log', [b.node('Max', [ratio, b.c(f'{n}/one', 1.0, F32)], f'{n}/Max')], f'{n}/Log')
    log_scale = b.node('Add', [b.node('Mul', [log, b.c(f'{n}/alpha', LOG_ALPHA, F32)], f'{n}/alpha_mul'), b.c(f'{n}/one_b', 1.0, F32)], f'{n}/log_scale_1')
    log_scale = b.node('Unsqueeze', [log_scale, b.c(f'{n}/ls_axes', [2, 3])], f'{n}/log_scale')
    return {'full': full, 'sliding': sliding, 'rel_valid': rel_valid, 'rel_index': index, 'log_scale': log_scale}


def build(src: Path, out: Path, dense_experts=False, exact=False, debug=False, embed_exact=False, w13_override: Path | None = None):
    cfg = json.loads((src / 'config.json').read_text())
    cfg = cfg.get('text_config', cfg)
    W = Checkpoint(src)
    out.mkdir(parents=True, exist_ok=True)
    g = Graph(out / 'model.onnx.data')
    share_identical(g)
    b = Builder(g, W, cfg, dense_experts, exact)
    b.override = w13_override
    D, kv_heads, dim = cfg['hidden_size'], cfg['num_key_value_heads'], cfg['head_dim']
    local = set(cfg['local_layer_ids'])
    plan, loop_start, loop_end, passes = layer_plan(cfg)
    block_every = cfg['n_layers_per_attn_res_block']
    kv_ch = kv_heads * dim
    conv_channels = [kv_ch, kv_ch, D, D]
    vocab = W('embed.weight', native=True).shape[0]
    unpadded = min(cfg['unpadded_vocab_size'], W('unembed.weight', native=True).shape[0])

    g.inputs = [
        helper.make_tensor_value_info('input_ids', I64, ['batch_size', 'sequence_length']),
        helper.make_tensor_value_info('attention_mask', I64, ['batch_size', 'total_sequence_length']),
        helper.make_tensor_value_info('position_ids', I64, ['batch_size', 'sequence_length']),
    ]
    g.outputs = [helper.make_tensor_value_info('logits', F16, ['batch_size', 'sequence_length', unpadded])]
    for a in range(len(plan)):
        for kind in ('key', 'value'):
            g.inputs.append(helper.make_tensor_value_info(f'past_key_values.{a}.{kind}', F16, ['batch_size', kv_heads, 'past_sequence_length', dim]))
            g.outputs.append(helper.make_tensor_value_info(f'present.{a}.{kind}', F16, ['batch_size', kv_heads, 'total_sequence_length', dim]))
        g.inputs.append(helper.make_tensor_value_info(f'past_conv.{a}', F32, ['batch_size', CONV_TAPS - 1, sum(conv_channels)]))
        g.outputs.append(helper.make_tensor_value_info(f'present_conv.{a}', F32, ['batch_size', CONV_TAPS - 1, sum(conv_channels)]))
    m = masks(b, cfg['num_attention_heads'], cfg['sliding_window_size'])

    # The embedding first, at the start of the weight file; RMS-normalised with its weight.
    if exact or embed_exact:
        table = g.init('model.embed_tokens.weight', W('embed.weight').numpy().astype(np.float16), F16)
        embedded = b.f32(g.node('Gather', [table, 'input_ids'], '/model/embed_tokens/Gather', axis=0), '/model/embed_tokens/Cast')
    else:
        embedded = b.f32(embedding(g, W, vocab, D, Settings.bits), '/model/embed_tokens/Cast')
    h = b.rms(embedded, W('embed_norm.weight').numpy(), '/model/embed_tokens/embed_norm')
    if debug:
        g.outputs.append(helper.make_tensor_value_info(h, F32, None))

    attn_queries = {i + 1: W(f'attn_res.attn_queries.{i}').numpy().reshape(-1, D) for i in range(cfg['num_hidden_layers'] - 1)}
    mlp_queries = {i: W(f'attn_res.mlp_queries.{i}').numpy().reshape(-1, D) for i in range(cfg['num_hidden_layers'])}
    # Message queries per loop layer: query, key, value, MoE (see transformers' Inkling2LoopMergeMessageQueries).
    qm = W('middle_cycle_cross_pass_queries').numpy().reshape(-1, 2, D)
    kvm = W('middle_cycle_cross_pass_kv_queries').numpy().reshape(-1, 2, D)
    message_queries = np.concatenate([qm[:, :1], kvm, qm[:, 1:]], axis=1)  # [loop layers, 4, D]

    blocks = []
    pre_loop = None
    previous = None  # (block states, output) of the previous pass
    for a, (L, pas) in enumerate(plan):
        name = f'/model/layers.{a}'
        print(f'application {a}: layer {L}, pass {pas}', flush=True)
        if L == loop_start:
            if pas == 0:
                pre_loop = (h, list(blocks))
            else:
                h, blocks = pre_loop[0], list(pre_loop[1])
        full = L not in local
        looped = loop_start <= L < loop_end
        messages = None
        if looped and pas > 0:
            messages = b.residual_attention(message_queries[L - loop_start], previous[0] + [previous[1]], f'{name}/message_input')
        boundary = L % block_every == 0

        past_conv = f'past_conv.{a}'
        bounds = np.cumsum([0] + conv_channels)
        state = {}
        for idx, key in enumerate(('k_past', 'v_past', 'attn_past', 'mlp_past')):
            state[key] = b.node('Slice', [past_conv, b.c(f'{name}/conv_s{idx}', [int(bounds[idx])]), b.c(f'{name}/conv_e{idx}', [int(bounds[idx + 1])]), b.c(f'{name}/conv_ax{idx}', [2])], f'{name}/past_conv_{idx}')
        past = {'key': f'past_key_values.{a}.key', 'value': f'past_key_values.{a}.value'}

        block_residual = h
        p = f'layers.{L}'
        if full:
            if L > 0:
                qx, kx, vx = b.residual_attention(attn_queries[L], blocks + [h], f'{name}/attention_input')
            else:
                qx = kx = vx = h
            if messages is not None:
                qx = b.message_gate(L, 'attn', qx, messages[0], f'{name}/q_message_gate')
                kx = b.message_gate(L, 'k', kx, messages[1], f'{name}/k_message_gate')
                vx = b.message_gate(L, 'v', vx, messages[2], f'{name}/v_message_gate')
            qh = b.rms(qx, W(f'{p}.q_input_norm.weight').numpy(), f'{name}/q_input_layernorm')
            kh = b.rms(kx, W(f'{p}.k_input_norm.weight').numpy(), f'{name}/k_input_layernorm')
            vh = b.rms(vx, W(f'{p}.v_input_norm.weight').numpy(), f'{name}/v_input_layernorm')
        else:
            x = b.residual_attention(attn_queries[L], blocks + [h], f'{name}/attention_input')[0] if L > 0 else h
            norm_w = W(f'{p}.attn_norm.weight').numpy()
            if messages is None:
                qh = kh = vh = b.rms(x, norm_w, f'{name}/input_layernorm')
            else:
                qh = b.rms(b.message_gate(L, 'attn', x, messages[0], f'{name}/q_message_gate'), norm_w, f'{name}/q_input_layernorm')
                kh = b.rms(b.message_gate(L, 'k', x, messages[1], f'{name}/k_message_gate'), norm_w, f'{name}/k_input_layernorm')
                vh = b.rms(b.message_gate(L, 'v', x, messages[2], f'{name}/v_message_gate'), norm_w, f'{name}/v_input_layernorm')
        attn = b.attention(L, full, qh, kh, vh, past, state, m, f'{name}/attn')
        attn, state['attn'] = b.conv(attn, state['attn_past'], W(f'{p}.attn_sconv.weight').numpy()[:, 0, ::-1].T.copy(), f'{name}/post_attention_convolution')
        layer_blocks = blocks + [block_residual] if boundary else blocks
        residual = attn if boundary else b.node('Add', [block_residual, attn], f'{name}/residual')
        hm = b.residual_attention(mlp_queries[L], layer_blocks + [residual], f'{name}/mlp_input')[0]
        if messages is not None:
            hm = b.message_gate(L, 'mlp', hm, messages[3], f'{name}/mlp_message_gate')
        mlp = b.moe(L, b.rms(hm, W(f'{p}.mlp_norm.weight').numpy(), f'{name}/post_attention_layernorm'), f'{name}/mlp')
        mlp, state['mlp'] = b.conv(mlp, state['mlp_past'], W(f'{p}.mlp_sconv.weight').numpy()[:, 0, ::-1].T.copy(), f'{name}/post_mlp_convolution')
        new_h = b.node('Add', [residual, mlp], f'{name}/output')

        key, value = past['present']
        g.nodes.append(helper.make_node('Identity', [key], [f'present.{a}.key'], name=f'{name}/present_key'))
        g.nodes.append(helper.make_node('Identity', [value], [f'present.{a}.value'], name=f'{name}/present_value'))
        g.nodes.append(helper.make_node('Concat', [state['k'], state['v'], state['attn'], state['mlp']], [f'present_conv.{a}'], name=f'{name}/present_conv', axis=2))

        if debug:
            g.outputs.append(helper.make_tensor_value_info(new_h, F32, None))
        if boundary:
            blocks = blocks + [h]
        h = new_h
        if looped and L == loop_end - 1:
            # The pass's last block state (after the loop's input) and output feed the next pass's messages.
            previous = (blocks[len(pre_loop[1]) + 1:], h)

    final = b.residual_attention(W('attn_res.output_query').numpy().reshape(1, D), blocks + [h], '/model/final_input')[0]
    final = b.f16(b.rms(final, W('norm.weight').numpy(), '/model/norm'), '/model/norm/Cast')
    unembed = W('unembed.weight', native=True)[:unpadded]
    logits = b.linear(final, unembed, '/lm_head/MatMul')
    g.nodes.append(helper.make_node('Identity', [logits], ['logits'], name='/lm_head/Identity'))
    g.close()

    # Every value a node reads must exist.
    known = {i.name for i in g.inputs} | {t.name for t in g.inits} | {''}
    for n in g.nodes:
        missing = [x for x in n.input if x not in known]
        assert not missing, f'{n.name} reads {missing}, which nothing makes'
        known.update(n.output)
    graph = helper.make_graph(g.nodes, 'Inkling-2-Mini', g.inputs, g.outputs, g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21), helper.make_opsetid(MS, 1)], ir_version=10)
    onnx.save(model, str(out / 'model.onnx'))
    print('saved', out, g.offset, flush=True)


def embedding(g: Graph, W: Checkpoint, vocab: int, hidden: int, bits: int) -> str:
    """The token embeddings as a GatherBlockQuantized (symmetric, block 32, codes plus the default zero point: 8 for
    4 bits, 128 for 8)."""
    table = W('embed.weight', native=True)
    block = 32
    top = 2 ** (bits - 1) - 1
    codes = np.empty((vocab, hidden * bits // 8), np.uint8)
    scales = np.empty((vocab, hidden // block), np.float16)
    rows = 8192
    for start in range(0, vocab, rows):
        w = table[start:start + rows].float().numpy().reshape(-1, hidden // block, block)
        s = np.abs(w).max(-1, keepdims=True) / top
        q = np.clip(np.rint(w / np.where(s == 0, 1, s)), -top - 1, top).astype(np.int16) + top + 1
        q = q.reshape(q.shape[0], hidden)
        codes[start:start + rows] = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8) if bits == 4 else q.astype(np.uint8)
        scales[start:start + rows] = s[..., 0].astype(np.float16)
    data = g.init(f'model.embed_tokens.weight_Q{bits}', codes, U8)
    sc = g.init('model.embed_tokens.weight_scale', scales, F16)
    return g.node('GatherBlockQuantized', [data, 'input_ids', sc], '/model/embed_tokens/GatherBlockQuantized', domain=MS,
                  bits=bits, block_size=block, gather_axis=0, quantize_axis=1)


def finish(out: Path):
    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja', 'generation_config.json', 'config.json'):
        shutil.copy(SRC / name, out / name)
    (out / 'README.md').write_text(MODEL_CARD)


MODEL_CARD = f"""---
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
base_model:
- {SOURCE}
tags:
- onnx
- webgpu
- quick-inference
---

# {REPO}

The text model of Thinking Machines Lab's [Inkling 2 Mini](https://huggingface.co/{SOURCE}) (26.4B parameters, 3.4B
active) converted to ONNX for running in a web browser with ONNX Runtime Web's WebGPU backend, as used by Quick Inference.

## Changes from the original

- Made from the official checkpoint ([{SOURCE}](https://huggingface.co/{SOURCE}) at `{REVISION[:8]}`). The experts are
  copied bit for bit: NVFP4 in layers 0-19 (E2M1 codes; each float8 block scale times its expert's
  `amax / outer_scale_denominator` is stored as one float16 scale) and MXFP8 in layers 20-23 (float8 values; the
  power-of-two block scales as float16). The MoE's routing scale (8) is folded into the down projections' scales.
- They run in `com.microsoft.QMoE` with `weight_format="e2m1"` / `"e4m3"` and the softcapped SwiGLU
  (`swiglu_gate_softcap=4`, `swiglu_up_softcap=25`), which only Quick Inference's build of ONNX Runtime Web has
  (v1.30.0-qi.6 or later, [onnxruntime-web-kernels](https://github.com/mdanyalmalik/onnxruntime-web-kernels)).
- The other weights (attention, embeddings, output layer), bfloat16 in the original, are quantised to 8 bits (round to
  nearest, symmetric, block size 32): at 4 bits the model's outputs drifted far from the original's.
- Text only: the vision tower and the multi-token-prediction head are left out.
- Converted to an ONNX decoder by Quick Inference's own exporter (`tools/export/export_inkling.py` in
  [Quick Inference](https://github.com/mdanyalmalik/quick-inf)), following transformers' Inkling 2 code
  (`modeling_inkling2_loop.py`) and checked against it (`check_inkling.py`). Layers 6-17 run three times, so the graph
  has 48 layer applications, each with its own cache: inputs `input_ids`, `attention_mask`, `position_ids`,
  `past_key_values.{{i}}.{{key,value}}` [1, 8, past, 128] (float16) and `past_conv.{{i}}` [1, 3, 6656] (float32: the
  last 3 inputs of the key, value, post-attention and post-MoE convolutions); outputs `logits`, `present.{{i}}.*`,
  `present_conv.{{i}}`.
- `logits` are **before** the final soft cap: apply `30 * tanh(logits / 30)` before sampling, as Quick Inference does.
- Activations and the KV cache are float16; the residual stream, norms, convolutions, residual attentions, message
  gates and the router run in float32.

These are modified files: their outputs are close to, but not identical to, the original model's.

## Licence

Apache 2.0, as the original. Thinking Machines' [acceptable use policy](https://thinkingmachines.ai/model-acceptable-use-policy/) applies to the model.
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=Path, default=SRC)
    parser.add_argument('--out', type=Path, default=ROOT / 'exports' / REPO)
    parser.add_argument('--dense-experts', action='store_true', help='experts as plain MatMuls (for checking on the CPU)')
    parser.add_argument('--exact', action='store_true', help='float16 linear layers instead of 4-bit (for checking)')
    parser.add_argument('--debug', action='store_true', help='also output the hidden state after each layer application')
    parser.add_argument('--bits', type=int, default=8, help='linear layers and embeddings: 4 or 8 bits')
    parser.add_argument('--embed-exact', action='store_true', help='float16 embeddings')
    parser.add_argument('--w13-override', type=Path, help='with --dense-experts: gate/up experts from DIR/layer{L}.w13.npy')
    args = parser.parse_args()
    Settings.bits = args.bits
    build(args.src, args.out, args.dense_experts, args.exact, args.debug, args.embed_exact, args.w13_override)
    if args.src == SRC:
        finish(args.out)
