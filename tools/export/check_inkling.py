"""Checks export_inkling.py's graph against transformers' own Inkling 2 code (Inkling2LoopForCausalLM, float32, eager),
on a cut-down copy of the checkpoint small enough to run both here: layers 0-11 and 18-23 (the loop is 6-11, three
passes; blocks at 0, 6 and 12), the first 6 experts (all chosen, so float16 noise cannot flip a near tie), the first 32,768 tokens. The export is made with
--dense-experts --exact (experts as plain MatMuls, float16 linear layers and embeddings), so only the graph's structure
can differ; QMoE's float formats are checked on WebGPU separately. Compares the logits of a prompt, then of tokens
decoded one at a time against the cache.

    tools/export/.venv/bin/python tools/export/check_inkling.py mini DIR
    REFVENV/bin/python tools/export/check_inkling.py reference DIR ref.npz      # transformers with Inkling 2 (>= 5.24)
    tools/export/.venv/bin/python tools/export/export_inkling.py --src DIR --out ONNX --dense-experts --exact
    tools/export/.venv/bin/python tools/export/check_inkling.py compare ONNX ref.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / '.cache' / 'hf' / 'Inkling-2-Mini'
LAYERS = list(range(12)) + list(range(18, 24))
EXPERTS = 6
CHOSEN = 6
VOCAB = 32768
PROMPT = 40
DECODE = 4


def mini(dst: Path, vocab: int = VOCAB, chosen: int = CHOSEN, experts: int = EXPERTS):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    cfg = json.loads((SRC / 'config.json').read_text())['text_config']
    index = json.loads((SRC / 'model.safetensors.index.json').read_text())['weight_map']
    f = safe_open(str(SRC / 'model-00001-of-00001.safetensors'), framework='pt')
    P = 'model.llm.'
    new_of = {L: n for n, L in enumerate(LAYERS)}
    out = {}
    for name in index:
        if not name.startswith(P):
            continue
        key = name[len(P):]
        t = None
        if key.startswith('layers.'):
            L = int(key.split('.')[1])
            if L not in new_of:
                continue
            rest = key.split('.', 2)[2]
            t = f.get_tensor(name)
            if rest.startswith('mlp.experts.') and not rest.endswith('outer_scale_denominator'):
                t = t[:experts]
            elif rest.startswith('mlp.gate.'):
                t = t[:experts]
            key = f'layers.{new_of[L]}.{rest}'
        elif key.startswith('attn_res.attn_queries.'):
            L = int(key.rsplit('.', 1)[1]) + 1  # belongs to layer L
            if L not in new_of or new_of[L] == 0:
                continue
            key = f'attn_res.attn_queries.{new_of[L] - 1}'
        elif key.startswith('attn_res.mlp_queries.'):
            L = int(key.rsplit('.', 1)[1])
            if L not in new_of:
                continue
            key = f'attn_res.mlp_queries.{new_of[L]}'
        elif key.startswith('middle_cycle'):
            t = f.get_tensor(name)[:2 * 6]
        elif key in ('embed.weight', 'unembed.weight'):
            t = f.get_tensor(name)[:vocab]
        if t is None:
            t = f.get_tensor(name)
        out[P + key] = t.contiguous()
    dst.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst / 'model.safetensors'), metadata={'format': 'pt'})
    (dst / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {k: 'model.safetensors' for k in out}}))
    small = dict(cfg)
    small.update({
        'num_hidden_layers': len(LAYERS),
        'local_layer_ids': [new_of[L] for L in cfg['local_layer_ids'] if L in new_of],
        'loop_config': {**cfg['loop_config'], 'core_start': 6, 'core_end': 12},
        'n_routed_experts': experts,
        'num_experts_per_tok': chosen,
        'vocab_size': vocab,
        'unpadded_vocab_size': min(vocab, cfg['unpadded_vocab_size']),
        'model_type': 'inkling2_loop_text',
        'architectures': ['Inkling2LoopForCausalLM'],
        'eos_token_id': 0 if vocab == VOCAB else cfg.get('eos_token_id', 200006),
        'pad_token_id': 0 if vocab == VOCAB else 199999,
    })
    (dst / 'config.json').write_text(json.dumps(small, indent=1))
    print('wrote', dst, len(out), 'tensors')


def ids():
    rng = np.random.default_rng(0)
    return rng.integers(100, 32000, PROMPT + DECODE).tolist()


def reference(src: Path, out: Path):
    import torch
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(str(src), dtype=torch.float32, attn_implementation='eager')
    model.eval()
    tokens = ids()
    with torch.no_grad():
        res = model(torch.tensor([tokens[:PROMPT]]), use_cache=True, output_hidden_states=True)
        logits = [res.logits[0].numpy()]
        hidden = np.stack([h[0].numpy() for h in res.hidden_states])
        cache = res.past_key_values
        for i in range(DECODE):
            res = model(torch.tensor([[tokens[PROMPT + i]]]), past_key_values=cache, use_cache=True)
            cache = res.past_key_values
            logits.append(res.logits[0].numpy())
    np.savez(out, prompt=logits[0], decode=np.concatenate(logits[1:]), hidden=hidden)
    print('reference logits', logits[0].shape, 'decode', len(logits) - 1)


def compare(onnx_dir: Path, ref: Path, split: int = 25):
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_dir / 'model.onnx'), providers=['CPUExecutionProvider'])
    r = np.load(ref)
    tokens = ids()
    feeds = {}
    for i in session.get_inputs():
        if i.name.startswith('past_key_values.'):
            feeds[i.name] = np.zeros((1, i.shape[1], 0, i.shape[3]), np.float16)
        elif i.name.startswith('past_conv.'):
            feeds[i.name] = np.zeros((1, i.shape[1], i.shape[2]), np.float32)
    outputs = [o.name for o in session.get_outputs()]
    empty = dict(feeds)

    def reset():
        feeds.clear()
        feeds.update(empty)

    feeds_prompt = {}

    def step(new: list, start: int):
        if start == 0:
            feeds_prompt.update(feeds)
            feeds_prompt.update({'input_ids': np.array([new], np.int64), 'position_ids': np.arange(len(new), dtype=np.int64)[None],
                                 'attention_mask': np.ones((1, len(new)), np.int64)})
        feeds['input_ids'] = np.array([new], np.int64)
        feeds['position_ids'] = np.arange(start, start + len(new), dtype=np.int64)[None]
        feeds['attention_mask'] = np.ones((1, start + len(new)), np.int64)
        res = dict(zip(outputs, session.run(outputs, feeds)))
        for name in list(feeds):
            if name.startswith('past_key_values.'):
                feeds[name] = res[name.replace('past_key_values.', 'present.')]
            elif name.startswith('past_conv.'):
                feeds[name] = res[name.replace('past_conv.', 'present_conv.')]
        # The export leaves the final soft cap to the app; transformers' logits have it.
        return 30 * np.tanh(res['logits'][0].astype(np.float32) / 30)

    def report(label, got, want):
        diff = np.abs(got - want)
        top = (got.argmax(-1) == want.argmax(-1)).mean()
        p = np.exp(want - want.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        q = np.exp(got - got.max(-1, keepdims=True))
        q /= q.sum(-1, keepdims=True)
        kl = (p * (np.log(p + 1e-30) - np.log(q + 1e-30))).sum(-1)
        print(f'{label}: max |diff| {diff.max():.4f}, mean {diff.mean():.5f}, logit scale {np.abs(want).mean():.3f}, '
              f'same top token {top:.0%}, KL mean {kl.mean():.2e} max {kl.max():.2e}', flush=True)

    report('prompt', step(tokens[:PROMPT], 0), r['prompt'])
    if split:
        # The same prompt again in two passes, the second with the first's cache behind it (a later chat turn, or a
        # prompt longer than one chunk).
        reset()
        first = step(tokens[:split], 0)
        second = step(tokens[split:PROMPT], split)
        report(f'prompt in passes of {split} + {PROMPT - split}', np.concatenate([first, second]), r['prompt'])
    hidden = [o for o in outputs if o.endswith('/output_0') and ('/model/layers.' in o or 'embed_norm' in o)]
    if hidden and 'hidden' in r:
        got = session.run(hidden, {**feeds_prompt})
        for i, (name, h) in enumerate(zip(hidden, got)):
            want = r['hidden'][i] if i < len(r['hidden']) else None
            if want is None:
                break
            err = np.abs(h[0] - want).max() / (np.abs(want).max() + 1e-9)
            print(f'  {name}: relative max diff {err:.2e} (ref max {np.abs(want).max():.2f})')
    got = [step([tokens[PROMPT + i]], PROMPT + i) for i in range(DECODE)]
    report('decode', np.concatenate(got), r['decode'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', choices=['mini', 'reference', 'compare'])
    parser.add_argument('a', type=Path)
    parser.add_argument('b', type=Path, nargs='?')
    args = parser.parse_args()
    if args.cmd == 'mini':
        mini(args.a) if args.b is None else mini(args.a, vocab=201024, chosen=8, experts=16)
    elif args.cmd == 'reference':
        reference(args.a, args.b)
    else:
        compare(args.a, args.b)
    sys.exit(0)
