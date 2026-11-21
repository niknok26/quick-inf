"""Checks export_inkling.py's graph on WebGPU: the cut-down checkpoint of check_inkling.py exported twice with each layer
application's output exposed (--debug), with QMoE (as the app runs it) for the op-check page (opcheck.html), and with
dense experts for ONNX Runtime's CPU, whose outputs (logits and hidden states of a 40-token prompt) are the expected
ones. Written to the dev server's model cache (tools/reference/.cache/micro/0/inkling_mini*).

    tools/export/.venv/bin/python tools/export/check_inkling_gpu.py MINI_DIR SCRATCH_DIR
    node tools/bench/bench.mjs --page opcheck 'x:cases=inkling_mini'
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from check_inkling import PROMPT, ids

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'reference' / '.cache' / 'micro' / '0'
PY = sys.executable


def export(src: Path, out: Path, *flags: str):
    subprocess.run([PY, str(HERE / 'export_inkling.py'), '--src', str(src), '--out', str(out), '--debug', *flags], check=True,
                   stdout=subprocess.DEVNULL)


def main(mini: Path, scratch: Path):
    import onnxruntime as ort

    gpu_dir = OUT / 'inkling_mini_qmoe'
    cpu_dir = scratch / 'inkling-mini-dense'
    export(mini, gpu_dir)
    export(mini, cpu_dir, '--dense-experts')

    session = ort.InferenceSession(str(cpu_dir / 'model.onnx'), providers=['CPUExecutionProvider'])
    tokens = ids()[:PROMPT]
    feeds = {'input_ids': np.array([tokens], np.int64), 'attention_mask': np.ones((1, PROMPT), np.int64),
             'position_ids': np.arange(PROMPT, dtype=np.int64)[None]}
    inputs = []
    for i in session.get_inputs():
        if i.name.startswith('past_key_values.'):
            feeds[i.name] = np.zeros((1, i.shape[1], 0, i.shape[3]), np.float16)
        elif i.name.startswith('past_conv.'):
            feeds[i.name] = np.zeros((1, i.shape[1], i.shape[2]), np.float32)
    outputs = [o.name for o in session.get_outputs() if o.name == 'logits' or o.name.startswith('/model/')]
    got = dict(zip(outputs, session.run(outputs, feeds)))

    name = 'inkling_mini'
    types = {np.float16: 'float16', np.float32: 'float32', np.int64: 'int64'}
    for k, v in feeds.items():
        if v.size:
            v.tofile(OUT / f'{name}.{k}.bin')
        inputs.append({'name': k, 'type': types[v.dtype.type], 'dims': list(v.shape)})
    specs = []
    for k, v in got.items():
        safe = k.replace('/', '_')
        v.astype(np.float32).tofile(OUT / f'{name}.{safe}.bin')
        specs.append({'name': k, 'file': safe, 'type': types[v.dtype.type], 'dims': list(v.shape)})
    spec = {'inputs': inputs, 'outputs': specs, 'model': 'inkling_mini_qmoe/model.onnx', 'externalData': ['model.onnx.data']}
    (OUT / f'{name}.json').write_text(json.dumps(spec))
    print(name, len(specs), 'outputs')


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
