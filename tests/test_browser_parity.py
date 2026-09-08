"""Compare browser kernels to measured Python reference outputs, including odd sizes."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from fusion_agent.tools.fusion import wavelet_fusion, weighted_average_fusion
from fusion_agent.tools.metrics import calculate_metrics


@pytest.mark.skipif(shutil.which("node") is None, reason="Node needed for browser parity tests")
@pytest.mark.parametrize("shape", [(2, 2), (7, 9), (32, 40)])
def test_browser_python_parity(shape):
    rng = np.random.default_rng(42)
    v = rng.integers(0, 256, shape, dtype=np.uint8)
    i = rng.integers(0, 256, shape, dtype=np.uint8)
    payload = {
        "width": shape[1],
        "height": shape[0],
        "v": v.ravel().tolist(),
        "i": i.ravel().tolist(),
    }
    javascript = """
import {runFusion, metrics, wavelet, weighted} from './web/fusion.mjs';
let text=''; for await (const chunk of process.stdin) text+=chunk;
const p=JSON.parse(text), v={width:p.width,height:p.height,data:Uint8Array.from(p.v)},
 i={width:p.width,height:p.height,data:Uint8Array.from(p.i)};
const output=runFusion(v,i,0.5);
for (const item of Object.values(output.results)) item.image.data=Array.from(item.image.data);
output.identity=Array.from(wavelet(v,v).data);
let rejected=false; try {weighted(v,i,NaN)} catch {rejected=true};
output.rejectsNaN=rejected;
console.log(JSON.stringify(output));
"""
    process = subprocess.run(
        ["node", "--input-type=module", "-e", javascript],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    output = json.loads(process.stdout)
    assert output["rejectsNaN"]
    np.testing.assert_array_equal(output["identity"], v.ravel())
    references = {"weighted": weighted_average_fusion(v, i), "wavelet": wavelet_fusion(v, i)}
    for name, reference in references.items():
        result = output["results"][name]
        image = np.array(result["image"]["data"], dtype=np.uint8).reshape(shape)
        error = np.abs(image.astype(float) - reference.astype(float)).max()
        assert error <= (1 if name == "wavelet" else 0)
        for metric, value in calculate_metrics(image).model_dump().items():
            assert result["metrics"][metric] == pytest.approx(value, rel=1e-12, abs=1e-12)
