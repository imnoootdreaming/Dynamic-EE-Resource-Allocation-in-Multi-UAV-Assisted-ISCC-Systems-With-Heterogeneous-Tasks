"""Reuse the accepted U61 parser, preserving its frozen evidence directory."""
from pathlib import Path
import contextlib, importlib.util, io, json

OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'mhbppo_episode_reliability_20261020'
spec=importlib.util.spec_from_file_location('u61_analyze',OLD/'analyze.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.OUT=OUT;base.FROZEN=OUT/'frozen';base.FROZEN.mkdir(exist_ok=True)
with contextlib.redirect_stdout(io.StringIO()):base.main()
r=json.loads((OUT/'results.json').read_text(encoding='utf8'))
print(json.dumps({'freeze':r['freeze'],'windows':r['windows'],
    'blocks_after6500':[b for b in r['blocks500'] if b['range'][0]>=6501],
    'difference':r['recent_difference'],'validation':r['validation']},ensure_ascii=False))
