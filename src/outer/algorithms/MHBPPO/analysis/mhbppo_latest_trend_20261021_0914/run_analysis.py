"""Run accepted read-only parsers with new output paths; retain earlier freezes."""
from pathlib import Path
import contextlib,importlib.util,io,json,sys
OUT=Path(__file__).resolve().parent

def module(path):
    spec=importlib.util.spec_from_file_location('analysis_stage',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.OUT=OUT
    return mod

stage=sys.argv[1] if len(sys.argv)>1 else 'freeze'
if stage=='freeze':
    mod=module(OUT.parent/'mhbppo_episode_reliability_20261020/analyze.py')
    mod.FROZEN=OUT/'frozen';mod.FROZEN.mkdir(exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):mod.main()
    r=json.loads((OUT/'results.json').read_text(encoding='utf8'))
    print(json.dumps({'freeze':r['freeze'],'windows':r['windows'],
        'blocks_after9000':[b for b in r['blocks500'] if b['range'][0]>=9001],
        'difference':r['recent_difference'],'validation':r['validation']},ensure_ascii=False))
else:
    filename={'audit':'audit_latest.py','steps':'inspect_steps.py'}[stage]
    mod=module(OUT.parent/'mhbppo_latest_trend_20261020_2106'/filename)
    mod.main()
