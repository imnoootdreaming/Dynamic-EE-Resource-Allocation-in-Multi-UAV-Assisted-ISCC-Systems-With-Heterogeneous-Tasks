from pathlib import Path
R=Path.cwd();O=R/'analysis/mosek_runs_20261010';B=O/'before'
files=['src/inner/parameters.py','src/inner/fusion_pc3p.py','src/inner/pc3p.py','src/inner/gc3p.py','src/inner/main.py','src/inner/compare_pc3p_gc3p.py']
s=(O/'apply_changes.py').read_text(encoding='utf8')
start=s.index("edit(files[1],'    return cached")
body=s[start:]
body=body.replace("'    return cached\\n','    model = cached[0]", "'        _MODEL_CACHE[key] = cached\\n    return cached\\n','        _MODEL_CACHE[key] = cached\\n    model = cached[0]",1)
def edit(name,old,new,count=1):
 p=R/name;t=p.read_text(encoding='utf8');assert t.count(old)==count,(name,old[:80],t.count(old));p.write_text(t.replace(old,new),encoding='utf8')
exec(compile(body,str(O/'apply_changes.py'),'exec'))