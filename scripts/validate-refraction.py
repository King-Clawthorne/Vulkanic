"""Deterministic refraction GPU checks; optionally compare a 5c79591 executable."""
import argparse, copy, json, subprocess, re, sys
sys.dont_write_bytecode = True
from pathlib import Path


import importlib.util
spec=importlib.util.spec_from_file_location('hdr',Path(__file__).with_name('compare-hdr.py'))
hdr=importlib.util.module_from_spec(spec);spec.loader.exec_module(hdr)
p=argparse.ArgumentParser()
p.add_argument('--exe',type=Path,default=Path('cmake-build-ninja/Vulkanic.exe'))
p.add_argument('--baseline',type=Path)
p.add_argument('--allow-existing-present-warning',action='store_true',help='Record the restored baseline semaphore warning without treating it as a refraction failure')
p.add_argument('--output',type=Path,default=Path('cmake-build-ninja/refraction-validation'))
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
base=json.loads(Path('config/path_tracer_config.json').read_text())
base['render'].update(width=160,height=90,vsync=True)
base['camera'].update(initialLookAt=[-0.35,2.16,-10.25],fovYDegrees=65)
base['rainbow']['enabled']=1
base['sky']['spectralConstants'].update(VIEW_STEPS=2,Samples=1,secondarySamples=2,SCATTERING_ORDERS=1,SEA_LEVEL_REFRACTIVITY=0)
results={}
def run(name,c,exe=None):
    prefix=a.output/name
    config=prefix.with_suffix('.json');config.write_text(json.dumps(c,indent=2)+'\n')
    with prefix.with_suffix('.log').open('w') as log:
        r=subprocess.run([str((exe or a.exe).resolve()),'--benchmark','8','--warmup','4','--config',str(config.resolve()),'--capture-hdr',str(prefix.with_suffix('.hdrbin').resolve())],stdout=log,stderr=subprocess.STDOUT,timeout=180)
    text=prefix.with_suffix('.log').read_text()
    errors=re.findall(r'Validation Error: \[ ([^ ]+) \]',text)
    known='VUID-vkQueueSubmit-pSignalSemaphores-00067'
    if r.returncode or any(e != known or not a.allow_existing_present_warning for e in errors): raise RuntimeError(f'{name}: see log')
    dims,values=hdr.load(prefix.with_suffix('.hdrbin'))
    mean=sum(values[0::4])/len(values[0::4])
    results[name]={'validation_errors':errors,'mean_red':mean,'timings':[l for l in text.splitlines() if 'mean=' in l]}
    print(name,results[name],flush=True)
    return values
zero=run('zero',base)
if a.baseline:
    reference=run('baseline',base,a.baseline)
    errors=[abs(x-y) for x,y in zip(zero,reference)]
    if any(e>2e-6+2e-4*abs(y) for e,y in zip(errors,reference)): raise RuntimeError('Zero refraction differs from baseline')
    results['baseline_comparison']={'max_error':max(errors),'bitwise_equal':zero==reference}
for order in range(1,5):
    c=copy.deepcopy(base);c['sky']['spectralConstants'].update(SEA_LEVEL_REFRACTIVITY=0.000277,SCATTERING_ORDERS=order)
    changed=run(f'refracted-order-{order}',c)
    if order==1 and changed==zero: raise RuntimeError('Refraction has no effect')
c=copy.deepcopy(base);c['rainbow']['enabled']=0
c['sky']['spectralConstants'].update(BETA_R_550=0,BETA_M=0,GROUND_ALBEDO=0,SUN_DIRECTION=[1,-0.0069814304,0])
c['camera'].update(initialLookAt=[1,2.00174533,-10],fovYDegrees=2)
hidden=run('geometric-horizon',c)
c['sky']['spectralConstants']['SEA_LEVEL_REFRACTIVITY']=0.000277
visible=run('refracted-horizon',c)
if max(hidden[0::4])>1e-8 or max(visible[0::4])<=1e-4: raise RuntimeError('Refracted horizon visibility failed')
(a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print('PASS: finite HDR, orders 1-4, zero-refraction reference and refracted horizon. Validation warnings are recorded in results.json.')
