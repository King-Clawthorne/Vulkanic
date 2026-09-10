"""GPU regressions for independently configured rainbow multiple scattering."""
import argparse, copy, importlib.util, json, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('hdr',Path(__file__).with_name('compare-hdr.py'))
hdr=importlib.util.module_from_spec(spec);spec.loader.exec_module(hdr)
p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path);a=p.parse_args()
root=Path('cmake-build-ninja/rain-multiple-validation');root.mkdir(exist_ok=True)
base=json.loads(Path('config/path_tracer_config.json').read_text())
base['render'].update(width=160,height=90,vsync=False)
base['camera'].update(initialLookAt=[-0.35,2.16,-10.25],fovYDegrees=65)
base['rainbow'].update(enabled=1,scatteringOrders=1,multipleScatteringSamples=2,multipleScatteringSteps=8)
base['sky']['spectralConstants'].update(VIEW_STEPS=2,Samples=1,secondarySamples=1,SCATTERING_ORDERS=1,GROUND_ALBEDO=0)
results={}
def run(name,c,exe=Path('cmake-build-ninja/Vulkanic.exe')):
    prefix=root/name;config=prefix.with_suffix('.json');config.write_text(json.dumps(c,indent=2))
    with prefix.with_suffix('.log').open('w') as log:
        r=subprocess.run([str(exe.resolve()),'--benchmark','24','--warmup','8','--config',str(config.resolve()),'--capture-hdr',str(prefix.with_suffix('.hdrbin').resolve())],stdout=log,stderr=subprocess.STDOUT,timeout=180)
    text=prefix.with_suffix('.log').read_text()
    if r.returncode or 'Validation Error' in text:raise RuntimeError(name+' failed; see log')
    dims,v=hdr.load(prefix.with_suffix('.hdrbin'))
    mean=sum(0.2126*v[i]+0.7152*v[i+1]+0.0722*v[i+2] for i in range(0,len(v),4))/(len(v)//4)
    results[name]={'luminance':mean,'timings':[l for l in text.splitlines() if 'mean=' in l]}
    print(name,results[name],flush=True);return v,mean
one,_=run('order-1',base)
if a.baseline:
    old,_=run('baseline',base,a.baseline)
    if one!=old:raise RuntimeError('Order 1 must retain baseline HDR exactly')
previous=results['order-1']['luminance']
for n in range(2,5):
    c=copy.deepcopy(base);c['rainbow']['scatteringOrders']=n
    values,mean=run('order-'+str(n),c)
    if mean<=previous:raise RuntimeError('Added scattering order must add energy')
    previous=mean
c=copy.deepcopy(base);c['rainbow'].update(scatteringCoefficient=0,extinctionCoefficient=0,scatteringOrders=4)
zero,_=run('zero-rain',c);c['rainbow']['enabled']=0;disabled,_=run('disabled-rain',c)
if zero!=disabled:raise RuntimeError('Zero-rain limit changed')
c=copy.deepcopy(base);c['sky']['spectralConstants'].update(BETA_R_550=0,BETA_M=0)
c['rainbow'].update(scatteringCoefficient=1e-6,extinctionCoefficient=1e-6)
_,a1=run('thin-1',c);c['rainbow']['scatteringOrders']=2;_,a2=run('thin-2',c)
c['rainbow'].update(scatteringCoefficient=2e-6,extinctionCoefficient=2e-6)
_,b2=run('thin-double-2',c);c['rainbow']['scatteringOrders']=1;_,b1=run('thin-double-1',c)
ratio=(b2-b1)/(a2-a1)
if not 3.8<ratio<4.05:raise RuntimeError('Thin second order must scale quadratically: '+str(ratio))
results['thin_density_ratio']=ratio
(root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print('PASS: unchanged order 1, finite HDR, orders 2-4, zero-rain limit, quadratic second order.')
