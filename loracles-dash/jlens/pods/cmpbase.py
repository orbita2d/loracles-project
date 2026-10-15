import torch, glob, sys
B = {f.split('/')[-2]: torch.load(f, weights_only=False) for f in sorted(glob.glob(sys.argv[1] + '/means_*/base.pt'))}
k = list(B); r = B[k[0]]
for n, x in B.items():
    for s in ('train', 'test'):
        a = r[s]['sum'] / r[s]['n']; b = x[s]['sum'] / x[s]['n']
        d = (a - b).norm(dim=-1) / a.norm(dim=-1)
        print(n, s, x[s]['n'], 'rel diff vs', k[0], 'max %.2e mean %.2e' % (d.max(), d.mean()))
# compare to a LoRA effect size
v = torch.load(glob.glob(sys.argv[1] + '/means_dolphin/v2-dolphin.pt')[0], weights_only=False)
d = ((v['test']['sum'] / v['test']['n']) - (r['test']['sum'] / r['test']['n'])).norm(dim=-1) / (r['test']['sum'] / r['test']['n']).norm(dim=-1)
print('v2-dolphin vs base rel diff mean %.2e max %.2e' % (d.mean(), d.max()))
