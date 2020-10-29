

import pickle
import numpy as np
from pylab import plt
import seaborn as sns



sns.set_palette('colorblind')
data = pickle.load(open('tlm_test_result.pkl','rb'))

pert_scales = data['pert_scales']

# loop over forecasts
plt.figure()
for tlm, finitdiff in zip(data['tlm'],data['finitdiff']):
    plt.plot(pert_scales, tlm, label='tlm', alpha=0.1, color='black')
    plt.plot(pert_scales, finitdiff, label='NN', alpha=0.1,color='red')

plt.xlabel('$\sigma$')
plt.ylabel("$y'$")
plt.savefig('plots/tlm_test_singlelines.png')
plt.xlim((-0.25,0.25))
plt.ylim((-0.004,0.004))
plt.savefig('plots/tlm_test_singlelines_closeup.png')

plt.show()

# compute mean response over all forecasts
tlm = np.mean(np.array(data['tlm']),axis=0)
finitdiff = np.mean(np.array(data['finitdiff']),axis=0)

plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(pert_scales, tlm, label='TLM')
plt.plot(pert_scales, finitdiff, label='NN')
plt.xlabel('$\sigma$')
plt.ylabel("$y'$")
plt.legend()
sns.despine()
plt.subplot(122)
plt.plot(pert_scales, tlm, label='TLM')
plt.plot(pert_scales, finitdiff, label='NN')
plt.xlabel('$\sigma$')
plt.ylabel("$y'$")
plt.legend()
sns.despine()
plt.xlim((-0.1,0.1))
plt.ylim((-0.00005,0.00005))
plt.tight_layout()
plt.savefig('plots/tlm_test.pdf', bbox='tight')