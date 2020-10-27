

import pickle
import numpy as np
from pylab import plt
import seaborn as sns

data = pickle.load(open('tlm_test_result.pkl','rb'))

pert_scales = data['pert_scales']

# loop over forecasts
plt.figure()
for tlm, finitdiff in zip(data['tlm'],data['finitdiff']):
    plt.plot(pert_scales, tlm, label='tlm', alpha=0.1, color='black')
    plt.plot(pert_scales, finitdiff, label='finitdiff', alpha=0.1,color='red')

plt.xlabel('pert_scale')
plt.savefig('tlm_test_singlelines.png')
plt.xlim((-0.25,0.25))
plt.ylim((-0.004,0.004))
plt.savefig('tlm_test_singlelines_closeup.png')

plt.show()

# compute mean response over all forecasts
tlm = np.mean(np.array(data['tlm']),axis=0)
finitdiff = np.mean(np.array(data['finitdiff']),axis=0)

plt.figure()
plt.plot(pert_scales, tlm, label='tlm')
plt.plot(pert_scales, finitdiff, label='finitdiff')
plt.xlabel('pert_scale')
plt.legend()
sns.despine()
plt.savefig('tlm_test.svg')
plt.xlim((-0.1,0.1))
plt.ylim((-0.00005,0.00005))
plt.savefig('tlm_test_closeup.svg')