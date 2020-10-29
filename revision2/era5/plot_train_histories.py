


from pylab import plt
import pickle
import seaborn as sns


mem=1
ifile=f'train_histories/train_history_params_era5_2.5deg_weynetal_1979-2016_mem{mem}.pkl'

hists = pickle.load(open(ifile,'rb'))
# combine the two parts of the hist
val_loss = hists[0]['val_loss'] + hists[1]['val_loss']
train_loss = hists[0]['loss'] + hists[1]['loss']

plt.figure()
plt.plot(train_loss, label = 'training_loss', alpha=0.4)
plt.plot(val_loss, label = 'validation_loss', alpha=0.4)
plt.legend()
plt.xlabel('epoch')
plt.savefig(f'plots/train_history_mem{mem}.png')