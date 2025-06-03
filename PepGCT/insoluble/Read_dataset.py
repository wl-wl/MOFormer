import numpy as np
pos_data = np.load('/tmp/pycharm_project_pepGCT/insoluble/soluble.npz')
neg_data=np.load('/tmp/pycharm_project_pepGCT/insoluble/insoluble.npz')
pos_data = pos_data['arr_0']
neg_data=neg_data['arr_0']
# loaded_data_array2 = loaded_data['array2']
labels = np.concatenate(
    (
        np.ones((pos_data.shape[0], 1), dtype=pos_data.dtype),
        np.zeros((neg_data.shape[0], 1), dtype=pos_data.dtype),
    ),
    axis=0,
)
features = np.concatenate((pos_data, neg_data), axis=0)
pos_data_lengths = np.count_nonzero(pos_data, axis=1)
neg_data_lengths = np.count_nonzero(neg_data, axis=1)

print('Positive data', pos_data.shape[0], pos_data.shape[0]/(pos_data.shape[0]+ neg_data.shape[0])*100)
print('Negative data', neg_data.shape[0], neg_data.shape[0]/(pos_data.shape[0]+ neg_data.shape[0])*100)