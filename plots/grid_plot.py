import pickle
import matplotlib.pyplot as plt
import tikzplotlib as tikz

with open('grid2__problemmemory_cellGRU_hidden_min12_hidden_max136_hidden_step16_time_min60_time_max60'
          '_time_step10_compression_modereset_batch_size200_lr0.001_n_train600000_n_test10000.pkl', 'rb') as gruFile:
    gru_result_lst = pickle.load(gruFile)

gru_test_acc_lst = []
gru_hidden_lst = []
gru_pt_lst = []
gru_time_lst = []
for exp in gru_result_lst:
    gru_test_acc_lst.append(exp[3])
    gru_time_lst.append(exp[1])
    gru_hidden_lst.append(exp[0])
    gru_pt_lst.append(exp[4])

with open('grid__problemmemory_cellWaveletGRU_hidden_min12_hidden_max136_hidden_step16_time_min60'
          '_time_max60_time_step10_compression_modereset_batch_size200_lr0.001_n_train600000'
          '_n_test10000_infcuda.pkl', 'rb') as wavegruFile:
    wave_gru_result_lst = pickle.load(wavegruFile)

wave_gru_test_acc_lst = []
wave_gru_hidden_lst = []
wave_gru_pt_lst = []
wave_gru_time_lst = []
for exp in wave_gru_result_lst:
    wave_gru_test_acc_lst.append(exp[3])
    wave_gru_time_lst.append(exp[1])
    wave_gru_hidden_lst.append(exp[0])
    wave_gru_pt_lst.append(exp[4])


plt.plot(gru_pt_lst[:-1], gru_test_acc_lst[:-1])
plt.plot(wave_gru_pt_lst[:-1], wave_gru_test_acc_lst[:-1])
plt.legend(['GRU', 'WaveletGRU'])
plt.xlabel('parameters')
plt.ylabel('accuracy')
# plt.show()
tikz.save('pt_gru_wavegru.tex', standalone=True)
plt.show()