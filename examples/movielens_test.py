from py_mf_maxnorm import PSG
from time import perf_counter as pc
from itertools import count
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

""" Obtain data: "Movielens" """
s_time = pc()

# Warning: other movielens sets (e.g. 20M) need a change to param: 'delimiter'
data = np.loadtxt('ml-1m/ratings.dat',
                  dtype={'names': ('userId', 'movieId', 'rating'),
                         'formats': ('i4', 'i4', 'f4')},
                  skiprows=1,
                  usecols=(0,1,2),
                  delimiter='::')

print('n ratings: ', data.shape[0])
print('n users: ', np.unique(data['userId']).shape[0])
print('n items: ', np.unique(data['movieId']).shape[0])
print('time (read data): ', pc() - s_time)

TRAIN_SIZE = int(data.shape[0] * 0.9)  # 90%/10%

"""" Preprocess data: map to [0,n-1] """
s_time = pc()
user_map, user_map_inv = {}, {}
item_map, item_map_inv = {}, {}
user_counter = count(0)
item_counter = count(0)

for u, v, r in data:
    if u not in user_map:
        new_u_id = next(user_counter)
        user_map[u] = new_u_id
        user_map_inv[new_u_id] = u
    if v not in item_map:
        new_v_id = next(item_counter)
        item_map[v] = new_v_id
        item_map_inv[new_v_id] = v

mapped_u = np.vectorize(user_map.__getitem__)(data['userId'])
mapped_v = np.vectorize(item_map.__getitem__)(data['movieId'])
print('time (id-mapping): ', pc() - s_time)

""" Split data """
rand_perm = np.random.permutation(len(mapped_u))
u_train = mapped_u[rand_perm[:TRAIN_SIZE]]
v_train = mapped_v[rand_perm[:TRAIN_SIZE]]
r_train = data['rating'][rand_perm[:TRAIN_SIZE]]

u_test = mapped_u[rand_perm[TRAIN_SIZE:]]
v_test = mapped_v[rand_perm[TRAIN_SIZE:]]
r_test = data['rating'][rand_perm[TRAIN_SIZE:]]

r_train_mean = np.mean(r_train)
r_train -= r_train_mean
r_test -= r_train_mean

print('Train: ', u_train.shape[0])
print('Test: ', u_test.shape[0])

""" Train """
# Warning: hyper-param tuning needed!
psg = PSG(1, len(np.unique(mapped_u)),
             len(np.unique(mapped_v)),
             10, 1.6, 0, 0.001,
             u_train, v_train, r_train)
psg.set_testset(u_test, v_test, r_test)
psg.optimize(50, 0.025, 0.99)

""" Obtain factors """
# L, R = psg.get_L(), psg.get_R()
# print(L)  # numpy-array
# print(R)  # """

""" Plot convergence """

train_metrics_hist = psg.get_train_metrics_history()
test_metrics_hist = psg.get_test_metrics_history()

N_EPOCHS = len(train_metrics_hist[0])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(N_EPOCHS), train_metrics_hist[0], '--', label='Train RMSE', c='g')
ax.plot(np.arange(N_EPOCHS), train_metrics_hist[1], '--', label='Train MAE', c='b')
ax.plot(np.arange(N_EPOCHS), test_metrics_hist[0], label='Test RMSE', c='g')
ax.plot(np.arange(N_EPOCHS), test_metrics_hist[1], label='Test MAE', c='b')

min_rmse_epoch = np.argsort(test_metrics_hist[0])
min_mae_epoch = np.argsort(test_metrics_hist[1])
min_rmse_val = test_metrics_hist[0][min_rmse_epoch[0]]
min_mae_val = test_metrics_hist[1][min_mae_epoch[0]]

ax.annotate('Min Test RMSE\n%.3f' % min_rmse_val, xy=(min_rmse_epoch[0], min_rmse_val),
                                                  xytext=(min_rmse_epoch[0] * 0.95, min_rmse_val * 1.05),
            arrowprops=dict(facecolor='g', shrink=0.05),
            )

ax.annotate('Min Test MAE\n%.3f' % min_mae_val, xy=(min_mae_epoch[0], min_mae_val),
                                                xytext=(min_mae_epoch[0] * 0.95, min_mae_val * 1.05),
            arrowprops=dict(facecolor='b', shrink=0.05),
            )

ax.set_xlabel("Epoch")
ax.set_ylabel("Score")
ax.set_title("Movielens 1M (90%/10%)")
plt.legend(loc=3)
plt.tight_layout()
plt.show()
