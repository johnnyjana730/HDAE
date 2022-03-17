
from parser import parse_args

hyper_params = {}
args = parse_args()
for arg in vars(args):
    hyper_params[arg] = getattr(args, arg)

# ********************* model select *****************************
if hyper_params['train_type'] == 'tr_only': from Train.train_tc import Trainer as Trainer
elif hyper_params['train_type'] == 'ts_only': from Train.train import Trainer as Trainer
elif hyper_params['train_type'] == 'rec_only': from Train.train_rec import Trainer as Trainer
elif hyper_params['train_type'] == 'rec_mt': from Train.train_rec_mt import Trainer as Trainer

elif hyper_params['train_type'] == 'ts_only_neva': from Train.train_neva import Trainer as Trainer
elif hyper_params['train_type'] == 'ts_only_neva_plt': from Train.train_tc_neva_plot import Trainer as Trainer
elif hyper_params['train_type'] == 'ts_only_neva_tsne': from Train.train_tc_neva_tsne import Trainer as Trainer

elif hyper_params['train_type'] == 'tr_only_neva': from Train.train_tc_neva import Trainer as Trainer

elif hyper_params['train_type'] == 'rec_mt_neva_tratio': from Train.train_rec_mt_neva_tratio import Trainer as Trainer
elif hyper_params['train_type'] == 'rec_mt_tra_fin': from Train.train_rec_mt_neva_tratio_fin import Trainer as Trainer
elif hyper_params['train_type'] == 'rec_mt_tra_fin_2': from Train.train_rec_mt_neva_tratio_fin_2 import Trainer as Trainer

elif hyper_params['train_type'] == 'case_disen_fin_notea': from Train.train_rec_mt_neva_tf_no_tea_disen import Trainer as Trainer

elif hyper_params['train_type'] == 'case_disen_fin_notea_no_label': from Train.train_rec_mt_neva_tf_no_tea_disen_no_label import Trainer as Trainer

elif hyper_params['train_type'] == 'case_disen_fin_tea_no_label': from Train.train_rec_mt_neva_tf_tea_disen_no_label import Trainer as Trainer

elif hyper_params['train_type'] == 'rec_mt_tra_fin_notea': from Train.train_rec_mt_neva_tratio_fin_2_no_tea import Trainer as Trainer

elif hyper_params['train_type'] == 'rec_only_neva': from Train.train_rec_neva import Trainer as Trainer
elif hyper_params['train_type'] == 'rec_only_neva_plt': from Train.train_rec_neva_plot import Trainer as Trainer
elif hyper_params['train_type'] == 'rec_only_neva_tsne': from Train.train_rec_neva_tsne import Trainer as Trainer

elif hyper_params['train_type'] == 'rec_mt_neva': from Train.train_rec_mt_neva import Trainer as Trainer
elif hyper_params['train_type'] == 'rec_mt_neva_plt': from Train.train_rec_mt_neva_plot import Trainer as Trainer
elif hyper_params['train_type'] == 'rec_mt_neva_tsne': from Train.train_rec_mt_neva_tsne_2 import Trainer as Trainer
elif hyper_params['train_type'] == 'rec_mt_neva_tsne_2': from Train.train_rec_mt_neva_tsne_3 import Trainer as Trainer
elif hyper_params['train_type'] == 'rec_mt_neva_disen': from Train.train_rec_mt_neva_tsne_disen import Trainer as Trainer

elif hyper_params['train_type'] == 'rec_mt_neva_sr': from Train.train_rec_mt_neva_sr import Trainer as Trainer