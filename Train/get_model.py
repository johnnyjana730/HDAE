
from parser import parse_args

hyper_params = {}
args = parse_args()
for arg in vars(args):
    hyper_params[arg] = getattr(args, arg)

# ********************* model select *****************************
from Train.train_rec_mt_neva_tsne_2 import Trainer as Trainer