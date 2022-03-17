
from parser import parse_args

hyper_params = {}
args = parse_args()
for arg in vars(args):
    hyper_params[arg] = getattr(args, arg)

from Models.Teacher import Teacher as Teacher

# ********************* model select *****************************
from Models.model_hyp_dis_rec_10_2_gbl_tsne import Student_Hyp_DIS_REC as Student
