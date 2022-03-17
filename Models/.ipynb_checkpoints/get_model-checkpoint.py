
from parser import parse_args

hyper_params = {}
args = parse_args()
for arg in vars(args):
    hyper_params[arg] = getattr(args, arg)

from Models.Teacher import Teacher as Teacher

# ********************* model select *****************************
if hyper_params['student_type'] in [ 'bert_fin' ]: from Models.model_bert import Student_Bert as Student
# elif hyper_params['student_type'] in [ 'bert_rec' ]: from Models.model_bert_rec import Student_Bert as Student
elif hyper_params['student_type'] in [ 'bert_rec' ]: from Models.model_bert_rec_6 import Student_Bert as Student
elif hyper_params['student_type'] in [ 'bert_rec_7' ]: from Models.model_bert_rec_7 import Student_Bert as Student

elif hyper_params['student_type'] in [ 'bert_rec_wemb' ]: from Models.model_bert_rec_emb import Student_BertWD as Student
elif hyper_params['student_type'] in [ 'bert_rec_wemb_hyp' ]: from Models.model_bert_rec_emb_hyp import Student_BertWD_HYP as Student

elif hyper_params['student_type'] in [ 'hyp_bert_rec' ]: from Models.model_bert_rec_hpy import Student_HPY_Bert as Student

elif hyper_params['student_type'] in [ 'w2v_fin' ]: from Models.model_w2v_fin import Student_W2V as Student
elif hyper_params['student_type'] in [ 'w2v_tsne' ]: from Models.model_w2v_tsne import Student_W2V as Student


elif hyper_params['student_type'] in [ 'w2v_rec_fin' ]: from Models.model_w2v_rec_fin import Student_W2V_REC as Student
elif hyper_params['student_type'] in [ 'w2v_rec_tsne' ]: from Models.model_w2v_rec_tsne import Student_W2V_REC as Student

elif hyper_params['student_type'] in [ 'w2v_rec_inner' ]: from Models.model_w2v_rec_inner import Student_W2V_REC as Student

elif hyper_params['student_type'] in [ 'hyper' ]: from Models.model_hyp import Student_Hyp as Student
elif hyper_params['student_type'] in [ 'hyper_rec' ]: from Models.model_hyp_rec import Student_Hyp_REC as Student
elif hyper_params['student_type'] in [ 'hyper_rec_dis' ]: from Models.model_hyp_dis_rec import Student_Hyp_DIS_REC as Student
elif hyper_params['student_type'] in [ 'hyper_rec_dis_9' ]: from Models.model_hyp_dis_rec_9 import Student_Hyp_DIS_REC as Student
elif hyper_params['student_type'] in [ 'hyper_rec_dis_10' ]: from Models.model_hyp_dis_rec_10_2 import Student_Hyp_DIS_REC as Student
elif hyper_params['student_type'] in [ 'hyper_rec_dis_10_1' ]: from Models.model_hyp_dis_rec_10 import Student_Hyp_DIS_REC as Student
elif hyper_params['student_type'] in [ 'hyper_rec_dis_10_3' ]: from Models.model_hyp_dis_rec_10_3 import Student_Hyp_DIS_REC as Student

elif hyper_params['student_type'] in [ 'hyper_rec_dis_10_gbl' ]: from Models.model_hyp_dis_rec_10_2_gbl import Student_Hyp_DIS_REC as Student
elif hyper_params['student_type'] in [ 'hyper_rec_dis_10_gbl_tsne' ]: from Models.model_hyp_dis_rec_10_2_gbl_tsne import Student_Hyp_DIS_REC as Student
elif hyper_params['student_type'] in [ 'hyper_rec_dis_10_gbl_tsne_d_ratio' ]: from Models.model_hyp_dis_rec_10_2_gbl_tsne_d_ratio import Student_Hyp_DIS_REC as Student
elif hyper_params['student_type'] in [ 'hyper_rec_dis_10_gbl_tsne_noseed' ]: from Models.model_hyp_dis_rec_10_2_gbl_noseed import Student_Hyp_DIS_REC as Student


elif hyper_params['student_type'] in [ 'hyper_rec_disa_fin' ]: from Models.model_hyp_dis_rec_10_2_gbl_disa_fin import Student_Hyp_DIS_REC as Student

elif hyper_params['student_type'] in [ 'hyper_rec_disa_fin_no_lab' ]: from Models.model_hyp_dis_rec_10_2_gbl_disa_fin_no_label import Student_Hyp_DIS_REC as Student
# model_hyp_dis_rec_gbl_fin_no_label

elif hyper_params['student_type'] in [ 'hyper_rec_gbl_fin' ]: from Models.model_hyp_dis_rec_gbl_fin import Student_Hyp_DIS_REC as Student

elif hyper_params['student_type'] in [ 'hyper_rec_dis_10_gbl_disa' ]: from Models.model_hyp_dis_rec_10_2_gbl_disa import Student_Hyp_DIS_REC as Student

elif hyper_params['student_type'] in [ 'hyper_rec_dis_10_gbl_as_eu' ]: from Models.model_hyp_dis_rec_10_2_gbl_as_eu import Student_Hyp_DIS_REC as Student
elif hyper_params['student_type'] in [ 'hyper_rec_dis_10_gbl_as_eu_lr' ]: from Models.model_hyp_dis_rec_10_2_gbl_as_eu_lr import Student_Hyp_DIS_REC as Student
elif hyper_params['student_type'] in [ 'hyper_rec_dis_10_gbl_ca_eu' ]: from Models.model_hyp_dis_rec_10_2_gbl_ca_eu import Student_Hyp_DIS_REC as Student

elif hyper_params['student_type'] in [ 'hyper_rec_dis_10_gbl_lr' ]: from Models.model_hyp_dis_rec_10_2_gbl_lr import Student_Hyp_DIS_REC as Student
elif hyper_params['student_type'] in [ 'eu_rec_dis_10' ]: from Models.model_mt_dis_rec_eu import Student_Hyp_DIS_REC as Student
