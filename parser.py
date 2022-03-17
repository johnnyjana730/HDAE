import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument("--dataset", type=str, default='./data/bags_train.json', help="dataset")
    parser.add_argument("--expname", type=str, default='./ckpt/bags', help="expname")
    parser.add_argument("--save_dir", type=str, default='./ckpt/bags', help="save_dir")
    parser.add_argument("--test_file", type=str, default='./data/bags_test.json', help="test_file")
    parser.add_argument("--train_file", type=str, default='./data/bags_train.json', help="train_file")
    parser.add_argument("--aspect_seeds", type=str, default='./data/bags_train.json', help="aspect_seeds")
    parser.add_argument("--aspect_init_file", type=str, default='./data/seedwords/bags_and_cases.30.txt', help="aspect_init_file")

    parser.add_argument("--pic_file", type=str, default='', help="pic_file")
    parser.add_argument("--sumout_grum_probs", type=str, default='', help="sumout_grum_probs")
    parser.add_argument("--sumout", type=str, default='', help="sumout")
    parser.add_argument("--sumout_2", type=str, default='', help="sumout_2")

    parser.add_argument("--exp_info", type=str, default='', help="exp_info")

    # General
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--epoch", type=int, default=100, help="epoch")
    parser.add_argument("--maxlen", type=int, default=40, help="maxlen")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--description", type=str, default='bag baseline', help="description")

    # Hyperbolic
    parser.add_argument("--hyper_beta", type=float, default=0.02, help="hyper_beta")
    parser.add_argument("--training_amont", type=float, default=1.0, help="hyper_beta")

    # Disentangle
    parser.add_argument("--dis_1", type=float, default=25.0, help="dis_1")
    parser.add_argument("--dis_2", type=float, default=1.0, help="dis_2")
    parser.add_argument("--dis_3", type=float, default=20.0, help="dis_3")

    parser.add_argument("--tr_lab_ratio", type=float, default=0.5, help="lab_ratio")
    parser.add_argument("--tr_fin_lab_ratio", type=float, default=0.5, help="lab_ratio")

    # tr_fin_lab_ratio
    parser.add_argument("--dis_mun", type=int, default=2, help="dis_mun")
    parser.add_argument("--gb_temp", type=float, default=0.00001, help="grumbel_temperature")
    parser.add_argument("--mt_ratio", type=float, default=0.5, help="mt_ratio")
    parser.add_argument("--d1_ratio", type=float, default=1, help="d1_ratio")
    parser.add_argument("--d2_ratio", type=float, default=1, help="d2_ratio")
    parser.add_argument("--d3_ratio", type=float, default=1, help="d3_ratio")
    parser.add_argument("--w2v_ratio", type=float, default=0.1, help="w2v_ratio")

    # Teacher Student
    parser.add_argument("--train_type", type=str, default='rec_mt', help="train_type")
    parser.add_argument("--general_asp", type=int, default=4, help="general_asp")
    parser.add_argument("--student_type", type=str, default='hyper', help="student_type")

    # Other
    parser.add_argument("--st_pre", type=str, default='bert-base-uncased', help="st_pre")
    parser.add_argument("--st_pre_dim", type=int, default=768, help="st_pre_dim")
    parser.add_argument("--st_num_aspect", type=int, default=9, help="st_num_aspect")
    parser.add_argument("--aspect_tsne_bt", type=int, default=100, help="aspect_tsne_bt")

    args = parser.parse_args()

    try:
        if not os.path.isdir(args.sumout): 
            os.makedirs(args.sumout)
    except: pass

    args.pic_file = args.sumout 
    args.sumout_grum_probs = args.sumout 
    args.sumout = args.sumout + 'eval_score.txt'

    return args