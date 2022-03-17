import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from parser import parse_args
args = parse_args()

# darkgrid, whitegrid, dark, white and ticks
sns.set_style('ticks')

plt.rc('axes', titlesize=18)    # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)   # fontsize of the tick labels
plt.rc('ytick', labelsize=13)   # fontsize of the tick labels
plt.rc('legend', fontsize=13)   # legend fontsize
plt.rc('font', size=13)         # controls default text sizes

if args.dataset == "BOOTS":
    aspect_list = "Color|Comfort|Durable|Look|Materials|None|Price|Size|WT_resist"

elif args.dataset == "BAGS_AND_CASES":
    aspect_list = "Compartments|Service|Handles|Looks|None|Price|Protection|Quality|Size_Fit"

elif args.dataset == "TV":
    aspect_list = "Apps|Connectivity|Service|Ease_of_use|Image|None|Price|Size_Look|Sound"

elif args.dataset == "KEYBOARDS":
    aspect_list = "BD_Quality|Connectivity|Extra_function|Feel_Comfort|Layout|Looks|Noise|None|Price"

elif args.dataset == "VACUUMS":
    aspect_list = "Accessories|BD_Quality|Service|Ease_of_use|Noise|None|Price|SC_Power|Weight"

elif args.dataset == "BLUETOOTH":
    aspect_list = "Battery|Comfort|Connectivity|Durable|Ease_of_use|Look|None|Price|Sound"

elif args.dataset[:4] == "REST":
    aspect_list = "RESTAURANT#GENERAL|FOOD#QUALITY|SERVICE#GENERAL|AMBIENCE#GENERAL|FOOD#STYLE_OPTIONS|FOOD#PRICES|RESTAURANT#MISCELLANEOUS|RESTAURANT#PRICES|DRINKS#QUALITY|DRINKS#STYLE_OPTIONS|LOCATION#GENERAL|DRINKS#PRICES"

aspect_list = aspect_list.split('|')

def group_data(*data):
    df = pd.concat(data)
    df = df.groupby([df.index, "Aspect", "Name"], as_index=False).sum()
    return df

def plot_figure(predict_true, predict_total, ground_true, epoch):
    df_1 = pd.DataFrame(data={
            "Index": list(range(len(predict_true.keys()))),
            "Aspect": aspect_list,
            "Number": list(predict_true.values()),
            "Name": ["pred truth"] * args.st_num_aspect})

    df_2 = pd.DataFrame(data={
            "Index": list(range(len(predict_total.keys()))),
            "Aspect": aspect_list,
            "Number": predict_total.values(),
            "Name": "pred false"})

    df_3 = pd.DataFrame(data={
            "Index": list(range(len(ground_true.keys()))),
            "Aspect": aspect_list,
            "Number": ground_true.values(),
            "Name": "ground truth"})

    df_4 = pd.DataFrame(data={
            "Index": list(range(len(ground_true.keys()))),
            "Aspect": aspect_list,
            "Number": [0] * len(ground_true.keys()),
            "Name": "ground truth"})

    # #42bd79 #cfcf7c #9bdb95  #e8a2ac #a2a4e8
    barplot = group_data(df_2, df_3)
    plt.figure(figsize=(12, 9), tight_layout=True)
    bar1 = sns.barplot(x=barplot["Aspect"], y=barplot["Number"], hue=barplot["Name"], palette=["#c7f2da", "#ff9eab"])

    # #59ff72
    barplot = group_data(df_1, df_3)
    bar2 = sns.barplot(x=barplot["Index"], y=barplot["Number"], hue=barplot["Name"], estimator=sum, ci=None, palette=["#55e66b", "#91e4ff"])
    
    bar2.set(title="", xlabel="Aspect", ylabel="Number")
    bar2.get_legend().remove()

    plt.savefig(args.pic_file + '/plot_fig{:d}.png'.format(epoch))
    plt.close()