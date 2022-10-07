import numpy as np
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


matplotlib.rcParams.update({'font.size': 18})


sns.reset_defaults()
sns.set(
    style="whitegrid", 
    palette="muted", 
    color_codes=True
)

def smooth(scalars, weight= 0.8):
    last = scalars[0]  
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point 
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


def read_file(f_names):
    
    
    to_smooth = ["discounted_ProduceBuildingRewardFunction","ProduceCombatUnitRewardFunction","AttackRewardFunction","discounted_WinLossRewardFunction","ProduceBuildingRewardFunction","discounted_ProduceWorkerRewardFunction","ProduceWorkerRewardFunction","ProduceWorkerRewardFunction","discounted_AttackRewardFunction","ResourceGatherRewardFunction","discounted_ResourceGatherRewardFunction", "discounted_ProduceCombatUnitRewardFunction"]
    
    fig, axs = plt.subplots(3,5, figsize=(24,12))
    
    dfs = []
    for f_name in f_names:
        dfs.append(pd.read_csv(f_name))
    
    
    for df,i in zip(dfs, range(len(dfs))):
        df["episodic_return_smoothed"] = smooth(df["episodic_return"], 0.8)
        sns.lineplot(data=df,x='steps', y='episodic_return_smoothed', ax=axs[0][0], label = f_names[i], legend=False)
        axs[0][0].set_title("Return vs timesteps")
        axs[0][0].set_xlabel("Number of timesteps")
        axs[0][0].set_ylabel("Average return")
    
    
    columns = list(df.columns)
    i = 0
    for c in columns:
        if c not in ["steps", "episodic_return", "Unnamed: 0", "episodic_return_smoothed"]:
            i +=1
            x,y = i // 5, i%5
            for df in dfs:
                if c in to_smooth:
                    df[c+"_smoothed"] = smooth(df[c], 0.8)
                    sns.lineplot(data=df,x='steps', y=c+"_smoothed", ax=axs[x][y])
                else:
                    sns.lineplot(data=df,x='steps', y=c, ax=axs[x][y])
                axs[x][y].set_title( c + " vs timesteps")
                axs[x][y].set_xlabel("Number of timesteps")
                axs[x][y].set_ylabel("Average return")
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5,0))
    fig.tight_layout()
    plt.savefig(f_names[0] + "_plots.png")
    plt.show()

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name',  default=[], nargs='+',
        help='the name of this experiment')
    args = parser.parse_args()
    read_file(args.exp_name)




    

if __name__ == "__main__":
    args = parse_args()
