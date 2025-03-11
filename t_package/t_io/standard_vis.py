import matplotlib.pyplot as plt
import seaborn as sns

dpi=100

sns.set()
sns.set_style("whitegrid", {'grid.linestyle': '--'})
sns.set_context("paper", 1.5, {"lines.linewidth": 1.5})#font size, line width
sns.set_palette("deep")

def make_colors5():
    C = ['#528EFF','#FF5C7D','#FFC13B','#4CFF6A','#38FFF2']
    return C

def plot(file, x, ys, xlabel, ylabel, legend):
    #input:file=file name, x=x, ys=[y,,], xlabel=xlabel, ylabel=ylabel, legend=[legend,...]
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15, bottom=0.15)
    for i in range(len(ys)):
        ax.plot(x, ys[i], label=legend[i])
        #ax.scatter(x, ys[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.yaxis.offsetText.set_fontsize(16)
    plt.gca().ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    plt.savefig(file, format='pdf', dpi=dpi)
    plt.close()

def plot2(file, x, ys, xlabel, ylabel, legend):
    #input:file=file name, x=x, ys=[y,,], xlabel=xlabel, ylabel=ylabel, legend=[legend,...]
    colors = ['blue','orange','green']#make_colors5()
    fig = plt.figure(figsize=(8, 4), facecolor='white', dpi=dpi)
    for i in range(2):
        ax = fig.add_subplot(1, 2, i+1, frameon=True)
        #ax.set_aspect('equal', adjustable='box')
        ax.plot(x, ys[i], c=colors[i])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.title(legend[i])
        ax.yaxis.offsetText.set_fontsize(16)
        plt.gca().ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    plt.tight_layout()
    plt.savefig(file, format='pdf')
    plt.close()

def plot3(file, x, ys, xlabel, ylabel, legend):
    #input:file=file name, x=x, ys=[y,,], xlabel=xlabel, ylabel=ylabel, legend=[legend,...]
    colors = ['blue','orange','green']#make_colors5()
    fig = plt.figure(figsize=(12, 4), facecolor='white', dpi=dpi)
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1, frameon=True)
        #ax.set_aspect('equal', adjustable='box')
        ax.plot(x, ys[i], c=colors[i])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.title(legend[i])
        ax.yaxis.offsetText.set_fontsize(16)#e-2
        plt.gca().ticklabel_format(style="sci", scilimits=(0,0), axis="y")#e-2
    plt.tight_layout()
    plt.savefig(file, format='pdf')
    plt.close()

def plot_err_bar(file, x, ys, stds, xlabel, ylabel, legend, C):
    #input:file=file name, x=x, ys=[y,,], std=[std,..],xlabel=xlabel, ylabel=ylabel, legend=[legend,...]
    SIZE = 20
    #fig = plt.figure(figsize=(4, 3), facecolor='white', dpi=dpi)
    #ax = fig.add_subplot()
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15, bottom=0.15)
    for i in range(len(ys)):
        ax.errorbar(x, ys[i], yerr=stds[i], marker='o', label=legend[i], elinewidth=1.5, capthick=1, capsize=4, lw=2., color=C[i])
    ax.set_xlabel(xlabel, fontsize=SIZE)
    ax.set_ylabel(ylabel, fontsize=SIZE)
    ax.legend(fontsize=20, loc='upper right')
    ax.yaxis.offsetText.set_fontsize(SIZE)#e-2
    plt.gca().ticklabel_format(style="sci", scilimits=(0,0), axis="y")#e-2

    plt.tick_params(labelsize=SIZE)
    #plt.tight_layout()
    plt.savefig(file, format='pdf', dpi=dpi)
    plt.close()
