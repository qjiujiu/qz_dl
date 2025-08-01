import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import yaml

def load_chart_config(yaml_path, chart_key):
    with open(yaml_path, 'r') as file:
        charts = yaml.safe_load(file)
    return charts[chart_key]

FMTS = (
        'r-s',   # 红色实线方形标记
        'g--o',  # 绿色虚线实线圆形标记
        'b-.^',  # 蓝色点划线三角形向上标记
        'c:v',   # 青色点线三角形向下标记
        'y-s',   # 黄色实线方形标记
        'm-o',   # 洋红色实线圆形标记
        'k-^',   # 黑色实线三角形向上标记
        'b-v',   # 蓝色实线三角形向下标记
        'g-*',   # 绿色实线星形标记
        'r-p',   # 红色实线五角形标记
        'c-h',   # 青色实线六边形标记
        'm-D',   # 洋红色实线菱形标记
        'y-x',   # 黄色实线叉形标记
        'k-|',   # 黑色实线竖直线形标记
        'b-_',   # 蓝色实线水平线形标记
    )

# 辅助函数
def plot_figure(X, Y = None, xlabel = None, ylabel = None, legend = None, 
                xlim = None, ylim = None, xscale = 'linear', yscale = 'linear',
                fmts = FMTS, figsize = (3.5, 2.5), axes = None, title = None, save_name:str = 'tmp.png'):
    
    def set_figsize(figsize=(3.5, 2.5)):
        plt.rcParams['figure.figsize'] = figsize

    def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()
    
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    def has_one_axis(X):
        # 检查传入的数据是不是一个 numpy 数组且只有一个维度，或者X是不是一个列表，且列表元素不存在序列长度 len
        return (X is not None) and ( ((hasattr(X, "ndim") and X.ndim == 1) or (isinstance(X, list))) 
                and not hasattr(X[0], "__len__") ) or (isinstance(X, range))


    X = [X] if has_one_axis(X) else X
    Y = [Y] if has_one_axis(Y) else Y
    if Y is None:       
        X, Y = [[]] * len(X), X

    # DEBUG
    # print(f"X = {X}, X shape = {np.array(X).shape}")
    # print(f"Y = {Y}, Y shape = {np.array(Y).shape}")
    
    # 我们允许X长度小于Y, 我们通过重复X来实现多个曲线共享同一个横轴，但是仅当 len(X) = 1, len(Y) > 1 这种情况是有意义的
    if len(X) != len(Y):
        X = X * len(Y)
        
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    plt.tight_layout()
    if title is not None:
        plt.title(title)
    plt.savefig(save_name)



'''
param {*} matrix        热力图矩阵
param {*} xlabel        横轴坐标
param {*} ylabel        纵轴坐标
param {*} title         热力图矩阵标题
param {*} cmap          颜色主题，通常我们使用 magma 主题:
                            
    1. 'coolwarm': 冷暖色调交替的主题，适合表示正负值的差异。
    2. 'magma': 黑紫红色调的主题，具有较高的对比度和可读性。
    3. 'inferno': 黑橙红色调的主题，也具有较高的对比度。
    4. 'plasma': 蓝紫橙色调的主题，颜色变化较为均匀。
                            
param {*} save_path     图像存储路径
'''
def plot_heatmap(matrix, xlabel = '', ylabel = '', title = 'Heatmap', cmap = 'viridis', save_path = 'heatmap.png'):    
    plt.figure(figsize=(10, 8))

    # 使用seaborn绘制热力图
    sns.set_theme(font_scale = 1.8)
    ax = sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap, linewidths=.5,
                     cbar_kws={'label': 'Similarity'}, square=True)
    # 设置坐标轴标签
    ax.set_xlabel(xlabel, fontsize=16, fontname="sans-serif")
    ax.set_ylabel(ylabel, fontsize=16, fontname="sans-serif")
    
    # 设置颜色条标签字体大小
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    
    # 设置坐标轴刻度字体
    plt.xticks(fontsize=14, fontname="sans-serif")
    plt.yticks(fontsize=14, fontname="sans-serif")
    
    # 显示图形
    plt.title(title, fontsize=18, fontname="sans-serif")
    plt.tight_layout()
    plt.savefig(save_path)


def parse_mean_std(data):
    """
    将形如 '56.28 ± 0.43' 的字符串解析为 (mean, std)
    """
    means = []
    stds = []
    for row in data:
        row_means = []
        row_stds = []
        for item in row:
            match = re.match(r"([\d.]+)\s*±\s*([\d.]+)", item)
            if match:
                mean, std = float(match[1]), float(match[2])
            else:
                mean, std = float(item), 0.0
            row_means.append(mean)
            row_stds.append(std)
        means.append(row_means)
        stds.append(row_stds)
    return means, stds

def plot_bar_with_error(x_labels,            # 横坐标类标签（如方法名）
                        raw_data,            # shape = (num_class, num_group)，字符串形式 "均值 ± 方差"
                        legend=None,         # 每个 group 的名称，如 ["Acc", "Prec", "Recall", "F1"]
                        xlabel=None, ylabel=None,
                        title=None, 
                        figsize=(10, 5), 
                        save_name='bar_plot.png',
                        colors=None):

    means, stds = parse_mean_std(raw_data)

    num_classes = len(x_labels)             # 横轴类别数量（如 5 个方法）
    num_groups = len(means[0])              # 每个类别下的指标数（如 4 个指标）

    x = np.arange(num_classes)              # 类别位置
    width = 0.15                            # 每个柱子的宽度
    total_width = width * num_groups        # 所有柱子的总宽度
    offset_starts = - (total_width - width) / 2   # 居中起始偏移

    if colors is None:
        # 使用你指定的4种颜色（RGB转换为matplotlib格式）
        colors = [
            (57 / 255, 81 / 255, 162 / 255),  # R:057 G:081 B:162
            (114 / 255, 170 / 255, 207 / 255),  # R:114 G:170 B:207
            (253 / 255, 185 / 255, 107 / 255),  # R:253 G:185 B:107
            (236 / 255, 93 / 255, 59 / 255),  # R:236 G:093 B:059
        ]


    plt.figure(figsize=figsize)

    for i in range(num_groups):
        pos = x + offset_starts + i * width
        mean_vals = [m[i] for m in means]
        std_vals = [s[i] for s in stds]
        plt.bar(pos, mean_vals, yerr=std_vals, width=width, label=legend[i], color=colors[i], capsize=3, ecolor='#666666')

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(ticks=x, labels=x_labels, fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylim(bottom=30)
    if legend:
        plt.legend(fontsize=11)
    if title:
        plt.title(title, fontsize=14)

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_name)
    # plt.savefig(save_name, format='pdf', bbox_inches='tight')
    plt.show()




if __name__ == '__main__':
    # config = load_chart_config('config/charts.yaml', 'BiLSTM_Perturbations')
    # config = load_chart_config('config/charts.yaml', 'ConvNet1D_Perturbations')
    config = load_chart_config('config/charts.yaml', 'BiLSTM_Attention')
    # config = load_chart_config('config/charts.yaml', 'ConvNet1D_Attention')
    plot_bar_with_error(
        x_labels=config['x_labels'],
        raw_data=config['raw_data'],
        legend=config['legend'],
        xlabel=config['xlabel'],
        ylabel=config['ylabel'],
        title=config['title'],
        save_name=config['save_name']
    )












    # # X 轴：PGD迭代次数
    # x = [1, 2, 3, 4, 5]

    # # BiLSTM 模型的 Acc(%)
    # bilstm_acc = [66.91, 82.68, 82.37, 81.17, 80.64]

    # # ConvNet1D 模型的 Acc(%)
    # convnet_acc = [71.13, 80.28, 81.41, 82.00, 82.32]

    # # 构造数据结构
    # X = [x, x]  # 两条线共享相同的横坐标
    # Y = [bilstm_acc, convnet_acc]
    # legend = ['BiLSTM', 'ConvNet1D']

    # # 调用绘图函数
    # plot_figure(
    #     X=X,
    #     Y=Y,
    #     xlabel='PGD Iterations',
    #     ylabel='Accuracy (%)',
    #     legend=legend,
    #     xlim=(0.8, 5.2),
    #     ylim=(65, 85),
    #     title='Accuracy vs PGD Iterations',
    #     save_name='pgd_acc_curve.png'
    # )

