def set_my_style():
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("paper", font_scale=1.5)
    plt.rcParams["figure.figsize"] = (8, 6)  # figure size in inch, 横×縦
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams["font.family"] = "Times New Roman"  # 全体のフォントを設定
    plt.rcParams["xtick.direction"] = "in"  # x axis in
    plt.rcParams["ytick.direction"] = "in"  # y axis in
    plt.rcParams["axes.linewidth"] = 1.0  # axis line width
    plt.rcParams["axes.grid"] = True  # make grid

    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.default"] = "it"
    plt.rcParams["mathtext.it"] = "cmmi10"
    plt.rcParams["mathtext.bf"] = "CMU serif:italic:bold"
    plt.rcParams["mathtext.rm"] = "cmb10"
    plt.rcParams["mathtext.fallback"] = "cm"
