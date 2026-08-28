import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_stacked_hardware_profile(csv_file):
    # 1. 加载数据
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"错误：未找到文件 {csv_file}")
        return

    # 2. 数据清洗与分组 (这是复刻风格的关键步骤)
    # 我们将数百个列组合成可以堆叠的逻辑组。
    # 注意：如果某一组全部为0，需要确保它们的Shape一致，可以使用 .clip(lower=0)
    
    # 确保关键列为数值型
    cols_to_convert = ['ANE', 'Timestamp']
    # 临时移除非数值列进行转换，或者逐个转换
    for col in [c for c in cols_to_convert if c != 'Timestamp' and c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 我们定义堆叠的层（从底向上）：
    # 第一层：ANE (神经网络单元)
    layer1_ane = df['ANE']
    

    # 组合成堆叠数据列表
    # (重要：顺序决定了从底向上的显示位置)
    stacked_data = [layer1_ane]

    # 3. 创建画布并定义核心配色 (复刻 image_0.png 的扁平化风格)
    fig, ax = plt.subplots(figsize=(16, 6)) # 稍微放宽，更清晰
    fig.patch.set_facecolor('#ffffff') # 整个画布背景为白色

    # 复刻配色方案（十六进制）
    # 参考图中主要有三种颜色（蓝、绿、红），为了堆叠四层，我们定义一组。
    colors = ['#87CEFA', # 天蓝色 (对应 ANE)
] # 桃色 (对应 PCPU, 填充顶部空白)

    # 4. 绘制面积堆叠图
    # x 轴需要转换成序列或真实时间戳进行平滑
    x_axis = range(len(df)) # 使用数据点的索引作为 X 轴

    ax.stackplot(x_axis, stacked_data, 
                 labels=['ANE Unit'],
                 colors=colors, 
                 alpha=1.0) # 保持扁平化，不使用透明度
    
    # 5. 复刻极简主义美化 (Hide grids, axes ticks)
    # 隐藏网格线
    ax.grid(False)

    # 只保留 Y 轴（显示功耗），隐藏 X 轴刻度标签（防止 Timestamp 挤在一起影响美观）
    # 同时模仿参考图，只留一些主要的刻度。
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4)) # 只留 4 个主要 Y 轴刻度
    ax.set_ylabel('Power / Usage (Stacked)', fontsize=12)

    # 模仿参考图，完全隐藏 X 轴的刻度标签
    # ax.set_xticklabels([]) # 如果您仍然想在图下方显示，可以注释这行
    
    # 也可以只保留每隔10个点的 Timestamp（可选）
    # n = len(df) // 10
    # plt.xticks(x_axis[::n], df['Timestamp'][::n], rotation=0)

    # 隐藏上边框和右边框，减少干扰
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5) # 极细的下边框
    
    # 6. 图表标题与布局
    plt.title('Stacked Hardware Security Hardware Feature Profile (ANE)', fontsize=16, pad=20)
    plt.tight_layout()
    
    # 添加一个扁平化的图例 (放在底部)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False, fontsize=10)

    # 7. 保存并提示
    output_name = '01stacked_hardware_profile.png'
    plt.savefig(output_name, dpi=300)
    print(f"风格复刻成功！图表已保存为: {output_name}")

if __name__ == "__main__":
    plot_stacked_hardware_profile('hardware_features.csv')