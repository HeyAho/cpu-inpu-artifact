import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns  # 新增：用于绘制美观的热力图

def analyze_hardware_profile(csv_file):
    # 1. 加载数据
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"错误：未找到文件 {csv_file}")
        return

    # 2. 数据清洗与分组
    # 确保关键列为数值型
    cols_to_convert = ['ANE', 'ECPU', 'PCPU', 'Timestamp']
    for col in [c for c in cols_to_convert if c != 'Timestamp' and c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 我们定义堆叠的层（从底向上）：
    layer1_ane = df['ANE'] if 'ANE' in df.columns else pd.Series([0]*len(df))
    layer3_ecpu = df['ECPU'] if 'ECPU' in df.columns else pd.Series([0]*len(df))
    layer4_pcpu = df['PCPU'] if 'PCPU' in df.columns else pd.Series([0]*len(df))

    # 组合成堆叠数据列表
    stacked_data = [layer1_ane, layer3_ecpu, layer4_pcpu]

    # ==========================================
    # 第一部分：绘制堆叠面积图 (保留原有逻辑)
    # ==========================================
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor('#ffffff')

    colors = ['#87CEFA', '#FF6F61', '#FFDAB9']

    x_axis = range(len(df))

    ax.stackplot(x_axis, stacked_data, 
                 labels=['ANE Unit', 'ECPU Cluster', 'PCPU Cluster'],
                 colors=colors, 
                 alpha=1.0)
    
    ax.grid(False)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.set_ylabel('Power / Usage (Stacked)', fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # 设置 X 轴边界，消除两端空白
    ax.set_xlim(0, len(df) - 1)
    
    plt.title('Stacked Hardware Security Hardware Feature Profile (ANE/ECPU/PCPU)', fontsize=16, pad=20)
    plt.tight_layout()
    
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=10)

    output_stacked = '02stacked_hardware_profile.png'
    plt.savefig(output_stacked, dpi=300, bbox_inches='tight')
    print(f"堆叠图生成成功！已保存为: {output_stacked}")
    plt.close() # 关闭当前画布，准备绘制下一张

    # ==========================================
    # 第二部分：绘制相关性热力图 (新增逻辑)
    # ==========================================
    # 提取需要分析的列，并重命名以便图表显示更直观
    corr_data = pd.DataFrame({
        'ANE Unit': layer1_ane,
        'ECPU Cluster': layer3_ecpu,
        'PCPU Cluster': layer4_pcpu
    })

    # 计算皮尔逊相关系数矩阵
    corr_matrix = corr_data.corr()

    # 创建新的画布
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    fig2.patch.set_facecolor('#ffffff')

    # 绘制热力图
    # annot=True 显示数值, cmap='coolwarm' 提供冷暖色调区分正负相关
    sns.heatmap(corr_matrix, 
                annot=True,          # 在色块中显示相关系数数值
                fmt=".2f",           # 数值保留两位小数
                cmap="coolwarm",     # 配色方案：蓝-白-红
                vmin=-1, vmax=1,     # 相关系数范围必定是 -1 到 1
                center=0,            # 0 对应中心颜色 (白色)
                square=True,         # 保证每个格子是正方形
                linewidths=1,        # 格子之间的间距
                cbar_kws={"shrink": .8}) # 稍微缩小右侧颜色条

    plt.title('Hardware Correlation Heatmap (ANE, ECPU, PCPU)', fontsize=14, pad=15)
    plt.tight_layout()

    output_heatmap = '03correlation_heatmap.png'
    plt.savefig(output_heatmap, dpi=300, bbox_inches='tight')
    print(f"热力图生成成功！已保存为: {output_heatmap}")
    plt.close()

if __name__ == "__main__":
    analyze_hardware_profile('hardware_features.csv')


# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# def plot_stacked_hardware_profile(csv_file):
#     # 1. 加载数据
#     try:
#         df = pd.read_csv(csv_file)
#     except FileNotFoundError:
#         print(f"错误：未找到文件 {csv_file}")
#         return

#     # 2. 数据清洗与分组 (这是复刻风格的关键步骤)
#     # 我们将数百个列组合成可以堆叠的逻辑组。
#     # 注意：如果某一组全部为0，需要确保它们的Shape一致，可以使用 .clip(lower=0)
    
#     # 确保关键列为数值型
#     cols_to_convert = ['ANE', 'ECPU', 'PCPU', 'Timestamp']
#     # 临时移除非数值列进行转换，或者逐个转换
#     for col in [c for c in cols_to_convert if c != 'Timestamp' and c in df.columns]:
#         df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

#     # 我们定义堆叠的层（从底向上）：
#     # 第一层：ANE (神经网络单元)
#     layer1_ane = df['ANE']
    
#     # 第二层：GPU (图形单元)
#     # layer2_gpu = df['GPU']
    
#     # 第三层：ECPU (效率核心)
#     # 注意：CSV 表头有 'ECPU', 'ECPU0'...'ECPU3'。为了演示，我们用总 ECPU，
#     # 如果总 ECPU 不准，需要手动加和：df[['ECPU0','ECPU1','ECPU2','ECPU3']].sum(axis=1)
#     layer3_ecpu = df['ECPU']
    
#     # 第四层：PCPU (性能核心)
#     layer4_pcpu = df['PCPU']

#     # 组合成堆叠数据列表
#     # (重要：顺序决定了从底向上的显示位置)
#     stacked_data = [layer1_ane,layer3_ecpu, layer4_pcpu]

#     # 3. 创建画布并定义核心配色 (复刻 image_0.png 的扁平化风格)
#     fig, ax = plt.subplots(figsize=(16, 6)) # 稍微放宽，更清晰
#     fig.patch.set_facecolor('#ffffff') # 整个画布背景为白色

#     # 复刻配色方案（十六进制）
#     # 参考图中主要有三种颜色（蓝、绿、红），为了堆叠四层，我们定义一组。
#     colors = ['#87CEFA', # 天蓝色 (对应 ANE)
#             #   '#A3E4D7', # 淡绿色 (对应 GPU)
#               '#FF6F61', # 珊瑚红 (对应 ECPU)
#               '#FFDAB9'] # 桃色 (对应 PCPU, 填充顶部空白)

#     # 4. 绘制面积堆叠图
#     # x 轴需要转换成序列或真实时间戳进行平滑
#     x_axis = range(len(df)) # 使用数据点的索引作为 X 轴

#     ax.stackplot(x_axis, stacked_data, 
#                  labels=['ANE Unit', 'ECPU Cluster', 'PCPU Cluster'],
#                  colors=colors, 
#                  alpha=1.0) # 保持扁平化，不使用透明度
    
#     # 5. 复刻极简主义美化 (Hide grids, axes ticks)
#     # 隐藏网格线
#     ax.grid(False)

#     # 只保留 Y 轴（显示功耗），隐藏 X 轴刻度标签（防止 Timestamp 挤在一起影响美观）
#     # 同时模仿参考图，只留一些主要的刻度。
#     ax.yaxis.set_major_locator(ticker.MaxNLocator(4)) # 只留 4 个主要 Y 轴刻度
#     ax.set_ylabel('Power / Usage (Stacked)', fontsize=12)

#     # 模仿参考图，完全隐藏 X 轴的刻度标签
#     # ax.set_xticklabels([]) # 如果您仍然想在图下方显示，可以注释这行
    
#     # 也可以只保留每隔10个点的 Timestamp（可选）
#     # n = len(df) // 10
#     # plt.xticks(x_axis[::n], df['Timestamp'][::n], rotation=0)

#     # 隐藏上边框和右边框，减少干扰
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_linewidth(0.5) # 极细的下边框
    
#     # 6. 图表标题与布局
#     plt.title('Stacked Hardware Security Hardware Feature Profile (ANE/ECPU/PCPU)', fontsize=16, pad=20)
#     plt.tight_layout()
    
#     # 添加一个扁平化的图例 (放在底部)
#     ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False, fontsize=10)

#     # 7. 保存并提示
#     output_name = '02stacked_hardware_profile.png'
#     plt.savefig(output_name, dpi=300)
#     print(f"风格复刻成功！图表已保存为: {output_name}")

# if __name__ == "__main__":
#     plot_stacked_hardware_profile('hardware_features.csv')



