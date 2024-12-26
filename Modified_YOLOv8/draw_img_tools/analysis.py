import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 在文件开头添加配色方案
COLORS = {
    'primary': 'rgba(0, 63, 92, 0.8)',     # 深蓝色
    'secondary': 'rgba(188, 80, 144, 0.7)', # 紫色
    'accent': 'rgba(255, 166, 0, 0.8)',     # 橙色
    'complementary': 'rgba(44, 162, 95, 0.7)', # 绿色
    'neutral': 'rgba(128, 128, 128, 0.6)'    # 灰色
}

# 科学配色方案
SCIENTIFIC_COLORSCALE = [
    [0, 'rgba(0, 63, 92, 0.2)'],
    [0.5, 'rgba(188, 80, 144, 0.5)'],
    [1, 'rgba(255, 166, 0, 0.8)']
]

# 添加类别名称映射
CLASS_NAMES = ['pl80', 'p6', 'ph', 'w', 'pa', 'p27', 'i5', 'p1', 'il70', 'p5', 'pm', 'p19', 'ip', 
               'p11', 'p13', 'p26', 'i2', 'pn', 'p10', 'p23', 'pbp', 'p3', 'p12', 'pne', 'i4', 'pb', 
               'pg', 'pr', 'pl5', 'pl10', 'pl15', 'pl20', 'pl25', 'pl30', 'pl35', 'pl40', 'pl50', 
               'pl60', 'pl65', 'pl70', 'pl90', 'pl100', 'pl110', 'pl120', 'il50', 'il60', 'il80', 
               'il90', 'il100', 'il110']

def load_tt100k_from_txt(train_dir, val_dir):
    """
    读取训练集和验证集的标签文件
    train_dir: 训练集标签目录
    val_dir: 验证集标签目录
    """
    def load_from_dir(dir_path, split):
        objects = []
        # 展开波浪号路径
        dir_path = os.path.expanduser(dir_path)
        if not os.path.exists(dir_path):
            print(f"警告: 目录不存在 {dir_path}")
            return objects
            
        txt_files = list(Path(dir_path).glob('*.txt'))
        if not txt_files:
            print(f"警告: 在 {dir_path} 中没有找到txt文件")
            return objects
            
        print(f"正在处理 {split} 集，共找到 {len(txt_files)} 个标签文件")
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r') as f:
                    lines = f.readlines()
                    
                    for line in lines:
                        values = line.strip().split()
                        if len(values) == 5:
                            class_id = int(values[0])
                            x_center = float(values[1])
                            y_center = float(values[2])
                            width = float(values[3])
                            height = float(values[4])
                            
                            x1 = x_center - width/2
                            y1 = y_center - height/2
                            x2 = x_center + width/2
                            y2 = y_center + height/2
                            
                            objects.append({
                                'category': class_id,
                                'bbox': [x1, y1, x2, y2],
                                'center': [x_center, y_center],
                                'split': split,  # 添加数据集标记
                                'image_path': str(txt_file).replace('.txt', '.jpg')  # 添加图像路径
                            })
            except Exception as e:
                print(f"处理文件 {txt_file} 时出错: {str(e)}")
                continue
        
        print(f"{split} 集中找到 {len(objects)} 个目标")
        return objects
    
    train_objects = load_from_dir(train_dir, 'train')
    val_objects = load_from_dir(val_dir, 'val')
    
    total_objects = train_objects + val_objects
    if not total_objects:
        raise ValueError("没有找到任何有效的标签数据！请检查输入路径是否正确。")
        
    return total_objects

def save_figure(fig, output_dir, filename):
    """保存图表为高质量PNG文件"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存之前确保布局设置被应用
    fig.write_image(
        f"{output_dir}/{filename}.png",
        scale=2,
        width=800,   # 确保与布局中设置的宽度一致
        height=800,  # 确保与布局中设置的高度一致
        engine="kaleido"
    )
    fig.show()

def analyze_categories(objects, output_dir):
    """分析训练集和验证集的类别分布"""
    train_categories = {}
    val_categories = {}
    
    for obj in objects:
        cat = CLASS_NAMES[obj['category']]
        if obj['split'] == 'train':
            train_categories[cat] = train_categories.get(cat, 0) + 1
        else:
            val_categories[cat] = val_categories.get(cat, 0) + 1
    
    # 创建数据框并排序
    categories = sorted(set(train_categories.keys()) | set(val_categories.keys()))
    df = pd.DataFrame({
        'Category': categories,
        'Train': [train_categories.get(cat, 0) for cat in categories],
        'Val': [val_categories.get(cat, 0) for cat in categories],
        'Total': [train_categories.get(cat, 0) + val_categories.get(cat, 0) for cat in categories]
    })
    
    # 按总数排序并只选择top 15的类别
    df = df.sort_values('Total', ascending=False)
    
    # 创建横向条形图
    fig = go.Figure()
    
    # 添加训练集条形
    fig.add_trace(go.Bar(
        x=df['Category'],
        y=df['Train'],
        name='Train',
        marker_color=COLORS['primary'],
    ))
    
    # 添加验证集条形
    fig.add_trace(go.Bar(
        x=df['Category'],
        y=df['Val'],
        name='Val',
        marker_color=COLORS['secondary'],
    ))
    
    fig.update_layout(
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=16),
        height=500,
        width=1000,
        title_x=0.5,
        margin=dict(t=50, l=50, r=80, b=120),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            font=dict(size=16)
        ),
        xaxis=dict(
            tickangle=45,
            title='',
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title='Count',
            gridcolor='lightgray',
            gridwidth=1,
            tickfont=dict(size=14),
            title_font=dict(size=16)
        ),
        # 调整色标位置和大小
        coloraxis_colorbar=dict(
            x=1.02,        # 将色标移近图表
            xanchor='left', # 左对齐
            len=0.8,       # 色标长度
            thickness=15    # 色标宽度
        )
    )
    
    save_figure(fig, output_dir, 'category_distribution')
    
    # 保存完整的类别分布数据到CSV
    df_full = df.sort_values('Total', ascending=False)
    df_full.to_csv(f'{output_dir}/full_category_distribution.csv', index=False)

def analyze_spatial_distribution(objects, output_dir):
    """分析目标的空间分布（中心点位置）"""
    x_centers = []
    y_centers = []
    
    for obj in objects:
        x_centers.append(obj['center'][0])
        y_centers.append(obj['center'][1])
    
    # 修改热力图的生成方式
    heatmap = np.histogram2d(
        x_centers, 
        y_centers, 
        bins=(50, 50),
        range=[[0, 1], [0, 1]]  # 明确指定范围
    )
    counts = heatmap[0].T
    
    # 使用bin的中心点作为坐标
    x_centers = (heatmap[1][:-1] + heatmap[1][1:]) / 2
    y_centers = (heatmap[2][:-1] + heatmap[2][1:]) / 2
    
    fig = go.Figure(go.Heatmap(
        z=counts,
        x=x_centers,
        y=y_centers,
        colorscale=SCIENTIFIC_COLORSCALE,
        showscale=True
    ))
    
    # 更新布局，使其更紧凑
    fig.update_layout(
        xaxis_title="X coordinate",
        yaxis_title="Y coordinate",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        title_x=0.5,
        width=800,
        height=800,
        margin=dict(t=30, l=50, r=80, b=50),  # 减小顶部边距
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=False,
            range=[0, 1],
            constrain='domain'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=False,
            range=[0, 1],
            constrain='domain',
            scaleanchor="x",
            scaleratio=1
        ),
        # 调整色标的位置和大小
        coloraxis_colorbar=dict(
            x=1.02,        # 将色标移得更近
            xanchor='left',
            len=0.8,       # 色标长度
            thickness=15,   # 色标宽度
            y=0.5,         # 居中对齐
            yanchor='middle'
        )
    )
    
    # 添加这行来保存第一个热力图
    save_figure(fig, output_dir, 'spatial_distribution')
    
    # 对训练集和验证集的热力图也做相同的修改
    train_x = [x for x, obj in zip(x_centers, objects) if obj['split'] == 'train']
    train_y = [y for y, obj in zip(y_centers, objects) if obj['split'] == 'train']
    val_x = [x for x, obj in zip(x_centers, objects) if obj['split'] == 'val']
    val_y = [y for y, obj in zip(y_centers, objects) if obj['split'] == 'val']
    
    train_hist = np.histogram2d(
        train_x, train_y, 
        bins=(50, 50),
        range=[[0, 1], [0, 1]]
    )
    val_hist = np.histogram2d(
        val_x, val_y, 
        bins=(50, 50),
        range=[[0, 1], [0, 1]]
    )
    
    x_centers = (train_hist[1][:-1] + train_hist[1][1:]) / 2
    y_centers = (train_hist[2][:-1] + train_hist[2][1:]) / 2
    
    

def analyze_sizes(objects, output_dir):
    """分析目标尺寸的分布"""
    size_stats = defaultdict(list)
    
    # TT100K图像尺寸是2048x2048
    IMG_WIDTH = 2048
    IMG_HEIGHT = 2048
    
    for obj in objects:
        bbox = obj['bbox']
        # 转换归一化坐标为实际像素坐标
        w = (bbox[2] - bbox[0]) * IMG_WIDTH
        h = (bbox[3] - bbox[1]) * IMG_HEIGHT
        area = w * h
        category = CLASS_NAMES[obj['category']]
        
        size_stats['width'].append(w)
        size_stats['height'].append(h)
        size_stats['area'].append(area)
        size_stats['category'].append(category)
    
    # 根据实际数据分布调整范围
    width_ranges = ['0-32', '32-64', '64-96', '96-128', '128+']
    height_ranges = ['0-32', '32-64', '64-96', '96-128', '128+']
    
    def get_range_index(value):
        if value < 32: return 0
        elif value < 64: return 1
        elif value < 96: return 2
        elif value < 128: return 3
        else: return 4
    
    distribution = np.zeros((5, 5))
    total_count = 0
    for w, h in zip(size_stats['width'], size_stats['height']):
        w_idx = get_range_index(w)
        h_idx = get_range_index(h)
        distribution[h_idx][w_idx] += 1
        total_count += 1
    
    percentage = (distribution / total_count) * 100
    
    text = [[f'{int(distribution[i][j])} ({percentage[i][j]:.1f}%)' 
             for j in range(5)] for i in range(5)]
    
    fig = go.Figure(data=go.Heatmap(
        z=distribution,
        x=width_ranges,
        y=height_ranges,
        colorscale=SCIENTIFIC_COLORSCALE,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 10},
        showscale=True
    ))
    
    fig.update_layout(
        xaxis_title="Width Range (pixels)",
        yaxis_title="Height Range (pixels)",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        width=800,   # 设置固定宽度
        height=800,  # 设置相同高度确保正方形
        margin=dict(t=50, l=80, r=50, b=80),  # 适当调整边距
        xaxis=dict(
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            tickfont=dict(size=14),
            scaleanchor="x",  # 确保y轴与x轴比例相同
            scaleratio=1      # 设置比例为1:1
        )
    )
    
    save_figure(fig, output_dir, 'size_distribution_matrix')
    
    df = pd.DataFrame({
        'Width': size_stats['width'],
        'Height': size_stats['height'],
        'Area': size_stats['area'],
        'Split': size_stats['split']
    })
    
    # 按train/val分别计算统计信息
    stats = pd.DataFrame({
        'Split': ['train', 'val'],
        'Width_Mean': [df[df['Split']=='train']['Width'].mean(), 
                      df[df['Split']=='val']['Width'].mean()],
        'Height_Mean': [df[df['Split']=='train']['Height'].mean(),
                       df[df['Split']=='val']['Height'].mean()],
        'Area_Mean': [df[df['Split']=='train']['Area'].mean(),
                     df[df['Split']=='val']['Area'].mean()],
        'Count': [len(df[df['Split']=='train']), 
                 len(df[df['Split']=='val'])]
    }).round(1)
    
    # 保存到CSV
    stats.to_csv(f'{output_dir}/dataset_size_statistics.csv', index=False)

def save_analysis_to_csv(objects, output_dir):
    """保存基本统计信息到CSV"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 统计图片数量
    train_images = len(set(obj['image_path'] for obj in objects if obj['split'] == 'train'))
    val_images = len(set(obj['image_path'] for obj in objects if obj['split'] == 'val'))
    total_images = train_images + val_images
    
    # 数据集基本统计
    summary = {
        'Metric': [
            'Total_Images',
            'Train_Images',
            'Val_Images',
            'Total_Objects',
            'Train_Objects',
            'Val_Objects',
            'Number_of_Categories',
            'Train/Val_Ratio',
            'Objects_per_Image'
        ],
        'Value': [
            total_images,
            train_images,
            val_images,
            len(objects),
            len([obj for obj in objects if obj['split'] == 'train']),
            len([obj for obj in objects if obj['split'] == 'val']),
            len(CLASS_NAMES),
            len([obj for obj in objects if obj['split'] == 'train']) / 
            len([obj for obj in objects if obj['split'] == 'val']),
            round(len(objects) / total_images, 2)  # 平均每张图片的目标数量
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f'{output_dir}/dataset_statistics.csv', index=False)

def main():
    train_dir = '~/Downloads/mydata/tt100k/labels/train'
    val_dir = '~/Downloads/mydata/tt100k/labels/val'
    output_dir = 'analysis_results'
    figure_dir = f'{output_dir}/figures'
    
    try:
        objects = load_tt100k_from_txt(train_dir, val_dir)
        
        if not objects:
            print("未找到任何数据，请检查输入路径")
            return
            
        # 保存分析结果到CSV
        save_analysis_to_csv(objects, output_dir)
        
        # 可视化分析
        analyze_categories(objects, figure_dir)
        analyze_sizes(objects, figure_dir)
        analyze_spatial_distribution(objects, figure_dir)
        
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 