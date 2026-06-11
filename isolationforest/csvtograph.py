import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

# 1. 读取 CSV 文件（假设文件名为 'data.csv'）
df = pd.read_csv('prometheus_data.csv')
# 2. 时间戳处理（确保按时间排序）
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df = df.sort_values('datetime')

# 3. 解析 metric 列（提取 pod 名称，也可提取 namespace）
def parse_metric(metric_str):
    try:
        metric_str_json = metric_str.replace("'", '"')
        metric_dict = json.loads(metric_str_json)
        return metric_dict.get('pod', 'unknown')
    except:
        return 'unknown'

df['pod'] = df['metric'].apply(parse_metric)

# 4. 查看实际时间范围（确认是否真的是1小时数据）
print(f"数据时间范围: {df['datetime'].min()}  ->  {df['datetime'].max()}")
print(f"总数据点数: {len(df)}")

# 5. 绘图设置（针对1小时数据优化）
plt.figure(figsize=(14, 6))   # 宽一点，避免拥挤

# 6. 按 pod 分组绘制（如果只有一个 pod 就直接画）
if df['pod'].nunique() == 1:
    plt.plot(df['datetime'], df['value'], marker='.', linestyle='-', linewidth=1, markersize=2)
    plt.title(f"Pod {df['pod'].iloc[0]} - CPU Usage (Cumulative, 1-hour window)")
else:
    # 如果有多个 pod，取前几个或指定特定 pod
    for pod_name, group in df.groupby('pod'):
        # 可选：只绘制数据点较多的 pod
        if len(group) > 10:
            plt.plot(group['datetime'], group['value'], label=pod_name, marker='.', linewidth=1, markersize=2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xlabel('Time')
plt.ylabel('CPU Usage (seconds)')
plt.title('Container CPU Usage Over 30 minutes')
plt.grid(True, alpha=0.3)

# 7. 优化 X 轴时间显示
#    自动调整刻度标签密度，避免重叠
plt.gcf().autofmt_xdate()          # 自动旋转日期标签
plt.xticks(rotation=45, ha='right')

# 或者手动设置每隔 N 分钟显示一个刻度（例如每10分钟）
import matplotlib.dates as mdates
ax = plt.gca()
# 设置主刻度为每10分钟
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
# 设置刻度标签格式为 时:分
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.tight_layout()
plt.show()