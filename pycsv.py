import requests
import csv
import time
from datetime import datetime

# 配置参数
PROMETHEUS_URL = 'http://127.0.0.1:1355/'  # 当前 Prometheus 地址
QUERY = 'container_cpu_usage_seconds_total{namespace="sock-shop"}'  # PromQL 查询
# 时间范围：这里示例为 2024年5月10日到2024年5月11日
start_time = int(datetime(2026, 6, 11, 15, 0, 0).timestamp())
end_time = int(datetime(2026, 6, 11, 16, 0, 0).timestamp())
step = '15s'  # 数据点间隔

def fetch_and_save_prometheus_data(query, start, end, step):
    api_url = f'{PROMETHEUS_URL}/api/v1/query_range'
    params = {
        'query': query,
        'start': start,
        'end': end,
        'step': step,
    }
    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data['status'] != 'success':
            print("API 请求失败:", data.get('error', '未知错误'))
            return

        results = data['data']['result']
        if not results:
            print("警告：没有找到匹配的时间序列数据。")
            return

        # 写入 CSV
        with open('prometheus_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'value', 'metric'])

            for series in results:
                metric = str(series['metric'])
                for point in series['values']:
                    timestamp = point[0]
                    value = point[1]
                    writer.writerow([timestamp, value, metric])
        print("数据已成功保存到 'prometheus_data.csv'")

    except requests.exceptions.RequestException as e:
        print("网络请求错误:", e)
    except Exception as e:
        print("发生未知错误:", e)

if __name__ == "__main__":
    fetch_and_save_prometheus_data(QUERY, start_time, end_time, step)
    print("脚本执行完毕。")
