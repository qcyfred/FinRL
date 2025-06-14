import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import argparse
import yaml

def load_config(config_path="config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_data(data_dir, ticker):
    """Load all data files from the specified directory"""
    print(f"Loading data files for {ticker}...")
    
    # Load 5-minute market data
    market_file = os.path.join(data_dir, f"{ticker}_5m.csv")
    market_data = pd.read_csv(market_file,)
    market_data = market_data.iloc[:, 1:]
    print(f"Loaded market data: {market_data.shape}")
    
    # Load daily replay features
    daily_replay_file = os.path.join(data_dir, f"{ticker}_daily_replay.csv")
    daily_replay_data = pd.read_csv(daily_replay_file)
    print(f"Loaded daily replay data: {daily_replay_data.shape}")
    
    # Load intraday features
    intraday_file = os.path.join(data_dir, f"{ticker}_intraday.csv")
    intraday_data = pd.read_csv(intraday_file)
    print(f"Loaded intraday data: {intraday_data.shape}")
    
    # Load daily labels
    labels_file = os.path.join(data_dir, f"{ticker}_INDEX_daily_labels.csv")
    labels_data = pd.read_csv(labels_file)
    print(f"Loaded labels data: {labels_data.shape}")
    
    return market_data, daily_replay_data, intraday_data, labels_data

def preprocess_datetime_columns(market_data, daily_replay_data, intraday_data, labels_data):
    """Convert datetime columns to proper datetime format"""
    print("Processing datetime columns...")
    
    labels_data['datetime'] = pd.to_datetime(labels_data['datetime'])
    labels_data['datetime'] = labels_data['datetime'] + timedelta(hours=15)

    market_data['datetime'] = pd.to_datetime(market_data['datetime'])
    daily_replay_data['datetime'] = pd.to_datetime(daily_replay_data['datetime'])
    intraday_data['datetime'] = pd.to_datetime(intraday_data['datetime'])

    # Add date column for labels merging
    labels_data['date'] = labels_data['datetime'].dt.date
    market_data['date'] = market_data['datetime'].dt.date
    daily_replay_data['date'] = daily_replay_data['datetime'].dt.date
    intraday_data['date'] = intraday_data['datetime'].dt.date
    
    return market_data, daily_replay_data, intraday_data, labels_data

def merge_all_data(market_data, daily_replay_data, intraday_data, labels_data):
    """Merge all data on datetime"""
    print("Merging all data...")
    
    # Start with market data as base
    merged_data = market_data.copy()
    
    # Merge with daily replay data on datetime
    merged_data = pd.merge(merged_data, daily_replay_data, on='datetime', how='inner', suffixes=('', '_daily_replay'))
    print(f"After merging with daily replay: {merged_data.shape}")
    
    # Merge with intraday data on datetime
    merged_data = pd.merge(merged_data, intraday_data, on='datetime', how='inner', suffixes=('', '_intraday'))
    print(f"After merging with intraday: {merged_data.shape}")
    
    # Merge with labels on date (since labels are daily)
    merged_data = pd.merge(merged_data, labels_data, on='datetime', how='left').ffill().dropna()
    merged_data = merged_data[merged_data['datetime'] <= labels_data['datetime'].max()]

    print(f"After merging with labels: {merged_data.shape}")
    
    # Drop duplicate date columns
    cols_to_drop = [col for col in merged_data.columns if col.endswith('_daily_replay') or col.endswith('_intraday')]
    cols_to_drop.extend(['date_x', 'utility', 'date_y'])
    merged_data = merged_data.drop(columns=cols_to_drop, errors='ignore')

    return merged_data

def create_finrf_data_format(data, ticker):
    """Create finrf_data.csv format with required columns"""
    print("Creating finrf_data format...")
    
    data['time'] = data['datetime']
    data['index'] = range(len(data))
    data['tic'] = ticker  # 使用参数化的ticker

    # 调整列的顺序
    first_ordered = [
        'tic',
        'time',
        'index',
    ]

    other_cols = [col for col in data.columns if col not in first_ordered]
    data = data[first_ordered + other_cols]

    return data

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='数据预处理脚本')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--ticker', type=str, help='股票代码 (覆盖配置文件)')
    parser.add_argument('--input_dir', type=str, help='输入数据目录 (覆盖配置文件)')
    parser.add_argument('--output_dir', type=str, help='输出数据目录 (覆盖配置文件)')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 参数优先级：命令行 > 配置文件
    ticker = args.ticker or config['stock']['ticker']
    input_dir = args.input_dir or config['data']['input_dir']
    output_dir = args.output_dir or config['data']['output_dir']
    processed_file = config['data']['processed_file']
    
    print(f"使用配置:")
    print(f"  股票代码: {ticker}")
    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  输出文件: {processed_file}")
    
    # Load all data
    market_data, daily_replay_data, intraday_data, labels_data = load_data(input_dir, ticker)
    
    # Preprocess datetime columns
    market_data, daily_replay_data, intraday_data, labels_data = preprocess_datetime_columns(
        market_data, daily_replay_data, intraday_data, labels_data
    )
    
    # Merge all data
    merged_data = merge_all_data(market_data, daily_replay_data, intraday_data, labels_data)
    
    # Drop rows with missing labels
    merged_data = merged_data.dropna(subset=['label'])
    print(f"Data after dropping missing labels: {merged_data.shape}")
    
    # Save concatenated data
    output_file1 = os.path.join(output_dir, f"{ticker}_data_concat.csv")
    merged_data.to_csv(output_file1, index=False)
    print(f"Saved concatenated data to: {output_file1}")

    # 需要的列 - 从配置文件读取基础指标
    selected_features = [
        'datetime', 'label', 
        'open', 'high', 'low', 'close', 'volume', 
    ] + config['indicators']['basic']
    
    # 确保所有列都存在
    available_features = [col for col in selected_features if col in merged_data.columns]
    if len(available_features) != len(selected_features):
        missing_features = set(selected_features) - set(available_features)
        print(f"警告: 缺少以下特征列: {missing_features}")
    
    merged_data = merged_data[available_features]

    # TODO 这里不能直接过滤2，时间上会断开，直接计算reward会有问题

    # Filter for label == 2
    regime2_data = merged_data[merged_data['label'] == 2].copy()

    regime2_data = merged_data.copy()

    print(f"Regime 2 data shape: {regime2_data.shape}")

    if len(regime2_data) > 0:
        # Create finrf_data format for regime 2 data
        finrf_data_format_data = create_finrf_data_format(regime2_data, ticker)
        
        # Save filtered regime 2 data
        output_file2 = os.path.join(output_dir, processed_file)
        finrf_data_format_data.to_csv(output_file2, index=False)
        print(f"Saved filtered regime 2 data to: {output_file2}")
        
        # Print summary statistics
        print("\n=== Summary ===")
        print(f"Total merged records: {len(merged_data)}")
        print(f"Records with label==2: {len(regime2_data)}")
        print(f"Date range: {merged_data['datetime'].min()} to {merged_data['datetime'].max()}")
        print(f"Label distribution:\n{merged_data['label'].value_counts().sort_index()}")
        print(f"Available features: {available_features}")
    else:
        print("No data found with label == 2")

if __name__ == "__main__":
    main() 