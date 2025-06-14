import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import random
import torch
import os
import pickle
from IPython import display

display.set_matplotlib_formats("svg")

def check_and_make_directories(directories):
    """检查并创建目录"""
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def load_config(config_path="config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def set_global_seed(seed=42):
    """设置全局随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 设置随机种子
set_global_seed(42)

# from meta import config
# from meta.data_processor import DataProcessor
# from main import check_and_make_directories
# from meta.data_processors.tushare import Tushare, ReturnPlotter
from finrl.meta.env_stock_trading.env_signal_trading import SignalTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
# from meta.config import (
#     DATA_SAVE_DIR,
#     TRAINED_MODEL_DIR,
#     TENSORBOARD_LOG_DIR,
#     RESULTS_DIR,
#     INDICATORS,
# )

import yaml

# 加载配置
config = load_config()

print("所有模块导入完成!")


# 从配置中获取参数
STOCK_TICKER = config['stock']['ticker']
INPUT_DATA_DIR = config['data']['input_dir']
OUTPUT_DATA_DIR = config['data']['output_dir']
PROCESSED_FILE = config['data']['processed_file']

# 技术指标
INDICATORS_OMK = config['indicators']['basic']

# 训练参数
TRAIN_START_DATE = config['training']['train_start_date']
TRAIN_END_DATE = config['training']['train_end_date']
TRADE_START_DATE = config['training']['trade_start_date']
TRADE_END_DATE = config['training']['trade_end_date']

INITIAL_AMOUNT = config['training']['initial_amount']
BUY_COST_PCT = config['training']['buy_cost_pct']
SELL_COST_PCT = config['training']['sell_cost_pct']
FREQUENCY = config['training']['frequency']

# 输出目录
RESULTS_DIR = config['output']['results_dir']
TRAINED_MODEL_DIR = config['output']['trained_model_dir']
TENSORBOARD_LOG_DIR = config['output']['tensorboard_log_dir']
PRINT_VERBOSITY = config['output']['print_verbosity']
MAKE_PLOTS = config['output']['make_plots']
TOTAL_TIMESTEPS = config['output']['total_timesteps']

# 创建必要的文件夹
check_and_make_directories([TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

# ================================
# 配置参数
# ================================

# 单股票标的
ticker_list = ["600000.SH"]  # 浦发银行作为测试标的

# 奖励函数权重配置
REWARD_WEIGHTS = {
    'return': 1.0,        # 收益率权重
    'sharpe': 0.2,        # 夏普比率权重
    'max_drawdown': -1.0, # 最大回撤惩罚权重
    'volatility': -0.1    # 波动率惩罚权重
}

print(f"训练期间: {TRAIN_START_DATE} 到 {TRAIN_END_DATE}")
print(f"交易期间: {TRADE_START_DATE} 到 {TRADE_END_DATE}")
print(f"标的股票: {ticker_list}")

# 读取数据文件
data_file_path = os.path.join(INPUT_DATA_DIR, PROCESSED_FILE)
finrl_data = pd.read_csv(data_file_path)

print(f"数据形状: {finrl_data.shape}")
print(f"数据列: {finrl_data.columns.tolist()}")

# ================================
# 数据分割
# ================================

def data_split(df, start, end, target_date_col="time"):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


train = data_split(finrl_data, TRAIN_START_DATE, TRAIN_END_DATE)
trade = data_split(finrl_data, TRADE_START_DATE, TRADE_END_DATE)

print(f"训练数据形状: {train.shape}")
print(f"交易数据形状: {trade.shape}")
print(f"股票数量: {len(train.tic.unique())}")

# ================================
# 环境设置
# ================================

# 改进的奖励配置
reward_config_balanced = {
    'method': 'information_ratio',  # 使用信息比率方法
    'return_weight': 1.0,
    'risk_penalty_weight': 0.5,
    'trade_quality_weight': 0.1,
    'final_reward_weight': 2.0,
    'benchmark': 'buy_hold'
}

reward_config_multi_factor = {
    'method': 'multi_factor',  # 使用多因子方法
    'return_weight': 1.0,
    'risk_penalty_weight': 0.8,
    'trade_quality_weight': 0.15,
    'final_reward_weight': 1.5,
    'benchmark': 'buy_hold'
}

# 交易环境参数
env_kwargs = {
    "initial_amount": 1000000,
    "buy_cost_pct": 1.25e-3,
    "sell_cost_pct": 1.25e-3,
    "tech_indicator_list": INDICATORS_OMK,
    "turbulence_threshold": None,
    "make_plots": True,
    "print_verbosity": 10,
    "reward_config": reward_config_balanced,  # 使用新的reward_config
    "random_seed": 42,
}

# 创建训练环境
print("创建训练环境...")
e_train_gym = SignalTradingEnv(df=train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()

print(f"动作空间: {e_train_gym.action_space}")
print(f"观测空间: {e_train_gym.observation_space}")
print(f"状态维度: {e_train_gym.state_dim}")

# ================================
# 模型训练
# ================================

def train_signal_models():
    """训练多个模型并比较性能"""
    
    models_config = {
        "PPO": {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
        },
        "A2C": {
            "learning_rate": 0.0007,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "ent_coef": 0.01,
            "vf_coef": 0.25,
        },
        # "DQN": {
        #     "learning_rate": 1e-3,
        #     "buffer_size": 50000,
        #     "learning_starts": 1000,
        #     "batch_size": 32,
        #     "tau": 1.0,
        #     "gamma": 0.99,
        #     "train_freq": 4,
        #     "gradient_steps": 1,
        #     "target_update_interval": 1000,
        # }
    }
    
    trained_models = {}
    
    for model_name, params in models_config.items():
        print(f"\n=== 训练 {model_name} 模型 ===")
        
        # 重新创建环境以避免状态干扰
        e_train_fresh = SignalTradingEnv(df=train, **env_kwargs)
        env_train_fresh, _ = e_train_fresh.get_sb_env()
        
        agent = DRLAgent(env=env_train_fresh)
        
        try:
            model = agent.get_model(model_name.lower(), model_kwargs=params, seed=42)
            trained_model = agent.train_model(
                model=model, 
                tb_log_name=f"{model_name.lower()}_signal_trading", 
                total_timesteps=50000
            )
            trained_models[model_name] = trained_model
            print(f"{model_name} 模型训练完成!")
            
        except Exception as e:
            print(f"{model_name} 模型训练失败: {e}")
            continue
    
    return trained_models

# 训练模型
print("开始模型训练...")
trained_models = train_signal_models()

# ================================
# 模型评估和回测
# ================================

def evaluate_model(model, model_name, trade_data):
    """评估单个模型"""
    print(f"\n=== 评估 {model_name} 模型 ===")
    
    # 创建交易环境
    env_kwargs_trade = env_kwargs.copy()
    env_kwargs_trade["print_verbosity"] = 1
    env_kwargs_trade["make_plots"] = True
    
    e_trade_gym = SignalTradingEnv(df=trade_data, **env_kwargs_trade)
    
    # 执行交易
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=model, environment=e_trade_gym
    )
    
    # 保存结果
    df_account_value.to_csv(f"{RESULTS_DIR}/{model_name.lower()}_account_value.csv", index=False)
    df_actions.to_csv(f"{RESULTS_DIR}/{model_name.lower()}_actions.csv", index=False)
    
    # 计算性能指标
    if not df_account_value.empty:
        initial_value = env_kwargs["initial_amount"]
        final_value = df_account_value['account_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # 计算日收益率
        df_account_value['daily_return'] = df_account_value['account_value'].pct_change()
        daily_returns = df_account_value['daily_return'].dropna()
        
        # 计算风险指标
        sharpe_ratio = 0
        max_drawdown = 0
        volatility = 0
        
        if len(daily_returns) > 1:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
            volatility = daily_returns.std() * np.sqrt(252)
            
            # 计算最大回撤
            cumulative = (1 + daily_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
        
        performance_stats = {
            'Model': model_name,
            'Total Return': f"{total_return:.4f}",
            'Final Value': f"{final_value:.2f}",
            'Sharpe Ratio': f"{sharpe_ratio:.4f}",
            'Max Drawdown': f"{max_drawdown:.4f}",
            'Volatility': f"{volatility:.4f}",
            'Total Trades': len(df_actions) if not df_actions.empty else 0
        }
        
        print("性能统计:")
        for key, value in performance_stats.items():
            print(f"  {key}: {value}")
            
        return df_account_value, df_actions, performance_stats
    
    return None, None, None

# 评估所有模型
print("\n开始模型评估...")
all_results = {}
performance_comparison = []

for model_name, model in trained_models.items():
    try:
        account_value, actions, stats = evaluate_model(model, model_name, trade)
        if stats:
            all_results[model_name] = {
                'account_value': account_value,
                'actions': actions,
                'stats': stats
            }
            performance_comparison.append(stats)
    except Exception as e:
        print(f"评估 {model_name} 模型时出错: {e}")

# ================================
# 结果比较和可视化
# ================================

def plot_net_value_comparison(all_results, trade_data, env_kwargs):
    """
    绘制净值曲线对比图
    包括各模型的净值曲线和股票基准净值曲线
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    
    plt.figure(figsize=(16, 10))
    
    # 创建两个子图：一个显示绝对净值，一个显示相对收益率
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # 准备基准数据（股票本身的净值曲线）
    if not trade_data.empty and 'close' in trade_data.columns:
        trade_dates = pd.to_datetime(trade_data['time'] if 'time' in trade_data.columns else trade_data['date'])
        initial_amount = env_kwargs["initial_amount"]
        
        # 计算可购买的股票数量（基于第一天的价格）
        initial_price = trade_data['close'].iloc[0]
        shares = initial_amount / initial_price
        
        # 计算股票基准净值曲线
        benchmark_values = trade_data['close'] * shares
        benchmark_returns = (benchmark_values / initial_amount - 1) * 100
        
        # 绘制基准净值曲线（绝对值）
        ax1.plot(trade_dates, benchmark_values, 
                label='Stock Benchmark (Buy & Hold)', 
                linewidth=3, color='black', linestyle='--', alpha=0.8)
        
        # 绘制基准收益率曲线
        ax2.plot(trade_dates, benchmark_returns, 
                label='Stock Benchmark (Buy & Hold)', 
                linewidth=3, color='black', linestyle='--', alpha=0.8)
        
        print(f"基准策略信息:")
        print(f"  初始股价: {initial_price:.2f}")
        print(f"  最终股价: {trade_data['close'].iloc[-1]:.2f}")
        print(f"  可购买股数: {shares:.2f}")
        print(f"  基准最终净值: {benchmark_values.iloc[-1]:.2f}")
        print(f"  基准总收益率: {benchmark_returns.iloc[-1]:.2f}%")
    
    # 颜色和线型配置
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    line_styles = ['-', '-', '-', '-', '-', '-']
    
    # 绘制各模型的净值曲线
    for i, (model_name, result) in enumerate(all_results.items()):
        if result['account_value'] is not None:
            account_value = result['account_value']
            model_dates = pd.to_datetime(account_value['date'])
            model_values = account_value['account_value']
            
            # 计算相对收益率（相对于初始金额）
            model_returns = (model_values / initial_amount - 1) * 100
            
            color = colors[i % len(colors)]
            line_style = line_styles[i % len(line_styles)]
            
            # 绘制净值曲线（绝对值）
            ax1.plot(model_dates, model_values, 
                    label=f'{model_name} Strategy', 
                    linewidth=2.5, color=color, linestyle=line_style, alpha=0.9)
            
            # 绘制收益率曲线
            ax2.plot(model_dates, model_returns, 
                    label=f'{model_name} Strategy', 
                    linewidth=2.5, color=color, linestyle=line_style, alpha=0.9)
            
            print(f"{model_name} 策略信息:")
            print(f"  最终净值: {model_values.iloc[-1]:.2f}")
            print(f"  总收益率: {model_returns.iloc[-1]:.2f}%")
    
    # 设置第一个子图（绝对净值）
    ax1.set_title('Portfolio Net Value Comparison', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Portfolio Value (CNY)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加水平线显示初始金额
    ax1.axhline(y=initial_amount, color='gray', linestyle=':', alpha=0.5, label='Initial Amount')
    
    # 格式化y轴显示
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'¥{x:,.0f}'))
    
    # 设置第二个子图（收益率）
    ax2.set_title('Portfolio Return Rate Comparison', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Return Rate (%)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加零线
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    # 格式化日期轴
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/net_value_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建性能统计表
    create_performance_summary_table(all_results, trade_data, initial_amount)

def create_performance_summary_table(all_results, trade_data, initial_amount):
    """创建详细的性能统计表"""
    import matplotlib.pyplot as plt
    
    performance_data = []
    
    # 添加基准数据
    if not trade_data.empty:
        initial_price = trade_data['close'].iloc[0]
        final_price = trade_data['close'].iloc[-1]
        benchmark_return = (final_price - initial_price) / initial_price * 100
        
        # 计算基准的其他指标
        price_returns = trade_data['close'].pct_change().dropna()
        benchmark_volatility = price_returns.std() * np.sqrt(252) * 100
        benchmark_sharpe = (benchmark_return / 100) / (benchmark_volatility / 100) if benchmark_volatility > 0 else 0
        
        # 计算最大回撤
        cumulative_returns = (1 + price_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max * 100
        benchmark_max_dd = drawdowns.min()
        
        performance_data.append({
            'Strategy': 'Stock Benchmark',
            'Total Return (%)': f"{benchmark_return:.2f}",
            'Volatility (%)': f"{benchmark_volatility:.2f}",
            'Sharpe Ratio': f"{benchmark_sharpe:.3f}",
            'Max Drawdown (%)': f"{benchmark_max_dd:.2f}",
            'Final Value (¥)': f"{final_price * (initial_amount / initial_price):,.0f}"
        })
    
    # 添加各模型数据
    for model_name, result in all_results.items():
        if result['account_value'] is not None:
            account_value = result['account_value']
            final_value = account_value['account_value'].iloc[-1]
            total_return = (final_value - initial_amount) / initial_amount * 100
            
            # 计算其他指标
            daily_returns = account_value['account_value'].pct_change().dropna()
            if len(daily_returns) > 1:
                volatility = daily_returns.std() * np.sqrt(252) * 100
                sharpe_ratio = (total_return / 100) / (volatility / 100) if volatility > 0 else 0
                
                # 最大回撤
                cumulative = (1 + daily_returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max * 100
                max_drawdown = drawdown.min()
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
            
            performance_data.append({
                'Strategy': f'{model_name} Strategy',
                'Total Return (%)': f"{total_return:.2f}",
                'Volatility (%)': f"{volatility:.2f}",
                'Sharpe Ratio': f"{sharpe_ratio:.3f}",
                'Max Drawdown (%)': f"{max_drawdown:.2f}",
                'Final Value (¥)': f"{final_value:,.0f}"
            })
    
    # 创建表格
    performance_df = pd.DataFrame(performance_data)
    print("\n" + "="*80)
    print("详细性能对比表")
    print("="*80)
    print(performance_df.to_string(index=False))
    print("="*80)
    
    # 保存到CSV
    performance_df.to_csv(f"{RESULTS_DIR}/detailed_performance_comparison.csv", index=False)
    print(f"详细性能对比表已保存到: {RESULTS_DIR}/detailed_performance_comparison.csv")
    
    return performance_df

# 绘制比较图表
if len(all_results) > 1:
    plot_net_value_comparison(all_results, trade, env_kwargs)

if performance_comparison:
    print("\n=== 模型性能比较 ===")
    comparison_df = pd.DataFrame(performance_comparison)
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(f"{RESULTS_DIR}/model_performance_comparison.csv", index=False)

# ================================
# 保存最终结果
# ================================

# 单独保存训练好的模型（使用模型自己的save方法）
print("\n保存训练好的模型...")
for model_name, model in trained_models.items():
    try:
        model_path = f"{TRAINED_MODEL_DIR}/{model_name.lower()}_signal_trading"
        model.save(model_path)
        print(f"✓ {model_name} 模型已保存到: {model_path}")
    except Exception as e:
        print(f"✗ 保存 {model_name} 模型失败: {e}")

# 准备可序列化的结果数据（排除不可序列化的模型对象）
serializable_results = {
    'performance_comparison': performance_comparison,
    'trade_data_info': {
        'shape': trade.shape if not trade.empty else (0, 0),
        'columns': trade.columns.tolist() if not trade.empty else [],
        'date_range': [trade['time'].min(), trade['time'].max()] if not trade.empty and 'time' in trade.columns else [],
        'stock_symbols': trade['tic'].unique().tolist() if not trade.empty and 'tic' in trade.columns else []
    },
    'env_kwargs': env_kwargs,
    'model_names': list(trained_models.keys()),
    'training_completed': True,
    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

# 添加账户价值数据（不包含模型对象）
if all_results:
    serializable_results['account_values'] = {}
    serializable_results['actions'] = {}
    
    for model_name, result in all_results.items():
        if result['account_value'] is not None:
            serializable_results['account_values'][model_name] = result['account_value'].to_dict('records')
        if result['actions'] is not None:
            serializable_results['actions'][model_name] = result['actions'].to_dict('records')

# 保存可序列化的结果
try:
    with open(f"{RESULTS_DIR}/signal_trading_final_results.pkl", "wb") as f:
        pickle.dump(serializable_results, f)
    print(f"✓ 可序列化结果已保存到: {RESULTS_DIR}/signal_trading_final_results.pkl")
except Exception as e:
    print(f"✗ 保存序列化结果失败: {e}")

# 另外保存为JSON格式（更通用）
try:
    import json
    # 转换为JSON兼容格式
    json_results = serializable_results.copy()
    
    # 将可能的numpy类型转换为Python原生类型
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # 递归转换
    def recursive_convert(obj):
        if isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(v) for v in obj]
        else:
            return convert_numpy(obj)
    
    json_results = recursive_convert(json_results)
    
    with open(f"{RESULTS_DIR}/signal_trading_final_results.json", "w", encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON格式结果已保存到: {RESULTS_DIR}/signal_trading_final_results.json")
except Exception as e:
    print(f"✗ 保存JSON结果失败: {e}")

print("\n=== 信号交易策略开发完成 ===")
print("结果已保存到以下位置:")
print("\n模型文件:")
for model_name in trained_models.keys():
    model_path = f"{TRAINED_MODEL_DIR}/{model_name.lower()}_signal_trading"
    print(f"- {model_path}.zip: {model_name} 训练好的模型")

print("\n结果数据:")
print("- results/signal_trading_final_results.pkl: 完整结果（pickle格式）")
print("- results/signal_trading_final_results.json: 完整结果（JSON格式）")
print("- results/model_performance_comparison.csv: 模型性能比较")
print("- results/net_value_comparison.png: 净值曲线对比图")
print("- results/detailed_performance_comparison.csv: 详细性能对比表")
print("- results/*_account_value.csv: 各模型的资产曲线")
print("- results/*_actions.csv: 各模型的交易动作")

print("\n可视化图表:")
print("- results/net_value_comparison.png: 包含股票基准的净值曲线对比图")
print("  └── 上半部分: 绝对净值对比（包含初始金额参考线）")
print("  └── 下半部分: 收益率对比（包含零线）")
print("- 图表特色:")
print("  └── 股票基准: 黑色虚线（买入持有策略）")
print("  └── 各模型策略: 不同颜色实线")
print("  └── 详细的性能统计信息会在控制台输出")

print("\n如何加载保存的模型:")
for model_name in trained_models.keys():
    print(f"- {model_name}: model = {model_name}(\"ppo\").load(\"{TRAINED_MODEL_DIR}/{model_name.lower()}_signal_trading\")")

# 显示最佳模型
if performance_comparison:
    best_model = max(performance_comparison, key=lambda x: float(x['Total Return']))
    print(f"\n最佳模型: {best_model['Model']}")
    print(f"总收益率: {best_model['Total Return']}")
    print(f"夏普比率: {best_model['Sharpe Ratio']}")
    print(f"最大回撤: {best_model['Max Drawdown']}") 