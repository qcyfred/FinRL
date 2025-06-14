import gym
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv


class SignalTradingEnv(gym.Env):
    """A single stock signal trading environment for OpenAI gym with discrete actions
    
    优化版本：解决了奖励机制、状态空间shift问题、命名混乱等问题
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        initial_amount,
        buy_cost_pct,
        sell_cost_pct,
        tech_indicator_list,
        turbulence_threshold=None,
        make_plots=False,
        print_verbosity=2,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        random_seed=None,
        reward_config=None,  # 改进的奖励配置
    ):
        # 基本参数
        self.day = day
        self.df = df
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.tech_indicator_list = tech_indicator_list
        self.turbulence_threshold = turbulence_threshold
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        # 改进的奖励配置
        self.reward_config = reward_config or {
            'method': 'information_ratio',  # 'multi_factor' or 'information_ratio'
            'return_weight': 1.0,
            'risk_penalty_weight': 0.5,
            'trade_quality_weight': 0.1,
            'final_reward_weight': 2.0,
            'benchmark': 'buy_hold'  # 基准策略
        }

        # 动作空间: 0=无持仓(卖出), 1=多头持有(买入)
        self.action_space = spaces.Discrete(2)
        
        # 状态空间计算 - 解决shift问题，使用延迟的风控指标
        # [现金, 股价, 持仓状态(0/1), 股票数量, 技术指标..., 延迟风控指标...]
        self.state_dim = 4 + len(self.tech_indicator_list) + 4  # +4 for lagged risk indicators
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,)
        )

        # 初始化数据
        self.data = self.df.loc[self.day, :]
        self.terminal = False

        # 初始化交易变量
        self.immediate_reward = 0.0  # 明确命名：即时RL奖励
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        
        # 持仓状态: 0=无持仓, 1=多头持有
        self.position = 0
        
        # 记录变量
        self.portfolio_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.asset_memory = []
        
        # 风控指标 - 分为当前和延迟两套
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_value = self.initial_amount
        self.trade_returns = []  # 每笔交易的收益率
        self.daily_returns = []
        
        # 延迟风控指标（用于状态空间，避免shift问题）- 必须在_initiate_state之前初始化
        self.lagged_risk_indicators = [0.0, 0.0, 0.0, 0.0]
        
        # 交易记录
        self.trade_start_value = None
        self.trade_start_day = None
        
        # 基准收益率记录（用于信息比率计算）
        self.benchmark_returns = []
        self.tracking_errors = []

        # 初始化状态 - 放在所有属性初始化之后
        self.state = self._initiate_state()

    def _get_current_price(self):
        """获取当前股价"""
        return self.data.close

    def _get_total_asset(self):
        """获取当前总资产"""
        return self.state[0] + self.state[3] * self.state[1]  # 现金 + 股票数量 * 股价

    def _calculate_risk_indicators(self):
        """计算当前风控指标"""
        current_value = self._get_total_asset()
        
        # 更新峰值
        if current_value > self.peak_value:
            self.peak_value = current_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value
            
        # 更新最大回撤
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # 计算波动率（基于最近20天的日收益率）
        if len(self.daily_returns) > 1:
            recent_returns = self.daily_returns[-20:] if len(self.daily_returns) >= 20 else self.daily_returns
            volatility = np.std(recent_returns) * np.sqrt(252)  # 年化波动率
        else:
            volatility = 0.0
            
        # 计算夏普比率（基于最近收益率）
        if len(self.daily_returns) > 1 and np.std(self.daily_returns) > 0:
            sharpe = np.mean(self.daily_returns) / np.std(self.daily_returns) * np.sqrt(252)
        else:
            sharpe = 0.0
            
        return self.max_drawdown, self.current_drawdown, volatility, sharpe

    def _get_benchmark_return(self):
        """计算基准收益率（买入持有策略）"""
        if self.day > 0:
            prev_price = self.df.loc[self.day - 1, 'close']
            current_price = self.data.close
            return (current_price - prev_price) / prev_price
        return 0.0

    def _calculate_information_ratio_reward(self, prev_value, current_value):
        """基于信息比率的奖励计算 - 解决指标可加性问题"""
        # 计算策略收益率
        if prev_value > 0:
            strategy_return = (current_value - prev_value) / prev_value
        else:
            strategy_return = 0.0
            
        self.daily_returns.append(strategy_return)
        
        # 计算基准收益率
        benchmark_return = self._get_benchmark_return()
        self.benchmark_returns.append(benchmark_return)
        
        # 计算超额收益
        excess_return = strategy_return - benchmark_return
        
        # 计算跟踪误差（超额收益的标准差）
        if len(self.daily_returns) > 2:
            excess_returns = np.array(self.daily_returns) - np.array(self.benchmark_returns)
            tracking_error = np.std(excess_returns[-20:])  # 使用最近20天的数据
            self.tracking_errors.append(tracking_error)
        else:
            tracking_error = 0.1  # 默认值
            
        # 计算信息比率 (IR = 超额收益 / 跟踪误差)
        if tracking_error > 1e-6:
            information_ratio = excess_return / tracking_error
        else:
            information_ratio = excess_return / 0.01  # 避免除零
            
        # 将IR转换为合适的奖励尺度
        base_reward = information_ratio * 0.01
        
        return base_reward

    def _calculate_multi_factor_reward(self, prev_value, current_value):
        """改进的多因子奖励计算 - 分层设计"""
        # 1. 基础收益奖励
        if prev_value > 0:
            return_rate = (current_value - prev_value) / prev_value
        else:
            return_rate = 0.0
            
        self.daily_returns.append(return_rate)
        
        # 基础收益奖励（主要组件）
        return_reward = self.reward_config['return_weight'] * return_rate
        
        # 2. 风险惩罚（独立计算，避免可加性问题）
        risk_penalty = 0.0
        if len(self.daily_returns) > 5:  # 至少需要一些历史数据
            recent_returns = self.daily_returns[-10:]
            volatility = np.std(recent_returns)
            
            # 超额波动率惩罚
            if volatility > 0.02:  # 日波动率超过2%
                risk_penalty -= self.reward_config['risk_penalty_weight'] * (volatility - 0.02)
                
            # 连续亏损惩罚
            if len([r for r in recent_returns[-3:] if r < 0]) == 3:  # 连续3天亏损
                risk_penalty -= 0.001
        
        return return_reward + risk_penalty

    def _calculate_trade_quality_reward(self):
        """计算交易质量奖励（仅在交易时触发）"""
        trade_reward = 0.0
        
        # 交易成本惩罚
        if hasattr(self, 'prev_position') and self.prev_position != self.position:
            trade_reward -= 0.0001  # 降低交易成本惩罚的影响
            
            # 完成交易的质量评估
            if self.prev_position == 1 and self.position == 0:  # 平仓
                if self.trade_start_value is not None:
                    current_value = self._get_total_asset()
                    trade_return = (current_value - self.trade_start_value) / self.trade_start_value
                    self.trade_returns.append(trade_return)
                    
                    # 交易质量奖励 - 基于收益率和持有时间
                    hold_days = self.day - self.trade_start_day if self.trade_start_day else 1
                    if trade_return > 0:
                        # 盈利交易奖励，考虑持有时间
                        quality_bonus = self.reward_config['trade_quality_weight'] * trade_return
                        if hold_days >= 5:  # 持有超过5天的盈利交易额外奖励
                            quality_bonus *= 1.2
                        trade_reward += quality_bonus
                    else:
                        # 亏损交易惩罚，但不过重
                        trade_reward += self.reward_config['trade_quality_weight'] * trade_return * 0.5
                        
            # 开仓记录
            elif self.prev_position == 0 and self.position == 1:  # 开仓
                self.trade_start_value = self._get_total_asset()
                self.trade_start_day = self.day
                
        return trade_reward

    def _calculate_reward(self, prev_value, current_value):
        """主奖励计算函数 - 整合改进的奖励机制"""
        if self.reward_config['method'] == 'information_ratio':
            base_reward = self._calculate_information_ratio_reward(prev_value, current_value)
        else:  # multi_factor
            base_reward = self._calculate_multi_factor_reward(prev_value, current_value)
            
        # 添加交易质量奖励
        trade_reward = self._calculate_trade_quality_reward()
        
        # 总奖励
        total_reward = base_reward + trade_reward
        
        return total_reward

    def _calculate_final_reward(self):
        """计算基于整体表现的终极奖励 - 数值尺度统一"""
        if len(self.daily_returns) == 0:
            return 0.0
            
        # 1. 总收益率表现
        total_return = (self._get_total_asset() - self.initial_amount) / self.initial_amount
        
        # 2. 与基准比较的超额收益
        if len(self.benchmark_returns) > 0:
            strategy_cumret = (1 + np.array(self.daily_returns)).prod() - 1
            benchmark_cumret = (1 + np.array(self.benchmark_returns)).prod() - 1
            excess_return = strategy_cumret - benchmark_cumret
        else:
            excess_return = total_return
            
        # 3. 风险调整后的终极奖励
        if len(self.daily_returns) > 1:
            sharpe_ratio = np.mean(self.daily_returns) / np.std(self.daily_returns) * np.sqrt(252)
            # 夏普比率超过1.0的部分给额外奖励
            sharpe_bonus = max(0, sharpe_ratio - 1.0) * 0.01
        else:
            sharpe_bonus = 0.0
            
        # 4. 交易效率奖励
        if len(self.trade_returns) > 0:
            win_rate = sum(1 for r in self.trade_returns if r > 0) / len(self.trade_returns)
            trade_efficiency_bonus = (win_rate - 0.5) * 0.005  # 胜率超过50%的部分
        else:
            trade_efficiency_bonus = 0.0
            
        # 计算终极奖励 - 统一数值尺度到即时奖励的量级
        final_reward = self.reward_config['final_reward_weight'] * (
            excess_return * 0.1 +     # 超额收益奖励
            sharpe_bonus +            # 夏普比率奖励
            trade_efficiency_bonus    # 交易效率奖励
        )
        
        return final_reward

    def step(self, action):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        
        if self.terminal:
            print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()

            # 计算最终统计 - 修正命名混乱
            begin_total_asset = self.initial_amount
            end_total_asset = self._get_total_asset()
            total_pnl = end_total_asset - begin_total_asset  # 正确命名：P&L金额
            total_return = total_pnl / begin_total_asset      # 总收益率

            # 计算最终风控指标
            if len(self.daily_returns) > 1:
                sharpe_final = np.mean(self.daily_returns) / np.std(self.daily_returns) * np.sqrt(252)
                volatility_final = np.std(self.daily_returns) * np.sqrt(252)
            else:
                sharpe_final = 0.0
                volatility_final = 0.0

            # 计算终极奖励并加到最后一步
            final_reward = self._calculate_final_reward()
            total_step_reward = self.immediate_reward + final_reward

            if self.print_verbosity > 0 and self.episode % self.print_verbosity == 0:
                print(f"=== Episode {self.episode} Results ===")
                print(f"Total Return: {total_return:.4f}")
                print(f"Total P&L: {total_pnl:.2f}")  # 修正命名
                print(f"Final RL Reward: {final_reward:.4f}")  # 新增：终极RL奖励
                print(f"Max Drawdown: {self.max_drawdown:.4f}")
                print(f"Sharpe Ratio: {sharpe_final:.4f}")
                print(f"Volatility: {volatility_final:.4f}")
                print(f"Total Trades: {self.trades}")
                print(f"Profitable Trades: {sum(1 for r in self.trade_returns if r > 0)}/{len(self.trade_returns)}")
                if len(self.benchmark_returns) > 0:
                    benchmark_total = (1 + np.array(self.benchmark_returns)).prod() - 1
                    print(f"Benchmark Return: {benchmark_total:.4f}")
                    print(f"Excess Return: {(total_return - benchmark_total):.4f}")
                print("=================================")

            return self.state, total_step_reward, self.terminal, {
                'total_return': total_return,
                'total_pnl': total_pnl,
                'final_rl_reward': final_reward,
                'max_drawdown': self.max_drawdown,
                'sharpe': sharpe_final,
                'volatility': volatility_final,
                'total_trades': self.trades
            }

        # 记录前一状态
        prev_value = self._get_total_asset()
        self.prev_position = self.position

        # 执行动作
        if action == 1:  # 买入/持有
            self._go_long()
        else:  # 卖出/保持无持仓
            self._go_neutral()

        # 计算即时奖励
        current_value = self._get_total_asset()
        self.immediate_reward = self._calculate_reward(prev_value, current_value)

        # 更新延迟风控指标（用于下一步的状态空间）
        self._update_lagged_risk_indicators()

        # 更新turbulence
        if self.turbulence_threshold is not None:
            self.turbulence = self.data.get("turbulence", 0)

        # 记录
        self.actions_memory.append(action)
        date = self._get_date()
        
        self.portfolio_memory.append({
            "date": date,
            "total_asset": current_value,
            "cash": self.state[0],
            "position": self.position,
            "stock_num": self.state[3],
            "price": self.state[1],
            "immediate_reward": self.immediate_reward,  # 记录即时奖励
            "action": action,
        })
        self.date_memory.append(date)

        # 更新下一个状态
        self.day += 1
        self.data = self.df.loc[self.day, :]
        self.state = self._update_state()

        return self.state, self.immediate_reward, self.terminal, {}

    def _update_lagged_risk_indicators(self):
        """更新延迟风控指标 - 解决shift问题"""
        # 使用当前计算的风控指标作为下一步的延迟指标
        current_risk = self._calculate_risk_indicators()
        self.lagged_risk_indicators = list(current_risk)

    def _go_long(self):
        """执行多头操作：满仓买入"""
        current_price = self._get_current_price()
        
        if current_price > 0 and self.position == 0:  # 只有在无持仓时才买入
            # 计算可买入股数
            available_cash = self.state[0]
            cost_rate = 1 + self.buy_cost_pct
            max_shares = int(available_cash / (current_price * cost_rate))
            
            if max_shares > 0:
                # 执行买入
                buy_amount = max_shares * current_price
                cost = buy_amount * self.buy_cost_pct
                
                self.state[0] -= (buy_amount + cost)  # 减少现金
                self.state[3] = max_shares  # 设置股票数量
                self.position = 1  # 更新持仓状态
                
                self.cost += cost
                self.trades += 1

    def _go_neutral(self):
        """执行平仓操作：卖出所有股票"""
        current_price = self._get_current_price()
        
        if current_price > 0 and self.position == 1 and self.state[3] > 0:
            # 卖出所有股票
            sell_amount = self.state[3] * current_price
            cost = sell_amount * self.sell_cost_pct
            
            self.state[0] += (sell_amount - cost)  # 增加现金
            self.state[3] = 0  # 清空股票
            self.position = 0  # 更新持仓状态
            
            self.cost += cost
            self.trades += 1

    def reset(self):
        # 重置所有变量
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.position = 0
        
        # 重置记录
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.portfolio_memory = []
        self.asset_memory = []
        
        # 重置风控指标
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_value = self.initial_amount
        self.trade_returns = []
        self.daily_returns = []
        self.trade_start_value = None
        self.trade_start_day = None
        
        # 重置延迟指标和基准记录
        self.lagged_risk_indicators = [0.0, 0.0, 0.0, 0.0]
        self.benchmark_returns = []
        self.tracking_errors = []

        self.episode += 1
        return self.state

    def _initiate_state(self):
        """初始化状态 - 使用延迟风控指标"""
        if self.initial:
            # 初始状态: [现金, 股价, 持仓状态, 股票数量, 技术指标..., 延迟风控指标...]
            tech_indicators = [self.data[tech] for tech in self.tech_indicator_list]
            
            state = [self.initial_amount, self.data.close, 0, 0] + tech_indicators + self.lagged_risk_indicators
        else:
            # 使用之前的状态
            tech_indicators = [self.data[tech] for tech in self.tech_indicator_list]
            
            state = [
                self.previous_state[0],  # 现金
                self.data.close,         # 当前股价
                self.previous_state[2],  # 持仓状态
                self.previous_state[3],  # 股票数量
            ] + tech_indicators + self.lagged_risk_indicators
            
        return state

    def _update_state(self):
        """更新状态 - 使用延迟风控指标"""
        tech_indicators = [self.data[tech] for tech in self.tech_indicator_list]
        
        state = [
            self.state[0],           # 现金
            self.data.close,         # 当前股价
            self.position,           # 持仓状态
            self.state[3],           # 股票数量
        ] + tech_indicators + self.lagged_risk_indicators  # 使用延迟的风控指标
        
        return state

    def _get_date(self):
        """获取当前日期"""
        return self.data.time

    def render(self, mode="human", close=False):
        return self.state

    def get_portfolio_df(self):
        """获取投资组合DataFrame"""
        portfolio_df = pd.DataFrame(self.portfolio_memory)
        if not portfolio_df.empty:
            portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
            portfolio_df.sort_values("date", inplace=True)
        return portfolio_df

    def save_asset_memory(self):
        """保存资产记录"""
        portfolio_df = self.get_portfolio_df()
        if not portfolio_df.empty:
            df_account_value = portfolio_df[["date", "total_asset"]].rename(
                columns={"total_asset": "account_value"}
            )
            return df_account_value
        else:
            return pd.DataFrame(columns=["date", "account_value"])

    def save_action_memory(self):
        """保存动作记录"""
        if len(self.date_memory) > 1 and len(self.actions_memory) > 0:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
            return df_actions
        else:
            return pd.DataFrame(columns=["date", "actions"])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def _make_plot(self):
        """生成图表"""
        portfolio_df = self.get_portfolio_df()
        if not portfolio_df.empty:
            plt.figure(figsize=(15, 10))
            
            # 资产曲线
            plt.subplot(2, 3, 1)
            plt.plot(portfolio_df["date"], portfolio_df["total_asset"], color="r", label="Portfolio Value")
            plt.title("Portfolio Value Over Time")
            plt.xlabel("Date")
            plt.ylabel("Asset Value")
            plt.legend()
            plt.xticks(rotation=45)
            
            # 持仓状态
            plt.subplot(2, 3, 2)
            plt.plot(portfolio_df["date"], portfolio_df["position"], color="b", label="Position")
            plt.title("Position Over Time")
            plt.xlabel("Date")
            plt.ylabel("Position (0=Neutral, 1=Long)")
            plt.legend()
            plt.xticks(rotation=45)
            
            # 动作分布
            plt.subplot(2, 3, 3)
            action_counts = portfolio_df["action"].value_counts()
            plt.bar(action_counts.index, action_counts.values)
            plt.title("Action Distribution")
            plt.xlabel("Action (0=Sell/Hold, 1=Buy/Hold)")
            plt.ylabel("Count")
            
            # 即时奖励分布
            plt.subplot(2, 3, 4)
            if "immediate_reward" in portfolio_df.columns:
                plt.plot(portfolio_df["date"], portfolio_df["immediate_reward"], color="g", label="Immediate RL Reward")
                plt.title("Immediate RL Rewards")
                plt.xlabel("Date")
                plt.ylabel("Immediate Reward")
                plt.legend()
                plt.xticks(rotation=45)
            
            # 累积收益率对比
            plt.subplot(2, 3, 5)
            portfolio_df['strategy_cumret'] = (portfolio_df['total_asset'] / self.initial_amount - 1) * 100
            plt.plot(portfolio_df["date"], portfolio_df['strategy_cumret'], color="r", label="Strategy")
            
            # 如果有基准数据，也绘制基准曲线
            if len(self.benchmark_returns) > 0:
                benchmark_cumret = (np.cumprod(1 + np.array(self.benchmark_returns)) - 1) * 100
                benchmark_dates = portfolio_df["date"][:len(benchmark_cumret)]
                plt.plot(benchmark_dates, benchmark_cumret, color="b", label="Buy & Hold")
            
            plt.title("Cumulative Return Comparison (%)")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return (%)")
            plt.legend()
            plt.xticks(rotation=45)
            
            # 风险指标变化
            plt.subplot(2, 3, 6)
            if len(self.daily_returns) > 1:
                rolling_sharpe = []
                for i in range(10, len(self.daily_returns)):
                    recent_returns = self.daily_returns[i-10:i]
                    if np.std(recent_returns) > 0:
                        sharpe = np.mean(recent_returns) / np.std(recent_returns) * np.sqrt(252)
                    else:
                        sharpe = 0
                    rolling_sharpe.append(sharpe)
                
                if len(rolling_sharpe) > 0:
                    rolling_dates = portfolio_df["date"][10:10+len(rolling_sharpe)]
                    plt.plot(rolling_dates, rolling_sharpe, color="purple", label="Rolling Sharpe")
                    plt.title("Rolling Sharpe Ratio (10-day)")
                    plt.xlabel("Date")
                    plt.ylabel("Sharpe Ratio")
                    plt.legend()
                    plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"results/signal_trading_analysis_{self.episode}.png", dpi=150, bbox_inches='tight')
            plt.close() 