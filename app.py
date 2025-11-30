import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import date, datetime
import calendar
import plotly.graph_objects as go 
import math

# --- Configuration ---
# 修正后的配置函数名
st.set_page_config(layout="wide", page_title="J.Bull's Trading Journal")
st.title("J.Bull's Trading Journal & Analysis")

# --- Dynamic File Path Management ---
def get_file_paths(market_name):
    """根据市场名称生成动态文件路径，确保数据隔离。"""
    market_slug = market_name.replace(' ', '_').replace('-', '_').lower()
    return {
        'data_file': f'my_trades_{market_slug}.csv',
        'config_file': f'config_{market_slug}.json',
        'market_slug': market_slug
    }

# --- Config Management Functions ---
def load_config():
    """加载初始资本配置。"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            return float(config.get('initial_capital', 10000.0))
        except json.JSONDecodeError:
            return 10000.0
        except ValueError:
            return 10000.0
    return 10000.0

def save_config(capital):
    """保存初始资本配置。"""
    config = {'initial_capital': capital}
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
    except Exception as e:
        st.error(f"Error saving config file: {e}")

# --- Data Loading and Saving Functions ---
def load_data():
    """从动态设置的文件路径加载交易数据。"""
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df['date'] = pd.to_datetime(df['date'], errors='coerce') 
        df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce') 
        if 'fees' not in df.columns:
             df['fees'] = 0.0
        df['fees'] = pd.to_numeric(df['fees'], errors='coerce') 
        df = df.drop(columns=['strategy'], errors='ignore')
        return df.dropna(subset=['date', 'pnl']) 
    else:
        return pd.DataFrame(columns=['date', 'ticker', 'entry', 'exit', 'size', 'pnl', 'fees'])

def save_data(df):
    """保存交易数据到 CSV 文件。"""
    df.to_csv(DATA_FILE, index=False)

# --- Callback function to handle market change ---
def handle_market_change():
    """切换市场时，重置状态并强制重新加载数据。"""
    new_market = st.session_state.market_selector
    if new_market != st.session_state.current_market:
        st.session_state.current_market = new_market
        if 'trades' in st.session_state:
            del st.session_state.trades
        if 'initial_capital' in st.session_state:
            del st.session_state.initial_capital
        st.rerun() 

def handle_capital_change():
    """处理初始资本更改。"""
    new_capital = st.session_state.cap_input 
    if new_capital != st.session_state.initial_capital:
        st.session_state.initial_capital = new_capital
        save_config(new_capital)
        st.rerun() 

# --- Calculation Helper Functions ---

def calculate_bursa_fees(entry_price, exit_price, position_size):
    if st.session_state.current_market != 'Bursa Malaysia':
        return 0.0
    buy_value = entry_price * position_size
    sell_value = exit_price * position_size
    BROKERAGE_RATE = 0.0010 
    CLEARING_FEE_RATE = 0.0003 
    STAMP_DUTY_RATE = 0.0010 
    total_fees = (buy_value * BROKERAGE_RATE + buy_value * CLEARING_FEE_RATE + buy_value * STAMP_DUTY_RATE) + \
                 (sell_value * BROKERAGE_RATE + sell_value * CLEARING_FEE_RATE + sell_value * STAMP_DUTY_RATE)
    return total_fees

def calculate_max_drawdown(equity_series):
    running_peak = equity_series.expanding(min_periods=1).max()
    drawdown = (equity_series - running_peak) / running_peak
    max_drawdown = drawdown.min() * 100
    return max_drawdown

def calculate_period_summary(group):
    total_trades = len(group)
    if total_trades == 0:
        return pd.Series({'Trades': 0, 'Wins': 0, 'Losses': 0, 'Win Rate (%)': 0.0, 'Win/Loss Ratio': 0.0, 'Avg Gain ($)': 0.0, 'Avg Loss ($)': 0.0, 'R-Ratio': 0.0, 'Total P&L ($)': 0.0})

    winning_trades = group[group['pnl'] > 0]
    losing_trades = group[group['pnl'] <= 0]
    num_wins = len(winning_trades)
    num_losses = len(losing_trades)
    
    win_rate = (num_wins / total_trades) * 100
    avg_gain = winning_trades['pnl'].mean() if num_wins > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if num_losses > 0 else 0
    # Use np.inf for division by zero
    win_loss_ratio = num_wins / num_losses if num_losses > 0 else np.inf
    reward_risk = abs(avg_gain / avg_loss) if avg_loss != 0 else np.inf

    return pd.Series({
        'Trades': total_trades,
        'Wins': num_wins,
        'Losses': num_losses,
        'Win Rate (%)': win_rate,
        'Win/Loss Ratio': win_loss_ratio,
        'Avg Gain ($)': avg_gain,
        'Avg Loss ($)': avg_loss,
        'R-Ratio': reward_risk,
        'Total P&L ($)': group['pnl'].sum(),
    })

def calculate_periodic_mdd_and_returns(df, initial_capital, period_freq='M'):
    if df.empty: return pd.DataFrame()
    df_chrono = df.sort_values('date', ascending=True).reset_index(drop=True)
    df_chrono['date'] = pd.to_datetime(df_chrono['date'], errors='coerce')
    df_chrono['Period'] = df_chrono['date'].dt.to_period(period_freq).astype(str)
    df_chrono['Capital_End_of_Trade'] = initial_capital + df_chrono['pnl'].cumsum()
    
    # Use include_groups=False (default for groupby().apply in pandas 2.0+)
    summary_df = df_chrono.groupby('Period').apply(calculate_period_summary).reset_index().set_index('Period')
    
    mdd_results = {}
    period_capital = {}
    for period, group_indices in df_chrono.groupby('Period').groups.items():
        period_trades = df_chrono.loc[group_indices]
        first_trade_index = period_trades.index[0]
        start_cap = initial_capital if first_trade_index == 0 else df_chrono.loc[first_trade_index - 1, 'Capital_End_of_Trade']
        end_cap = period_trades['Capital_End_of_Trade'].iloc[-1]
        period_capital[period] = {'Start': start_cap, 'End': end_cap}
        equity_series_for_mdd = pd.concat([pd.Series([start_cap]), period_trades['pnl'].cumsum() + start_cap], ignore_index=True)
        mdd = calculate_max_drawdown(equity_series_for_mdd)
        mdd_results[period] = abs(mdd)
        
    summary_df['Max Drawdown (%)'] = summary_df.index.map(mdd_results)
    summary_df['Period_End_Capital'] = summary_df.index.map(lambda p: period_capital[p]['End'])
    summary_df['Period_Start_Capital'] = summary_df.index.map(lambda p: period_capital[p]['Start'])
    summary_df['Monthly Return (%)'] = (summary_df['Total P&L ($)'] / summary_df['Period_Start_Capital']) * 100
    summary_df['Return vs Initial Capital (%)'] = ((summary_df['Period_End_Capital'] - initial_capital) / initial_capital) * 100
    summary_df = summary_df.reset_index()
    try:
        summary_df['Year'] = pd.to_datetime(summary_df['Period'].astype(str).str[:4], format='%Y', errors='coerce').dt.year
    except:
        summary_df['Year'] = summary_df['Period'].astype(str).str[:4].astype(int, errors='ignore')
        
    if 'Year' in summary_df.columns and not summary_df['Year'].isnull().all():
        # Calculate yearly returns (YTD concept applied to full years)
        # Find the starting capital for each year
        year_start_capital = summary_df.groupby('Year')['Period_Start_Capital'].transform('first')
        # Find the ending capital for each year (last Period_End_Capital of the year)
        year_end_capital = summary_df.groupby('Year')['Period_End_Capital'].transform('last')
        
        # Calculate P&L for the year
        yearly_pnl = year_end_capital - year_start_capital
        
        # Calculate yearly return
        summary_df['Yearly Return (Total) (%)'] = (yearly_pnl / year_start_capital) * 100
        
    else:
        summary_df['Yearly Return (Total) (%)'] = 0.0
        
    cols_to_drop = ['Period_End_Capital', 'Period_Start_Capital']
    if period_freq == 'M': cols_to_drop.append('Year')
    
    summary_df = summary_df.drop(columns=[col for col in cols_to_drop if col in summary_df.columns], errors='ignore').sort_values('Period', ascending=False)
    return summary_df


def calculate_metrics(df, initial_capital):
    # Adjusted return list (15 values, removing total_rr)
    # Total trades, total pnl, win rate %, avg gain, avg loss, final capital, avg_rr, mdd, return vs initial %, expectancy $, expectancy %, total fees, avg gain %, avg loss %, adj_rr
    if df.empty: return 0, 0, 0, 0, 0, initial_capital, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 
    total_trades = len(df)
    total_pnl = df['pnl'].sum()
    total_fees = df['fees'].sum()
    equity_series = initial_capital + df['pnl'].cumsum()
    winning_trades = df[df['pnl'] > 0]
    losing_trades = df[df['pnl'] <= 0]
    win_rate_decimal = len(winning_trades) / total_trades
    loss_rate_decimal = 1 - win_rate_decimal
    win_rate_pct = win_rate_decimal * 100
    avg_gain = winning_trades['pnl'].mean() if not winning_trades.empty else 0
    avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
    avg_win_value = (winning_trades['entry'] * winning_trades['size']).mean() if not winning_trades.empty else 0
    avg_loss_value = (losing_trades['entry'] * losing_trades['size']).mean() if not losing_trades.empty else 0
    avg_gain_pct = (avg_gain / avg_win_value) * 100 if avg_win_value > 0 else 0
    avg_loss_pct = (abs(avg_loss) / avg_loss_value) * 100 if avg_loss_value > 0 else 0
    
    # 1. Avg. R:R (Average Gain / Average Loss)
    avg_rr = abs(avg_gain / avg_loss) if avg_loss != 0 else np.inf
    
    # 2. Total R:R is removed
    
    # 3. Adj. R:R (Average Gain * Win Rate) / (Average Loss * Loss Rate)
    adj_rr_numerator = avg_gain * win_rate_decimal
    adj_rr_denominator = abs(avg_loss) * loss_rate_decimal
    adj_rr = adj_rr_numerator / adj_rr_denominator if adj_rr_denominator != 0 else np.inf
    
    final_capital = equity_series.iloc[-1]
    max_drawdown = calculate_max_drawdown(equity_series)
    max_drawdown_magnitude = abs(max_drawdown)
    return_vs_initial_capital_pct = (total_pnl / initial_capital) * 100 if initial_capital != 0 else 0
    expectancy = (avg_gain * win_rate_decimal) - (abs(avg_loss) * loss_rate_decimal)
    avg_position_value = (df['entry'] * df['size']).mean() if not df.empty else 0
    expectancy_pct = (expectancy / avg_position_value) * 100 if avg_position_value != 0 else 0
    
    # Adjusted return list (15 values, removed total_rr)
    return total_trades, total_pnl, win_rate_pct, avg_gain, avg_loss, final_capital, avg_rr, max_drawdown_magnitude, return_vs_initial_capital_pct, expectancy, expectancy_pct, total_fees, avg_gain_pct, avg_loss_pct, adj_rr


def get_period_initial_capital(df_all_trades, start_date, overall_initial_capital):
    if df_all_trades.empty:
        return overall_initial_capital
        
    df_all_trades['date'] = pd.to_datetime(df_all_trades['date'], errors='coerce')
    start_date = pd.to_datetime(start_date)
    
    df_before = df_all_trades[df_all_trades['date'] < start_date]
    
    if df_before.empty:
        return overall_initial_capital
    else:
        pnl_before = df_before['pnl'].sum()
        return overall_initial_capital + pnl_before


# --- CHARTING HELPER FUNCTIONS ---

def plot_equity_curve_plotly(df, overall_initial_capital, start_capital_for_period, market_slug):
    if df.empty: return go.Figure()
    df_sorted = df.sort_values(by='date').reset_index(drop=True)
    df_sorted['Capital'] = start_capital_for_period + df_sorted['pnl'].cumsum()
    
    # 确定货币符号
    currency_symbol = '$' if market_slug == 'us_market' else 'RM'
    
    fig = go.Figure()
    
    # 1. 绘制 Equity Curve 轨迹 (主轨迹)
    fig.add_trace(go.Scatter(
        x=df_sorted['date'], 
        y=df_sorted['Capital'], 
        mode='lines', 
        line=dict(color='#00CC96', width=2), 
        name='Equity'
    ))
    
    # 2. FIX: 用 Scatter Trace 替代 fig.add_hline 来实现悬停文本
    dates = df_sorted['date']
    
    if not dates.empty:
        # 悬停文本格式化为货币
        hover_text = [f"Start Capital: {currency_symbol}{start_capital_for_period:,.2f}" for _ in range(len(dates))]

        fig.add_trace(go.Scatter(
            x=dates,
            y=[start_capital_for_period] * len(dates), # 所有点的 Y 值都是起始资本
            mode='lines',
            line=dict(color='gray', width=1, dash='dash'), # 虚线样式
            name='Start Capital',
            hoverinfo='text',
            hovertext=hover_text,
            showlegend=False # 不显示在图例中，但保留悬停功能
        ))

    fig.update_layout(title=None, 
                      xaxis_title="Date", 
                      yaxis_title="Capital", 
                      margin=dict(l=0, r=0, t=10, b=0), 
                      height=350, 
                      hovermode="x unified", 
                      showlegend=False)
    
    return fig

def plot_monthly_pnl_plotly(monthly_data):
    if monthly_data.empty: return go.Figure()
    colors = ['#00CC96' if v >= 0 else '#EF553B' for v in monthly_data['Net P&L']]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly_data['Period'], y=monthly_data['Net P&L'], marker_color=colors, text=monthly_data['Net P&L'].apply(lambda x: f"${x:,.0f}"), textposition='auto'))
    fig.update_layout(title=None,xaxis_title="Month",yaxis_title="Net P&L",margin=dict(l=0, r=0, t=10, b=0),height=350,showlegend=False)
    return fig


# --- Calendar Functions (CSS 优化 - 移除 Metric/Privacy CSS) ---
def inject_calendar_css():
    st.markdown("""
    <style>
    /* 移除 Metric/Privacy 相关的 CSS 调整，只保留日历样式 */
    
    .calendar-day-box {
        border-radius: 5px;padding: 10px 5px 5px 5px;margin: 3px 0;color: white;min-height: 80px; 
        display: flex;flex-direction: column;justify-content: space-between;text-align: center;box-shadow: 1px 1px 5px rgba(0, 0, 0, 0.3);
    }
    .pnl-positive { background-color: #28a745; }
    .pnl-negative { background-color: #dc3545; }
    .pnl-neutral { background-color: #6c757d; }
    .day-number { font-size: 1.1em; font-weight: bold; margin-bottom: 5px; text-align: left; }
    .trade-summary { font-size: 0.7em; line-height: 1.2; text-align: right; }
    </style>
    """, unsafe_allow_html=True)

def get_daily_summary(df):
    if df.empty or 'pnl' not in df.columns or 'date' not in df.columns: return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
    df_daily = df.groupby(df['date'].dt.date).agg(
        total_pnl=('pnl', 'sum'),trade_count=('date', 'size'),wins=('pnl', lambda x: (x > 0).sum()),losses=('pnl', lambda x: (x <= 0).sum())
    ).reset_index()
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily.set_index('date', inplace=True)
    return df_daily

def render_trading_calendar(df_daily, market_slug):
    st.header("Daily Trading Calendar")
    if df_daily.empty: st.info("No trading data available to generate calendar.")
    else:
        latest_date = df_daily.index.max().date()
        earliest_date = df_daily.index.min().date()
        col_year, col_month = st.columns(2)
        years = range(earliest_date.year, latest_date.year + 1)
        # Handle case where selected year might not be in years list
        try:
            default_year_index = list(years).index(latest_date.year)
        except ValueError:
            default_year_index = len(years)-1 if years else 0

        selected_year = col_year.selectbox("Select Year", options=list(years), index=default_year_index)
        available_months = df_daily.index[df_daily.index.year == selected_year].month.unique().sort_values()
        
        # Determine default month to show
        default_month = latest_date.month
        if selected_year == latest_date.year:
            months_to_show = available_months
        elif selected_year == earliest_date.year:
             months_to_show = available_months
             default_month = earliest_date.month
        else:
            months_to_show = available_months
            default_month = months_to_show.max() if not months_to_show.empty else 1
        
        # Determine default index for month selectbox
        if not months_to_show.empty and default_month in months_to_show.tolist():
             default_index = list(months_to_show).index(default_month)
        elif not months_to_show.empty:
            default_index = 0
        else:
            # If no data for the year, this should not happen if logic is correct, but safe fallback
            selected_month = 1 
            default_index = 0
            
        if not months_to_show.empty:
            selected_month = col_month.selectbox("Select Month", options=months_to_show, format_func=lambda x: datetime(2000, x, 1).strftime('%B'), index=default_index)
        else:
            st.info(f"No trade data found for year {selected_year}.")
            return

        try:
            start_date = date(selected_year, selected_month, 1)
            month_length = calendar.monthrange(selected_year, selected_month)[1]
            end_date = date(selected_year, selected_month, month_length)
        except ValueError: return
        
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D').normalize()
        daily_data_for_month = df_daily.reindex(all_dates, fill_value=0)
        weekdays = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        cols_header = st.columns(7)
        for i, day in enumerate(weekdays):
            cols_header[i].markdown(f"<div style='text-align: center;'>**{day}**</div>", unsafe_allow_html=True)
            
        first_day_weekday = start_date.weekday() 
        start_col_index = (first_day_weekday + 1) % 7 
        current_col = start_col_index
        calendar_cols = st.columns(7)
        for i in range(start_col_index): calendar_cols[i].markdown("", unsafe_allow_html=True) 
        currency_symbol = '$' if MARKET_SLUG == 'us_market' else 'RM'
        for dt, row in daily_data_for_month.iterrows():
            day_number = dt.day
            pnl = row['total_pnl']
            trade_count = int(row['trade_count'])
            wins = int(row['wins'])
            if trade_count == 0: color_class = "pnl-neutral"
            elif pnl > 0: color_class = "pnl-positive"
            else: color_class = "pnl-negative"
            win_rate_str = "N/A"
            if trade_count > 0:
                win_rate = (wins / trade_count) * 100
                win_rate_str = f"{win_rate:.0f}% Win"
            pnl_str = f"{currency_symbol}{pnl:,.2f}"
            cell_html = f"""
            <div class="calendar-day-box {color_class}">
                <span class="day-number">{day_number}</span>
                <div class="trade-summary">
                    {pnl_str} <br>
                    {trade_count} trades <br>
                    {win_rate_str}
                </div>
            </div>
            """
            calendar_cols[current_col].markdown(cell_html, unsafe_allow_html=True)
            current_col += 1
            if current_col > 6:
                current_col = 0
                calendar_cols = st.columns(7)


# --- Core Initialization ---
if 'current_market' not in st.session_state:
    st.session_state.current_market = 'US Market'
if 'initial_capital' not in st.session_state:
    st.session_state.initial_capital = 10000.0

# 1. Set dynamic file paths based on the current market
current_paths = get_file_paths(st.session_state.current_market)
DATA_FILE = current_paths['data_file']
CONFIG_FILE = current_paths['config_file']
MARKET_SLUG = current_paths['market_slug']

# 2. THE FIX: Force Data and Config Load upon Rerun/Initialization
if 'trades' not in st.session_state:
    st.session_state.trades = load_data() 
    st.session_state.initial_capital = load_config()


# --- Sidebar ---
with st.sidebar:
    st.header("Journal Management")
    markets = ['US Market', 'Bursa Malaysia'] 
    selected_market = st.selectbox("Select Trading Journal:", markets, index=markets.index(st.session_state.current_market) if st.session_state.current_market in markets else 0, key='market_selector', on_change=handle_market_change)
    st.markdown(f"**Current Data File:** `{DATA_FILE}`")
    if st.session_state.current_market == 'Bursa Malaysia':
        st.markdown("**Bursa Malaysia Fees Applied:**\n- Brokerage: 0.10%\n- Clearing Fees: 0.03%\n- Stamp Duty: 0.10%")
    st.markdown("---")
    st.header("Data Entry & Management")
    st.number_input("Initial Trading Capital:", min_value=100.0, value=st.session_state.initial_capital, step=1000.0, key='cap_input', on_change=handle_capital_change)
    st.markdown("---")
    st.subheader("Add New Trade")
    default_ticker = "MAYBANK" if st.session_state.current_market == 'Bursa Malaysia' else "AAPL"
    trade_date = st.date_input("Trade Date:", date.today())
    ticker = st.text_input("Ticker:", value=default_ticker) 
    entry_price = st.number_input("Entry Price:", min_value=0.01, step=0.01, format="%.4f")
    exit_price = st.number_input("Exit Price:", min_value=0.01, step=0.01, format="%.4f")
    position_size = st.number_input("Position Size:", min_value=1, step=100)
    
    if st.button("Add Trade"):
        if entry_price > 0 and exit_price > 0 and position_size > 0:
            gross_pnl = (exit_price - entry_price) * position_size
            fees = calculate_bursa_fees(entry_price, exit_price, position_size)
            net_pnl = gross_pnl - fees
            new_trade = pd.DataFrame([{'date': trade_date.strftime('%Y-%m-%d'), 'ticker': ticker, 'entry': entry_price, 'exit': exit_price, 'size': position_size, 'pnl': net_pnl, 'fees': fees}])
            st.session_state.trades = pd.concat([st.session_state.trades, new_trade], ignore_index=True)
            save_data(st.session_state.trades) 
            st.success(f"Trade recorded: {ticker}, Net P&L {net_pnl:,.2f}, Fees {fees:,.2f}")
            st.rerun() 
        else:
            st.error("Invalid input.")
    st.markdown("---")
    st.header("Remove Trade Record")
    if not st.session_state.trades.empty:
        max_idx = len(st.session_state.trades) - 1
        idx_to_remove = st.number_input(f"Enter Index (0 to {max_idx}):", min_value=0, max_value=max_idx, step=1)
        if st.button("Confirm Remove"):
            st.session_state.trades = st.session_state.trades.drop(index=idx_to_remove).reset_index(drop=True)
            save_data(st.session_state.trades) 
            st.success("Trade removed.")
            st.rerun()


# --- Main App Logic ---
if st.session_state.trades.empty:
    st.info("Welcome! Please enter your first trade in the sidebar. (Current selected journal has no data).")
else:
    # Calculate metrics - 15 values returned (removed total_rr)
    total_trades, total_pnl, win_rate_pct, avg_gain, avg_loss, final_capital, avg_rr, max_drawdown, return_vs_initial_capital_pct, expectancy, expectancy_pct, total_fees, avg_gain_pct, avg_loss_pct, adj_rr = calculate_metrics(st.session_state.trades, st.session_state.initial_capital)
    
    inject_calendar_css()

    tab_dashboard, tab_calendar, tab_raw_data = st.tabs(["Dashboard", "Daily Calendar", "Raw Data"])

    # --- TAB 1: THE DASHBOARD ---
    with tab_dashboard:
        st.header(f"{st.session_state.current_market} Key Performance Indicators (KPIs)")
        
        # ADJUSTED TO 5 COLUMNS
        c1, c2, c3, c4, c5 = st.columns(5)
        
        # c1: Total Trades
        c1.metric("Total Trades", total_trades)
        
        # c2: Current Capital 
        delta_str = f"${total_pnl:,.2f} ({return_vs_initial_capital_pct:.2f}%) (Net P&L)"
        c2.metric(
            "Current Capital", 
            value=f"${final_capital:,.2f}", 
            delta=delta_str, 
            delta_color="inverse" if total_pnl < 0 else "normal"
        )
        
        # c3: Max Drawdown
        c3.metric("Max Drawdown (MDD)", f"{max_drawdown:.2f}%", delta_color="inverse")
        
        # c4: Avg. R:R (Average Gain / Average Loss)
        avg_rr_disp = f"{avg_rr:.2f}" if avg_rr != np.inf else "INF"
        c4.metric("Avg. R:R", avg_rr_disp)
        
        # c5: Adj. R:R (Avg Gain * Win Rate / Avg Loss * Loss Rate)
        adj_rr_disp = f"{adj_rr:.2f}" if adj_rr != np.inf else "INF"
        c5.metric("Adj. R:R (Profit Factor)", adj_rr_disp)
        
        st.markdown("---")
        
        st.subheader("Trading Efficiency")
        e1, e2, e3 = st.columns(3)
        
        # e1: Win Rate (%)
        e1.metric("Win Rate (%)", f"{win_rate_pct:.2f}%")
        
        # e2: Expectancy ($)
        e2.metric("Expectancy ($)", f"${expectancy:,.2f}")
        
        # e3: Expectancy (%)
        e3.metric("Expectancy (%)", f"{expectancy_pct:.2f}%")
        
        st.markdown("---")
        
        st.subheader("Average Trade Profit/Loss")
        a1, a2, a3, a4 = st.columns(4)
        
        # a1: Avg Gain ($)
        a1.metric("Avg Gain ($)", f"${avg_gain:,.2f}")

        # a2: Avg Loss ($)
        a2.metric("Avg Loss ($)", f"${avg_loss:,.2f}", delta_color="inverse")
        
        # a3: Avg Gain (%)
        a3.metric("Avg Gain (%)", f"{avg_gain_pct:.2f}%")

        # a4: Avg Loss (%)
        a4.metric("Avg Loss (%)", f"{abs(avg_loss_pct):.2f}%", delta_color="inverse")
        
        st.markdown("---")
        
        # --- Equity Curve (Line Chart + Period Filter) ---
        st.header("Equity Curve (Net of Fees)")
        col_select, col_empty = st.columns([1, 4])
        
        filter_option = col_select.selectbox(
            "Select Equity Curve Period:",
            ["All Time", "Year to Date (YTD)", "Month to Date (MTD)"],
            index=0
        )
        
        # --- Filtering Logic ---
        df_all_trades = st.session_state.trades.copy()
        current_date = pd.to_datetime(date.today())
        
        df_filtered = df_all_trades
        period_start_capital = st.session_state.initial_capital
        
        if not df_all_trades.empty:
            df_all_trades['date'] = pd.to_datetime(df_all_trades['date'], errors='coerce')
            
            if filter_option == "Year to Date (YTD)":
                start_date = pd.to_datetime(f'{current_date.year}-01-01')
                period_start_capital = get_period_initial_capital(df_all_trades, start_date, st.session_state.initial_capital)
                df_filtered = df_all_trades[df_all_trades['date'] >= start_date]

            elif filter_option == "Month to Date (MTD)":
                start_date = pd.to_datetime(f'{current_date.year}-{current_date.month:02d}-01')
                period_start_capital = get_period_initial_capital(df_all_trades, start_date, st.session_state.initial_capital)
                df_filtered = df_all_trades[df_all_trades['date'] >= start_date]

        if not df_filtered.empty:
            fig_equity = plot_equity_curve_plotly(df_filtered, st.session_state.initial_capital, period_start_capital, MARKET_SLUG)
            
            # HIDDEN MODEBAR
            st.plotly_chart(
                fig_equity, 
                use_container_width=True,
                config={'displayModeBar': False} 
            )
        else:
            st.info(f"No trading data available for the selected period: **{filter_option}**.")
            
        
        # Helper for coloring tables
        def color_pnl(val):
            if isinstance(val, (float, int)) and val != np.inf:
                if val < 0: return 'color: red'
                elif val > 0: return 'color: green'
            return ''
        
        with st.expander("Ticker Performance Summary & Fees (Click to expand)"):
            if not st.session_state.trades.empty:
                if st.session_state.current_market == 'Bursa Malaysia':
                    st.subheader("Total Trading Fees (Bursa Malaysia)")
                    st.metric("Total Fees Paid", f"${total_fees:,.2f}", delta_color="off")
                    st.markdown("---")
                st.subheader("Performance by Ticker (Net P&L)")
                ticker_sum = st.session_state.trades.groupby('ticker')['pnl'].agg(['count', 'sum', 'mean'])
                ticker_sum = ticker_sum.rename(columns={'count': 'Trades', 'sum': 'Total Net P&L', 'mean': 'Avg Net P&L'})
                st.dataframe(ticker_sum.style.format({'Total Net P&L': '${:,.2f}', 'Avg Net P&L': '${:,.2f}'}).applymap(color_pnl, subset=['Total Net P&L', 'Avg Net P&L']), use_container_width=True)
        
        with st.expander("Detailed Performance Statistics (Yearly & Monthly)"):
            if not st.session_state.trades.empty:
                df_stats = st.session_state.trades.copy()
                df_stats['date'] = pd.to_datetime(df_stats['date'])
                
                # --- FIX: Custom Formatter for Ratios to handle 'INF' string ---
                # This function handles the case where the value is the string 'INF'
                def ratio_formatter(val):
                    # Check if the value is the string 'INF'
                    if isinstance(val, str) and val == 'INF':
                        return val
                    # Check for NaN/None and handle it with 'N/A'
                    if pd.isna(val) or val is None:
                        return 'N/A'
                    # Otherwise, apply the standard numeric format
                    return f'{val:.2f}'

                # Formatters Dictionary
                fmt_dict = {
                    'Yearly Return (Total) (%)': '{:.2f}%', 
                    'Max Drawdown (%)': '{:.2f}%', 
                    'Win Rate (%)': '{:.2f}%', 
                    # Use custom formatter for columns that might contain the string 'INF'
                    'Win/Loss Ratio': ratio_formatter,
                    'R-Ratio': ratio_formatter,
                    'Avg Gain ($)': '${:,.2f}', 
                    'Avg Loss ($)': '${:,.2f}', 
                    'Net P&L': '${:,.2f}',
                    'Monthly Return (%)': '{:.2f}%'
                }
                
                def fmt_table(d):
                    d['Trades (W/L)'] = d.apply(lambda r: f"{r['Trades']} ({r['Wins']}/{r['Losses']})", axis=1)
                    # Replace np.inf with the string 'INF' *before* formatting
                    d = d.replace([np.inf, -np.inf], 'INF').rename(columns={'Total P&L ($)': 'Net P&L'}) 
                    
                    display_cols = [c for c in d.columns if c in fmt_dict.keys() or c in ['Period', 'Trades (W/L)']]
                    if 'Yearly Return (Total) (%)' in display_cols:
                        cols = ['Period', 'Trades (W/L)', 'Yearly Return (Total) (%)', 'Max Drawdown (%)', 'Win Rate (%)', 'Win/Loss Ratio', 'Avg Gain ($)', 'Avg Loss ($)', 'R-Ratio', 'Net P&L']
                    else:
                        cols = ['Period', 'Trades (W/L)', 'Monthly Return (%)', 'Max Drawdown (%)', 'Win Rate (%)', 'Win/Loss Ratio', 'R-Ratio', 'Avg Gain ($)', 'Avg Loss ($)', 'Net P&L']
                    
                    # Pass the fmt_dict which now contains the custom function for the string columns
                    return d[cols].style.format(fmt_dict, na_rep='N/A').applymap(color_pnl, subset=[c for c in cols if c in ['Yearly Return (Total) (%)', 'Monthly Return (%)', 'Win Rate (%)', 'Avg Gain ($)', 'Avg Loss ($)', 'Net P&L']])

                yearly_sum = calculate_periodic_mdd_and_returns(df_stats.copy(), st.session_state.initial_capital, 'Y')
                monthly_sum = calculate_periodic_mdd_and_returns(df_stats.copy(), st.session_state.initial_capital, 'M')
                
                if 'Year' not in monthly_sum.columns and 'Period' in monthly_sum.columns:
                     try:
                        monthly_sum['Year'] = pd.to_datetime(monthly_sum['Period'].astype(str).str[:4], format='%Y', errors='coerce').dt.year
                     except:
                         monthly_sum['Year'] = monthly_sum['Period'].astype(str).str[:4].astype(int, errors='ignore')

                t_y, t_m = st.tabs(["Yearly Summary", "Monthly Details"])
                with t_y:
                    st.header("Annual Performance Summary")
                    st.dataframe(fmt_table(yearly_sum), use_container_width=True)
                with t_m:
                    st.header("Monthly Trading Details")
                    
                    st.subheader("Monthly Net P&L Chart")
                    monthly_chart_data = monthly_sum.sort_values('Period', ascending=True).rename(columns={'Total P&L ($)': 'Net P&L'})
                    fig_monthly = plot_monthly_pnl_plotly(monthly_chart_data)
                    st.plotly_chart(fig_monthly, use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("Monthly Trading Statistics by Year (Expand by Year)")
                    if 'Year' in monthly_sum.columns:
                        uy = monthly_sum['Year'].dropna().unique()
                        uy.sort()
                        for y in uy[::-1]:
                            with st.expander(f"Monthly Statistics - Year {int(y)}", expanded=(y == uy[-1])):
                                st.dataframe(fmt_table(monthly_sum[monthly_sum['Year'] == y].sort_values('Period', ascending=False)), use_container_width=True)

    # --- TAB 2: CALENDAR ---
    with tab_calendar:
        render_trading_calendar(get_daily_summary(st.session_state.trades.copy()), MARKET_SLUG)

    # --- TAB 3: RAW DATA ---
    with tab_raw_data:
        st.header("Raw Trade Data - Click Table to Edit or Delete") 
        if not st.session_state.trades.empty:
            st.session_state.trades['date'] = pd.to_datetime(st.session_state.trades['date'], errors='coerce')
        
        edited_df = st.data_editor(
            st.session_state.trades, 
            column_config={
                "date": st.column_config.DateColumn("Trade Date", format="YYYY-MM-DD"), 
                "ticker": st.column_config.TextColumn("Ticker"),
                "entry": st.column_config.NumberColumn("Entry Price", format="%.4f"), 
                "exit": st.column_config.NumberColumn("Exit Price", format="%.4f"), 
                "size": st.column_config.NumberColumn("Position Size", format="%d"), 
                "pnl": st.column_config.NumberColumn("Net P&L ($)", format="$%.2f"), 
                "fees": st.column_config.NumberColumn("Fees ($)", format="$%.2f"), 
            },
            num_rows="dynamic", use_container_width=True, hide_index=False
        )
        if not edited_df.equals(st.session_state.trades):
            st.session_state.trades = edited_df.copy()
            save_data(st.session_state.trades)
            st.success("Data updated successfully. Re-running app.")
            st.rerun()