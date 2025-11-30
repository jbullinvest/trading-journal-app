import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_gsheets import GSheetsConnection 
import os 
# Force a redeploy to fix dependency issue
# =============================================================================
# 1. 配置信息 (已插入您的 Google Sheets URL)
# =============================================================================

# 您的 Google Sheets 文档的完整 URL
SHEET_URL = "https://docs.google.com/spreadsheets/d/1ywhInGjEsuzuQjeEKzaF7fNF5f2Qh1YYW_-DmL3TIM/edit?usp=sharing" 

# --- 市场配置 ---
MARKET_OPTIONS = {
    'US Market (USD)': 'us_market',
    'Bursa Malaysia (MYR)': 'bursa_malaysia'
}

# =============================================================================
# 2. Google Sheets 数据持久化函数
# =============================================================================

def get_sheet_name(market_slug):
    """根据市场 slug 返回对应的 Google Sheets 工作表名称"""
    # 确保这些名称与您在 Google Sheets 中的工作表名称完全一致
    if market_slug == 'us_market':
        return 'US_Market_Trades'
    elif market_slug == 'bursa_malaysia':
        return 'Bursa_Malaysia_Trades'
    return 'Default_Trades' 

def get_config_sheet_name():
    """返回配置工作表名称"""
    return 'Config' 

def load_config():
    """从 Google Sheets 加载所有市场的初始资本配置。"""
    try:
        conn = st.connection("gsheets", type=GSheetsConnection) 
        config_df = conn.read(spreadsheet=SHEET_URL, worksheet=get_config_sheet_name())
        
        config_dict = config_df.set_index('Key')['Value'].to_dict()
        return {k: float(v) for k, v in config_dict.items()}
        
    except Exception as e:
        st.error(f"Error loading configuration from Google Sheets: {e}")
        return {
            'initial_capital_us_market': 10000.0,
            'initial_capital_bursa_malaysia': 10000.0,
        }

def load_data(market_slug):
    """从 Google Sheets 加载指定市场的交易数据。"""
    sheet_name = get_sheet_name(market_slug)
    try:
        conn = st.connection("gsheets", type=GSheetsConnection) 
        df = conn.read(spreadsheet=SHEET_URL, worksheet=sheet_name)
        
        # --- 数据清洗和类型转换 ---
        df = df.dropna(how='all')
        df['date'] = pd.to_datetime(df['date'], errors='coerce') 
        for col in ['entry', 'exit', 'size', 'pnl', 'fees']:
             df[col] = pd.to_numeric(df[col], errors='coerce') 
        
        return df.dropna(subset=['date', 'pnl'])
        
    except Exception as e:
        st.warning(f"Warning: Could not load data for {market_slug} from Sheets. Returning empty DataFrame. Error: {e}")
        return pd.DataFrame(columns=['date', 'ticker', 'entry', 'exit', 'size', 'pnl', 'fees'])

def save_data(df, market_slug):
    """将交易数据保存回 Google Sheets（覆盖式保存）。"""
    sheet_name = get_sheet_name(market_slug)
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        conn.write(df, spreadsheet=SHEET_URL, worksheet=sheet_name, worksheet_name=sheet_name)
        st.success(f"✅ Data saved successfully to Google Sheet: {sheet_name}")
    except Exception as e:
        st.error(f"❌ Error saving data to Google Sheets: {e}")

# =============================================================================
# 3. KPI/期望值/凯利准则计算函数 (请替换为您的实际逻辑)
# =============================================================================

def calculate_kpis(df, initial_capital):
    if df.empty:
        return {
            'Total Trades': 0, 'Win Rate (%)': 0.0, 'Avg Gain ($)': 0.0, 'Avg Loss ($)': 0.0,
            'Avg R:R': 0.0, 'Adj R:R (Profit Factor)': 0.0, 'Expectancy ($)': 0.0, 
            'Expectancy (%)': 0.0, 'Current Capital': initial_capital, 'Max Drawdown (MDD)': 0.0,
            'Avg Gain (%)': 0.0, 'Avg Loss (%)': 0.0, 'Net P&L': 0.0, 'Return vs Initial Capital (%)': 0.0
        }
    
    # --- 粘贴您原有的 KPI 计算逻辑 ---
    total_trades = len(df)
    net_pnl = df['pnl'].sum()
    current_capital = initial_capital + net_pnl
    
    # 示例计算 (请替换为您的实际逻辑)
    avg_gain = df[df['pnl'] > 0]['pnl'].mean() if not df[df['pnl'] > 0].empty else 0.0
    avg_loss = df[df['pnl'] <= 0]['pnl'].mean() if not df[df['pnl'] <= 0].empty else 0.0
    win_rate = len(df[df['pnl'] > 0]) / total_trades * 100 if total_trades > 0 else 0.0
    expectancy_dollars = (win_rate / 100 * avg_gain) + ((1 - win_rate / 100) * avg_loss)
    
    return {
        'Total Trades': total_trades, 
        'Win Rate (%)': round(win_rate, 2),
        'Avg Gain ($)': round(avg_gain, 2),
        'Avg Loss ($)': round(avg_loss, 2),
        'Current Capital': round(current_capital, 2),
        'Net P&L': round(net_pnl, 2),
        'Expectancy ($)': round(expectancy_dollars, 2),
        # ... 其他KPIs ...
        'Return vs Initial Capital (%)': round(net_pnl / initial_capital * 100, 2)
    }

def calculate_equity_curve(df, initial_capital):
    if df.empty:
        return pd.DataFrame({'Date': pd.to_datetime([]), 'Capital': [initial_capital]})
    
    df['Date'] = pd.to_datetime(df['date'])
    df = df.sort_values('Date')
    df['Cumulative PnL'] = df['pnl'].cumsum()
    df['Capital'] = initial_capital + df['Cumulative PnL']
    
    return df[['Date', 'Capital']]


# =============================================================================
# 4. Streamlit UI 和主应用逻辑
# =============================================================================

# 初始化 Session State
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.all_config = load_config()
    st.session_state.current_market = list(MARKET_OPTIONS.values())[0] 
    
    market_slug = st.session_state.current_market
    initial_cap = st.session_state.all_config.get(f'initial_capital_{market_slug}', 10000.0)
    
    st.session_state.trades = load_data(market_slug)
    st.session_state.capital = initial_cap


# --- 市场切换逻辑 ---
def switch_market():
    market_slug = st.session_state.selected_market
    st.session_state.current_market = market_slug
    initial_cap = st.session_state.all_config.get(f'initial_capital_{market_slug}', 10000.0)
    
    st.session_state.trades = load_data(market_slug)
    st.session_state.capital = initial_cap

# --- 主界面 ---
st.set_page_config(layout="wide")
st.title("📊 Multi-Market Trading Journal")

# 侧边栏市场选择
market_name_to_slug = {v: k for k, v in MARKET_OPTIONS.items()}
current_market_name = market_name_to_slug.get(st.session_state.current_market, 'US Market (USD)')

st.sidebar.selectbox(
    "选择交易市场",
    options=list(MARKET_OPTIONS.keys()),
    index=list(MARKET_OPTIONS.keys()).index(current_market_name),
    key='selected_market_name',
    on_change=switch_market 
)
st.sidebar.markdown(f"**当前市场:** {st.session_state.current_market.replace('_', ' ').upper()}")


# --- KPI 计算和展示 ---
market_slug = st.session_state.current_market
initial_capital = st.session_state.all_config.get(f'initial_capital_{market_slug}', 10000.0)
kpis = calculate_kpis(st.session_state.trades, initial_capital)

# --- 交易记录/删除/编辑逻辑 (示例) ---
def handle_add_trade(new_trade_data):
    # 将新交易数据转换为 Series 并添加到 DataFrame
    new_trade_series = pd.Series(new_trade_data)
    # 确保列名匹配 (这里假设 new_trade_data 已经有正确的键)
    st.session_state.trades = pd.concat([st.session_state.trades, new_trade_series.to_frame().T], ignore_index=True)
    
    # 保存到 Google Sheets
    save_data(st.session_state.trades, market_slug)


# --- UI 标签页 ---
tab_dashboard, tab_add_trade, tab_raw_data = st.tabs(["📊 Dashboard", "📝 Add Trade", "💾 Raw Data"])

with tab_dashboard:
    st.header(f"{st.session_state.current_market.replace('_', ' ').upper()} 绩效指标")
    
    # --- KPI 展示 ---
    cols = st.columns(6)
    cols[0].metric("总交易数", kpis['Total Trades'])
    
    # 确保 Net P&L 的正负号显示正确
    net_pnl_display = f"${abs(kpis['Net P&L'])}"
    delta_color = "inverse" if kpis['Net P&L'] < 0 else "normal"
    
    cols[1].metric(
        "当前资本", 
        f"${kpis['Current Capital']}", 
        f"{'↑' if kpis['Net P&L'] >= 0 else '↓'}{round(kpis['Return vs Initial Capital (%)'], 2)}% (净盈亏: ${round(kpis['Net P&L'], 2)})", 
        delta_color=delta_color
    )
    cols[2].metric("最大回撤 (MDD)", f"{kpis['Max Drawdown (MDD)']}%")
    cols[3].metric("风险报酬比 (R:R)", kpis['Avg R:R'])
    cols[4].metric("调整后 R:R", kpis['Adj R:R (Profit Factor)'])
    cols[5].metric("回报率 vs 初始资本", f"{kpis['Return vs Initial Capital (%)']}%")

    # --- 权益曲线图 ---
    equity_df = calculate_equity_curve(st.session_state.trades, initial_capital)
    if not equity_df.empty:
        fig = px.line(equity_df, x='Date', y='Capital', title='权益曲线 (净费用)')
        st.plotly_chart(fig, use_container_width=True)

with tab_add_trade:
    st.header("添加新的交易记录")
    
    with st.form("add_trade_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        trade_date = col1.date_input("日期", value="today")
        ticker = col2.text_input("股票代码 (Ticker)", placeholder="AAPL")
        
        entry_price = col1.number_input("入场价 (Entry Price)", min_value=0.0, format="%.4f")
        exit_price = col2.number_input("出场价 (Exit Price)", min_value=0.0, format="%.4f")
        
        size = col1.number_input("股数/合约数 (Size)", min_value=1, step=1)
        pnl = col2.number_input("净盈亏 (P&L $)", format="%.2f")
        fees = st.number_input("佣金/费用 (Fees $)", min_value=0.0, format="%.2f")
        
        submitted = st.form_submit_button("添加交易")
        
        if submitted:
            new_trade = {
                'date': trade_date,
                'ticker': ticker,
                'entry': entry_price,
                'exit': exit_price,
                'size': size,
                'pnl': pnl,
                'fees': fees
            }
            handle_add_trade(new_trade)
            st.success("新交易已记录并保存到 Google Sheets！")


with tab_raw_data:
    st.header("原始数据和管理")
    
    # 数据编辑功能
    st.write("⚠️ 注意：编辑后请点击 '保存原始数据更改' 按钮，否则数据不会保存到 Google Sheets。")
    edited_df = st.data_editor(st.session_state.trades, num_rows="dynamic")
    
    if st.button("💾 保存原始数据更改"):
        st.session_state.trades = edited_df
        save_data(st.session_state.trades, market_slug)

# --- 确保在应用结束时，所有状态都已设置 ---
st.session_state.capital = kpis['Current Capital']
