import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import hashlib
import json
st.set_page_config(page_title="GrainShield Pro - Oilseed Hedging Platform", page_icon="🌾", layout="wide")
if 'blockchain' not in st.session_state:
    st.session_state.blockchain = []
if 'users' not in st.session_state:
    st.session_state.users = {}
if 'contracts' not in st.session_state:
    st.session_state.contracts = []
if 'positions' not in st.session_state:
    st.session_state.positions = []
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'alerts' not in st.session_state:
    st.session_state.alerts = [
        {"time": datetime.now() - timedelta(hours=2), "message": "Price surge detected: +3.5% in soybean", "type": "warning", "read": False},
        {"time": datetime.now() - timedelta(hours=5), "message": "Favorable conditions for selling mustard", "type": "success", "read": False},
        {"time": datetime.now() - timedelta(days=1), "message": "NCDEX volumes increased by 15%", "type": "info", "read": False}
    ]
if 'notifications' not in st.session_state:
    st.session_state.notifications = []
if 'price_data' not in st.session_state:
    dates = pd.date_range(end=datetime.now(), periods=180, freq='D')
    base_price = 6000
    trend = np.linspace(0, 500, 180)
    seasonality = 300 * np.sin(np.linspace(0, 4*np.pi, 180))
    noise = np.random.normal(0, 150, 180)
    prices = base_price + trend + seasonality + noise
    st.session_state.price_data = pd.DataFrame({
        'date': dates,
        'price': prices,
        'volume': np.random.randint(1000, 5000, 180),
        'commodity': np.random.choice(['Soybean', 'Mustard', 'Groundnut', 'Sunflower'], 180)
    })
if 'market_data' not in st.session_state:
    st.session_state.market_data = {
        'Soybean': {'price': 6500, 'change': 2.3, 'volume': 4500, 'volatility': 12.5},
        'Mustard': {'price': 7200, 'change': -1.2, 'volume': 3200, 'volatility': 15.2},
        'Groundnut': {'price': 5800, 'change': 0.8, 'volume': 2800, 'volatility': 10.8},
        'Sunflower': {'price': 6900, 'change': 1.5, 'volume': 3600, 'volatility': 13.4}
    }
if 'educational_modules' not in st.session_state:
    st.session_state.educational_modules = [
        {
            'title': 'Introduction to Hedging',
            'description': 'Learn the basics of hedging and how it protects against price volatility',
            'duration': '15 mins',
            'level': 'Beginner',
            'completed': False
        },
        {
            'title': 'Understanding Futures Contracts',
            'description': 'Deep dive into futures trading mechanisms and strategies',
            'duration': '25 mins',
            'level': 'Intermediate',
            'completed': False
        },
        {
            'title': 'Forward Contract Management',
            'description': 'Learn to create and manage forward contracts effectively',
            'duration': '20 mins',
            'level': 'Intermediate',
            'completed': False
        },
        {
            'title': 'Risk Assessment & Management',
            'description': 'Advanced techniques for assessing and managing market risks',
            'duration': '30 mins',
            'level': 'Advanced',
            'completed': False
        },
        {
            'title': 'NCDEX Trading Basics',
            'description': 'Understanding NCDEX platform and trading mechanisms',
            'duration': '20 mins',
            'level': 'Beginner',
            'completed': False
        }
    ]
def hash_block(block):
    block_string = json.dumps(block, sort_keys=True)
    return hashlib.sha256(block_string.encode()).hexdigest()
def add_to_blockchain(transaction_type, data):
    previous_hash = st.session_state.blockchain[-1]['hash'] if st.session_state.blockchain else "0"
    block = {
        'index': len(st.session_state.blockchain),
        'timestamp': datetime.now().isoformat(),
        'transaction_type': transaction_type,
        'data': data,
        'previous_hash': previous_hash
    }
    block['hash'] = hash_block(block)
    st.session_state.blockchain.append(block)
    return block['hash']
def verify_blockchain():
    for i in range(1, len(st.session_state.blockchain)):
        current = st.session_state.blockchain[i]
        previous = st.session_state.blockchain[i-1]
        if current['previous_hash'] != previous['hash']:
            return False
        if current['hash'] != hash_block({k: v for k, v in current.items() if k != 'hash'}):
            return False
    return True
def predict_prices(days_ahead=30, commodity='Soybean'):
    df = st.session_state.price_data[st.session_state.price_data['commodity'] == commodity].copy()
    if len(df) < 30:
        df = st.session_state.price_data.copy()
    df['day'] = (df['date'] - df['date'].min()).dt.days
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['price_lag_7'] = df['price'].shift(7).fillna(method='bfill')
    df['price_lag_30'] = df['price'].shift(30).fillna(method='bfill')
    df['rolling_mean_7'] = df['price'].rolling(7).mean().fillna(method='bfill')
    df['rolling_std_7'] = df['price'].rolling(7).std().fillna(method='bfill')
    features = ['day', 'day_of_week', 'month', 'price_lag_7', 'price_lag_30', 'rolling_mean_7', 'rolling_std_7']
    X = df[features]
    y = df['price']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='D')
    future_predictions = []
    last_row = df.iloc[-1].copy()
    for i, future_date in enumerate(future_dates):
        future_row = {
            'day': last_row['day'] + i + 1,
            'day_of_week': future_date.dayofweek,
            'month': future_date.month,
            'price_lag_7': last_row['price'] if i < 7 else future_predictions[i-7],
            'price_lag_30': last_row['price'] if i < 30 else df.iloc[-(30-i)]['price'],
            'rolling_mean_7': df['price'].tail(7).mean() if i == 0 else np.mean(future_predictions[-7:]) if i >= 7 else np.mean(list(df['price'].tail(7-i)) + future_predictions[:i]),
            'rolling_std_7': df['price'].tail(7).std() if i == 0 else np.std(future_predictions[-7:]) if i >= 7 else np.std(list(df['price'].tail(7-i)) + future_predictions[:i])
        }
        X_future = scaler.transform([[future_row[f] for f in features]])
        prediction = model.predict(X_future)[0]
        future_predictions.append(prediction)
    confidence_interval = np.array(future_predictions) * 0.05
    return pd.DataFrame({
        'date': future_dates,
        'predicted_price': future_predictions,
        'lower_bound': np.array(future_predictions) - confidence_interval,
        'upper_bound': np.array(future_predictions) + confidence_interval
    })
def calculate_risk_metrics(current_price, contract_price, quantity):
    price_difference = contract_price - current_price
    total_exposure = abs(price_difference * quantity)
    risk_percentage = (price_difference / current_price) * 100
    return {
        'price_diff': price_difference,
        'total_exposure': total_exposure,
        'risk_pct': risk_percentage
    }
def generate_market_alert(commodity, price_change):
    if abs(price_change) > 3:
        alert_type = "warning" if price_change > 0 else "info"
        message = f"Significant price movement in {commodity}: {price_change:+.2f}%"
        st.session_state.alerts.insert(0, {
            "time": datetime.now(),
            "message": message,
            "type": alert_type,
            "read": False
        })
        if st.session_state.logged_in:
            st.session_state.notifications.append(message)
def calculate_portfolio_metrics(username):
    user_positions = [p for p in st.session_state.positions if p['username'] == username and p['status'] == 'Open']
    if not user_positions:
        return {'total_value': 0, 'total_pnl': 0, 'num_positions': 0, 'avg_return': 0}
    total_value = 0
    total_pnl = 0
    for pos in user_positions:
        current_price = st.session_state.market_data.get(pos['commodity'], {}).get('price', pos['entry_price'])
        pnl = (current_price - pos['entry_price']) * pos['quantity'] if pos['type'] == "Long (Buy)" else (pos['entry_price'] - current_price) * pos['quantity']
        total_pnl += pnl
        total_value += pos['entry_price'] * pos['quantity']
    avg_return = (total_pnl / total_value * 100) if total_value > 0 else 0
    return {
        'total_value': total_value,
        'total_pnl': total_pnl,
        'num_positions': len(user_positions),
        'avg_return': avg_return
    }
def export_transaction_history(username):
    user_blocks = [b for b in st.session_state.blockchain if 'username' in str(b.get('data', {})) and username in str(b['data'])]
    df = pd.DataFrame([{
        'Block': b['index'],
        'Timestamp': b['timestamp'],
        'Type': b['transaction_type'],
        'Hash': b['hash'][:16]
    } for b in user_blocks])
    return df
st.title("🌾 GrainShield Pro - Oilseed Hedging Platform")
st.caption("AI-Powered Risk Management & Blockchain-Secured Trading")
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        with tab1:
            st.subheader("Login to Your Account")
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔓 Login", use_container_width=True):
                    if username in st.session_state.users:
                        hashed_pass = hashlib.sha256(password.encode()).hexdigest()
                        if st.session_state.users[username]['password'] == hashed_pass:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.notifications.append(f"Welcome back, {username}!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
                    else:
                        st.error("❌ User not found")
            with col_b:
                if st.button("Demo Login", use_container_width=True):
                    if 'demo_user' not in st.session_state.users:
                        st.session_state.users['demo_user'] = {
                            'password': hashlib.sha256('demo123'.encode()).hexdigest(),
                            'type': 'Farmer',
                            'location': 'Maharashtra',
                            'balance': 100000
                        }
                    st.session_state.logged_in = True
                    st.session_state.username = 'demo_user'
                    st.rerun()
        with tab2:
            st.subheader("Create New Account")
            new_username = st.text_input("Choose Username", key="reg_user")
            new_password = st.text_input("Create Password", type="password", key="reg_pass")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_pass")
            col1, col2 = st.columns(2)
            with col1:
                user_type = st.selectbox("Account Type", ["Farmer", "FPO", "Trader", "Buyer"])
            with col2:
                location = st.text_input("Location/District")
            phone = st.text_input("Mobile Number")
            if st.button("✅ Create Account", use_container_width=True):
                if new_password != confirm_password:
                    st.error("Passwords do not match!")
                elif new_username in st.session_state.users:
                    st.error("Username already exists!")
                elif len(new_username) < 3 or len(new_password) < 6:
                    st.error("Username must be 3+ chars and password 6+ chars")
                else:
                    hashed_pass = hashlib.sha256(new_password.encode()).hexdigest()
                    st.session_state.users[new_username] = {
                        'password': hashed_pass,
                        'type': user_type,
                        'location': location,
                        'phone': phone,
                        'balance': 100000,
                        'created_at': datetime.now().isoformat()
                    }
                    add_to_blockchain('USER_REGISTRATION', {'username': new_username, 'type': user_type, 'location': location})
                    st.success("✅ Registration successful! Please login.")
else:
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=80)
        st.title(f"👤 {st.session_state.username}")
        user_info = st.session_state.users[st.session_state.username]
        st.markdown(f"**Type:** {user_info['type']}")
        st.markdown(f"**Location:** {user_info.get('location', 'N/A')}")
        st.markdown(f"**Balance:** ₹{user_info['balance']:,.2f}")
        portfolio = calculate_portfolio_metrics(st.session_state.username)
        if portfolio['num_positions'] > 0:
            st.markdown(f"**Active Positions:** {portfolio['num_positions']}")
            st.markdown(f"**Portfolio P&L:** ₹{portfolio['total_pnl']:,.2f}")
            st.markdown(f"**Avg Return:** {portfolio['avg_return']:.2f}%")
        st.divider()
        unread_alerts = len([a for a in st.session_state.alerts if not a['read']])
        if unread_alerts > 0:
            st.warning(f"🔔 {unread_alerts} New Alerts")
        if st.session_state.notifications:
            with st.expander("📬 Notifications"):
                for notif in st.session_state.notifications[-5:]:
                    st.info(notif)
                if st.button("Clear Notifications"):
                    st.session_state.notifications = []
                    st.rerun()
        st.divider()
        menu = st.selectbox("📍 Navigation", [
            "🏠 Dashboard",
            "📈 Price Forecast",
            "💹 Virtual Hedging",
            "📋 Forward Contracts",
            "💼 My Portfolio",
            "🔗 Blockchain Ledger",
            "📚 Learning Hub",
            "📊 Analytics",
            "⚙️ Settings"
        ])
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.notifications = []
            st.rerun()
    if menu == "🏠 Dashboard":
        st.header("📊 Market Dashboard")
        if st.session_state.notifications:
            for notif in st.session_state.notifications[-3:]:
                st.success(f"✅ {notif}")
        col1, col2, col3, col4, col5 = st.columns(5)
        current_price = st.session_state.price_data['price'].iloc[-1]
        price_change = current_price - st.session_state.price_data['price'].iloc[-2]
        col1.metric("Current Price", f"₹{current_price:.2f}", f"{price_change:+.2f}")
        col2.metric("Daily Volume", f"{st.session_state.price_data['volume'].iloc[-1]:,} qtl")
        col3.metric("Active Contracts", len([c for c in st.session_state.contracts if c['status'] == 'Open']))
        col4.metric("Your Positions", len([p for p in st.session_state.positions if p['username'] == st.session_state.username and p['status'] == 'Open']))
        portfolio = calculate_portfolio_metrics(st.session_state.username)
        col5.metric("Portfolio P&L", f"₹{portfolio['total_pnl']:,.0f}", f"{portfolio['avg_return']:.1f}%")
        st.subheader("🌾 Live Commodity Prices")
        cols = st.columns(4)
        for idx, (commodity, data) in enumerate(st.session_state.market_data.items()):
            with cols[idx]:
                delta_color = "normal" if data['change'] >= 0 else "inverse"
                st.metric(
                    commodity,
                    f"₹{data['price']:,.0f}",
                    f"{data['change']:+.1f}%",
                    delta_color=delta_color
                )
                st.caption(f"Vol: {data['volume']} | σ: {data['volatility']:.1f}%")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📈 Historical Price Trend")
            commodity_filter = st.selectbox("Select Commodity", ['All'] + list(st.session_state.market_data.keys()), key="dash_commodity")
            if commodity_filter == 'All':
                plot_data = st.session_state.price_data
            else:
                plot_data = st.session_state.price_data[st.session_state.price_data['commodity'] == commodity_filter]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_data['date'],
                y=plot_data['price'],
                mode='lines',
                name='Price',
                line=dict(color='#1f77b4', width=2),
                fill='tonexty'
            ))
            fig.add_trace(go.Scatter(
                x=plot_data['date'],
                y=plot_data['price'].rolling(7).mean(),
                mode='lines',
                name='7-Day MA',
                line=dict(color='orange', dash='dash')
            ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Price (₹/quintal)",
                height=400,
                hovermode='x unified',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("📊 Volume Distribution")
            volume_by_commodity = st.session_state.price_data.groupby('commodity')['volume'].sum().reset_index()
            fig_pie = px.pie(
                volume_by_commodity,
                values='volume',
                names='commodity',
                title='Trading Volume by Commodity'
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        st.subheader("🔔 Recent Market Alerts")
        alert_cols = st.columns(3)
        for idx, alert in enumerate(st.session_state.alerts[:3]):
            with alert_cols[idx % 3]:
                time_diff = datetime.now() - alert['time']
                if time_diff.days > 0:
                    time_str = f"{time_diff.days}d ago"
                elif time_diff.seconds > 3600:
                    time_str = f"{time_diff.seconds // 3600}h ago"
                else:
                    time_str = f"{time_diff.seconds // 60}m ago"
                if alert['type'] == 'warning':
                    st.warning(f"⚠️ **{time_str}**\n\n{alert['message']}")
                elif alert['type'] == 'success':
                    st.success(f"✅ **{time_str}**\n\n{alert['message']}")
                else:
                    st.info(f"ℹ️ **{time_str}**\n\n{alert['message']}")
        if st.button("Mark All as Read"):
            for alert in st.session_state.alerts:
                alert['read'] = True
            st.rerun()
        st.subheader("🎯 Quick Actions")
        qcol1, qcol2, qcol3, qcol4 = st.columns(4)
        with qcol1:
            if st.button("📈 New Hedge Position", use_container_width=True):
                st.session_state.quick_action = "hedge"
                st.rerun()
        with qcol2:
            if st.button("📋 Create Contract", use_container_width=True):
                st.session_state.quick_action = "contract"
                st.rerun()
        with qcol3:
            if st.button("🔮 View Forecast", use_container_width=True):
                st.session_state.quick_action = "forecast"
                st.rerun()
        with qcol4:
            if st.button("📚 Start Learning", use_container_width=True):
                st.session_state.quick_action = "learn"
                st.rerun()
    elif menu == "📈 Price Forecast":
        st.header("🔮 AI-Powered Price Forecasting")
        col1, col2, col3 = st.columns(3)
        with col1:
            forecast_commodity = st.selectbox("Select Commodity", list(st.session_state.market_data.keys()))
        with col2:
            forecast_days = st.slider("Forecast Period (days)", 7, 90, 30)
        with col3:
            confidence_level = st.selectbox("Confidence Level", ["68%", "95%", "99%"])
        if st.button("🚀 Generate Forecast", use_container_width=True):
            with st.spinner("🤖 AI analyzing market patterns..."):
                predictions = predict_prices(forecast_days, forecast_commodity)
                historical = st.session_state.price_data[st.session_state.price_data['commodity'] == forecast_commodity].copy()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=historical['date'],
                    y=historical['price'],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='blue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=predictions['date'],
                    y=predictions['predicted_price'],
                    mode='lines',
                    name='Predicted Price',
                    line=dict(color='red', width=2, dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=predictions['date'].tolist() + predictions['date'].tolist()[::-1],
                    y=predictions['upper_bound'].tolist() + predictions['lower_bound'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval'
                ))
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Price (₹/quintal)",
                    height=500,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                col1, col2, col3, col4 = st.columns(4)
                last_historical = historical['price'].iloc[-1]
                predicted_30d = predictions['predicted_price'].iloc[min(29, len(predictions)-1)]
                expected_change = ((predicted_30d - last_historical) / last_historical) * 100
                col1.metric("Current Price", f"₹{last_historical:.2f}")
                col2.metric("30-Day Forecast", f"₹{predicted_30d:.2f}", f"{expected_change:+.2f}%")
                col3.metric("Volatility", f"{predictions['predicted_price'].std():.2f}")
                col4.metric("Trend", "📈 Bullish" if expected_change > 0 else "📉 Bearish")
                st.subheader("📊 Forecast Statistics")
                forecast_stats = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Min', 'Max', 'Std Dev'],
                    'Value': [
                        f"₹{predictions['predicted_price'].mean():.2f}",
                        f"₹{predictions['predicted_price'].median():.2f}",
                        f"₹{predictions['predicted_price'].min():.2f}",
                        f"₹{predictions['predicted_price'].max():.2f}",
                        f"₹{predictions['predicted_price'].std():.2f}"
                    ]
                })
                st.dataframe(forecast_stats, use_container_width=True, hide_index=True)
                st.subheader("💡 AI Insights & Recommendations")
                if expected_change > 5:
                    st.success("🎯 **Strong Uptrend Expected:** Consider long positions or holding existing stock")
                elif expected_change > 2:
                    st.info("📈 **Moderate Uptrend:** Good time for partial hedging")
                elif expected_change < -5:
                    st.error("⚠️ **Strong Downtrend Expected:** Consider hedging or forward contracts")
                elif expected_change < -2:
                    st.warning("📉 **Moderate Downtrend:** Monitor closely and consider protective positions")
                else:
                    st.info("➡️ **Stable Market:** Range-bound trading expected")
                volatility_index = predictions['predicted_price'].std() / predictions['predicted_price'].mean() * 100
                if volatility_index > 10:
                    st.warning(f"⚡ High volatility detected ({volatility_index:.1f}%). Risk management strongly recommended.")
    elif menu == "💹 Virtual Hedging":
        st.header("💹 Virtual Futures Trading Simulator")
        tab1, tab2, tab3 = st.tabs(["📈 Open Position", "🧮 Calculator", "📚 Strategy Guide"])
        with tab1:
            st.subheader("🎯 Open New Hedging Position")
            col1, col2 = st.columns(2)
            with col1:
                position_type = st.selectbox("Position Type", ["Long (Buy)", "Short (Sell)"])
                commodity = st.selectbox("Commodity", list(st.session_state.market_data.keys()))
                quantity = st.number_input("Quantity (quintals)", min_value=1, max_value=10000, value=10)
            with col2:
                current_market_price = st.session_state.market_data[commodity]['price']
                entry_price = st.number_input("Entry Price (₹/quintal)", value=float(current_market_price), min_value=1000.0)
                expiry_date = st.date_input("Expiry Date", min_value=datetime.now().date() + timedelta(days=1), value=datetime.now().date() + timedelta(days=30))
                leverage = st.selectbox("Leverage", ["1x", "2x", "3x", "5x"])
            col1, col2, col3 = st.columns(3)
            position_value = entry_price * quantity
            margin_required = position_value * 0.1
            potential_gain = entry_price * quantity * 0.1
            col1.metric("Position Value", f"₹{position_value:,.2f}")
            col2.metric("Margin Required", f"₹{margin_required:,.2f}")
            col3.metric("Potential 10% Gain", f"₹{potential_gain:,.2f}")
            stop_loss = st.number_input("Stop Loss Price (₹/quintal)", value=float(entry_price * 0.95), min_value=1000.0)
            take_profit = st.number_input("Take Profit Price (₹/quintal)", value=float(entry_price * 1.1), min_value=1000.0)
            notes = st.text_area("Position Notes", "")
            if st.button("🚀 Open Position", use_container_width=True, type="primary"):
                if st.session_state.users[st.session_state.username]['balance'] >= margin_required:
                    position = {
                        'username': st.session_state.username,
                        'type': position_type,
                        'commodity': commodity,
                        'quantity': quantity,
                        'entry_price': entry_price,
                        'current_price': entry_price,
                        'expiry_date': expiry_date.isoformat(),
                        'leverage': leverage,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'margin_required': margin_required,
                        'notes': notes,
                        'status': 'Open',
                        'timestamp': datetime.now().isoformat()
                    }
                    st.session_state.positions.append(position)
                    st.session_state.users[st.session_state.username]['balance'] -= margin_required
                    tx_hash = add_to_blockchain('FUTURES_POSITION', position)
                    st.success(f"✅ Position opened successfully!")
                    st.info(f"📋 Transaction Hash: {tx_hash[:32]}...")
                    st.session_state.notifications.append(f"New {position_type} position opened for {commodity}")
                    st.balloons()
                    st.rerun()
                else:
                    st.error(f"❌ Insufficient balance! Required: ₹{margin_required:,.2f}, Available: ₹{st.session_state.users[st.session_state.username]['balance']:,.2f}")
        with tab2:
            st.subheader("🧮 Position Calculator & Risk Assessment")
            calc_col1, calc_col2 = st.columns(2)
            with calc_col1:
                calc_commodity = st.selectbox("Commodity", list(st.session_state.market_data.keys()), key="calc_commodity")
                calc_quantity = st.number_input("Quantity (quintals)", min_value=1, value=10, key="calc_qty")
                calc_entry = st.number_input("Entry Price (₹/quintal)", value=float(st.session_state.market_data[calc_commodity]['price']), key="calc_entry")
            with calc_col2:
                calc_target = st.number_input("Target Price (₹/quintal)", value=float(calc_entry * 1.1), key="calc_target")
                calc_stoploss = st.number_input("Stop Loss Price (₹/quintal)", value=float(calc_entry * 0.95), key="calc_sl")
                calc_type = st.selectbox("Position Type", ["Long", "Short"], key="calc_type")
            if calc_type == "Long":
                potential_profit = (calc_target - calc_entry) * calc_quantity
                potential_loss = (calc_entry - calc_stoploss) * calc_quantity
            else:
                potential_profit = (calc_entry - calc_target) * calc_quantity
                potential_loss = (calc_stoploss - calc_entry) * calc_quantity
            risk_reward_ratio = abs(potential_profit / potential_loss) if potential_loss != 0 else 0
            investment = calc_entry * calc_quantity
            roi_profit = (potential_profit / investment) * 100
            roi_loss = (potential_loss / investment) * 100
            st.subheader("📊 Risk-Reward Analysis")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric("Investment", f"₹{investment:,.2f}")
            metric_col2.metric("Potential Profit", f"₹{potential_profit:,.2f}", f"+{roi_profit:.1f}%")
            metric_col3.metric("Potential Loss", f"₹{potential_loss:,.2f}", f"-{roi_loss:.1f}%")
            metric_col4.metric("Risk:Reward", f"1:{risk_reward_ratio:.2f}")
            if risk_reward_ratio >= 2:
                st.success("✅ Excellent Risk-Reward Ratio (>2:1)")
            elif risk_reward_ratio >= 1.5:
                st.info("📊 Good Risk-Reward Ratio (>1.5:1)")
            else:
                st.warning("⚠️ Poor Risk-Reward Ratio (<1.5:1) - Consider adjusting targets")
            fig = go.Figure()
            prices = np.linspace(calc_entry * 0.8, calc_entry * 1.2, 100)
            if calc_type == "Long":
                pnl = (prices - calc_entry) * calc_quantity
            else:
                pnl = (calc_entry - prices) * calc_quantity
            fig.add_trace(go.Scatter(x=prices, y=pnl, mode='lines', name='P&L', line=dict(color='blue', width=2)))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=calc_entry, line_dash="dash", line_color="orange", annotation_text="Entry")
            fig.add_vline(x=calc_target, line_dash="dash", line_color="green", annotation_text="Target")
            fig.add_vline(x=calc_stoploss, line_dash="dash", line_color="red", annotation_text="Stop Loss")
            fig.update_layout(title="P&L Profile", xaxis_title="Price (₹/quintal)", yaxis_title="Profit/Loss (₹)", height=400)
            st.plotly_chart(fig, use_container_width=True)
        with tab3:
            st.subheader("📚 Hedging Strategies Guide")
            strategy_cols = st.columns(2)
            with strategy_cols[0]:
                st.markdown("### 🛡️ Protective Hedging Strategies")
                st.markdown("""
                **1. Simple Short Hedge**
                - Best for: Farmers expecting harvest
                - Action: Sell futures against expected production
                - Benefit: Lock in current prices, protect against decline
                **2. Rolling Hedge**
                - Best for: Continuous production/sales
                - Action: Maintain short positions, roll over monthly
                - Benefit: Ongoing price protection
                **3. Minimum Price Hedge**
                - Best for: Risk-averse farmers
                - Action: Combine futures with options
                - Benefit: Downside protection with upside potential
                """)
            with strategy_cols[1]:
                st.markdown("### 📈 Speculative Strategies")
                st.markdown("""
                **1. Trend Following**
                - Best for: Experienced traders
                - Action: Long in uptrend, short in downtrend
                - Risk: Medium to High
                **2. Spread Trading**
                - Best for: Volatility traders
                - Action: Long/short different commodities or months
                - Risk: Medium
                **3. Swing Trading**
                - Best for: Short-term traders
                - Action: Capture 2-5 day price swings
                - Risk: High
                """)
            st.info("💡 **Pro Tip:** Start with protective hedging strategies before attempting speculative trades. Always use stop losses!")
    elif menu == "📋 Forward Contracts":
        st.header("📋 Forward Contract Management System")
        tab1, tab2, tab3 = st.tabs(["✍️ Create Contract", "🔍 Browse Contracts", "📊 My Contracts"])
        with tab1:
            st.subheader("✍️ Create New Forward Contract")
            fc_col1, fc_col2 = st.columns(2)
            with fc_col1:
                contract_type = st.selectbox("Contract Type", ["Sell Forward", "Buy Forward"])
                commodity = st.selectbox("Commodity", list(st.session_state.market_data.keys()), key="fc_commodity")
                quantity = st.number_input("Quantity (quintals)", min_value=1, max_value=100000, value=100, key="fc_qty")
                quality_grade = st.selectbox("Quality Grade", ["Premium", "Grade A", "Grade B", "Standard"])
            with fc_col2:
                contract_price = st.number_input("Contract Price (₹/quintal)", value=float(st.session_state.market_data[commodity]['price']), key="fc_price")
                delivery_date = st.date_input("Delivery Date", min_value=datetime.now().date() + timedelta(days=30), value=datetime.now().date() + timedelta(days=60), key="fc_date")
                delivery_location = st.text_input("Delivery Location", "")
                payment_terms = st.selectbox("Payment Terms", ["Advance 25%", "Advance 50%", "On Delivery", "30 Days Credit"])
            terms = st.text_area("Contract Terms & Conditions", "1. Quality as per NCDEX standards\n2. Moisture content < 12%\n3. Packaging: New gunny bags\n4. Transportation: Buyer's responsibility\n5. Dispute resolution: Local arbitration")
            advance_amount = st.number_input("Advance Amount (₹)", min_value=0, value=int(contract_price * quantity * 0.1), key="fc_advance")
            if st.button("📝 Create Contract", use_container_width=True, type="primary"):
                contract = {
                    'id': len(st.session_state.contracts) + 1,
                    'creator': st.session_state.username,
                    'type': contract_type,
                    'commodity': commodity,
                    'quantity': quantity,
                    'quality_grade': quality_grade,
                    'price': contract_price,
                    'delivery_date': delivery_date.isoformat(),
                    'delivery_location': delivery_location,
                    'payment_terms': payment_terms,
                    'terms': terms,
                    'advance_amount': advance_amount,
                    'status': 'Open',
                    'counterparty': None,
                    'timestamp': datetime.now().isoformat(),
                    'views': 0
                }
                st.session_state.contracts.append(contract)
                tx_hash = add_to_blockchain('FORWARD_CONTRACT', contract)
                st.success(f"✅ Contract #{contract['id']} created successfully!")
                st.info(f"📋 Blockchain TX: {tx_hash[:32]}...")
                st.session_state.notifications.append(f"New forward contract created: {commodity} - {quantity} qtl")
                st.balloons()
                st.rerun()
        with tab2:
            st.subheader("🔍 Browse Available Contracts")
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                filter_commodity = st.multiselect("Filter by Commodity", ['All'] + list(st.session_state.market_data.keys()), default=['All'])
            with filter_col2:
                filter_type = st.multiselect("Contract Type", ['All', 'Sell Forward', 'Buy Forward'], default=['All'])
            with filter_col3:
                sort_by = st.selectbox("Sort By", ["Latest First", "Price: Low to High", "Price: High to Low", "Quantity: High to Low"])
            open_contracts = [c for c in st.session_state.contracts if c['status'] == 'Open' and c['creator'] != st.session_state.username]
            if 'All' not in filter_commodity:
                open_contracts = [c for c in open_contracts if c['commodity'] in filter_commodity]
            if 'All' not in filter_type:
                open_contracts = [c for c in open_contracts if c['type'] in filter_type]
            if sort_by == "Price: Low to High":
                open_contracts.sort(key=lambda x: x['price'])
            elif sort_by == "Price: High to Low":
                open_contracts.sort(key=lambda x: x['price'], reverse=True)
            elif sort_by == "Quantity: High to Low":
                open_contracts.sort(key=lambda x: x['quantity'], reverse=True)
            else:
                open_contracts.sort(key=lambda x: x['timestamp'], reverse=True)
            if open_contracts:
                st.info(f"📊 Found {len(open_contracts)} available contracts")
                for contract in open_contracts:
                    contract['views'] = contract.get('views', 0) + 1
                    with st.expander(f"🔖 Contract #{contract['id']} - {contract['commodity']} - {contract['type']} - {contract['quantity']} qtl @ ₹{contract['price']:.2f}"):
                        info_col1, info_col2, info_col3 = st.columns(3)
                        with info_col1:
                            st.markdown(f"**👤 Seller:** {contract['creator']}")
                            st.markdown(f"**📦 Quantity:** {contract['quantity']} quintals")
                            st.markdown(f"**⭐ Quality:** {contract['quality_grade']}")
                        with info_col2:
                            st.markdown(f"**💰 Price:** ₹{contract['price']:.2f}/qtl")
                            st.markdown(f"**📅 Delivery:** {contract['delivery_date']}")
                            st.markdown(f"**📍 Location:** {contract.get('delivery_location', 'N/A')}")
                        with info_col3:
                            total_value = contract['price'] * contract['quantity']
                            st.markdown(f"**💵 Total Value:** ₹{total_value:,.2f}")
                            st.markdown(f"**💳 Payment:** {contract.get('payment_terms', 'On Delivery')}")
                            st.markdown(f"**👁️ Views:** {contract.get('views', 0)}")
                        st.markdown("**📋 Terms & Conditions:**")
                        st.text(contract['terms'])
                        action_col1, action_col2, action_col3 = st.columns([2,1,1])
                        with action_col1:
                            if st.button(f"✅ Accept Contract #{contract['id']}", key=f"accept_{contract['id']}", use_container_width=True):
                                contract['status'] = 'Accepted'
                                contract['counterparty'] = st.session_state.username
                                contract['acceptance_date'] = datetime.now().isoformat()
                                tx_hash = add_to_blockchain('CONTRACT_ACCEPTED', {
                                    'contract_id': contract['id'],
                                    'counterparty': st.session_state.username,
                                    'acceptance_date': datetime.now().isoformat()
                                })
                                st.success(f"✅ Contract accepted! TX: {tx_hash[:16]}...")
                                st.session_state.notifications.append(f"Contract #{contract['id']} accepted successfully")
                                st.rerun()
                        with action_col2:
                            if st.button(f"💬 Contact", key=f"contact_{contract['id']}", use_container_width=True):
                                st.info(f"Contact: {contract['creator']} via platform messaging")
                        with action_col3:
                            if st.button(f"📊 Analyze", key=f"analyze_{contract['id']}", use_container_width=True):
                                current_market = st.session_state.market_data[contract['commodity']]['price']
                                price_diff = contract['price'] - current_market
                                st.metric("vs Market Price", f"₹{price_diff:+,.2f}", f"{(price_diff/current_market*100):+.2f}%")
            else:
                st.warning("⚠️ No contracts match your filters. Try adjusting the filters or create a new contract.")
        with tab3:
            st.subheader("📊 My Contracts Dashboard")
            my_contracts = [c for c in st.session_state.contracts if c['creator'] == st.session_state.username or c.get('counterparty') == st.session_state.username]
            if my_contracts:
                status_counts = {'Open': 0, 'Accepted': 0, 'Completed': 0, 'Cancelled': 0}
                total_value = 0
                for c in my_contracts:
                    status_counts[c['status']] = status_counts.get(c['status'], 0) + 1
                    if c['status'] in ['Open', 'Accepted']:
                        total_value += c['price'] * c['quantity']
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                metric_col1.metric("Total Contracts", len(my_contracts))
                metric_col2.metric("Active Value", f"₹{total_value:,.0f}")
                metric_col3.metric("Open", status_counts['Open'])
                metric_col4.metric("Accepted", status_counts['Accepted'])
                for contract in my_contracts:
                    is_creator = contract['creator'] == st.session_state.username
                    status_emoji = {"Open": "🟡", "Accepted": "🟢", "Completed": "✅", "Cancelled": "❌"}
                    with st.expander(f"{status_emoji.get(contract['status'], '⚪')} Contract #{contract['id']} - {contract['commodity']} - {contract['status']}"):
                        role_col1, role_col2 = st.columns(2)
                        with role_col1:
                            st.markdown(f"**Your Role:** {'Seller' if is_creator and contract['type'] == 'Sell Forward' else 'Buyer'}")
                            st.markdown(f"**Creator:** {contract['creator']}")
                            st.markdown(f"**Counterparty:** {contract.get('counterparty', 'Pending')}")
                        with role_col2:
                            st.markdown(f"**Commodity:** {contract['commodity']}")
                            st.markdown(f"**Quantity:** {contract['quantity']} qtl")
                            st.markdown(f"**Price:** ₹{contract['price']:.2f}/qtl")
                        st.markdown(f"**Delivery Date:** {contract['delivery_date']}")
                        st.markdown(f"**Status:** {contract['status']}")
                        st.markdown(f"**Total Value:** ₹{contract['price'] * contract['quantity']:,.2f}")
                        if contract['status'] == 'Open' and is_creator:
                            action_col1, action_col2 = st.columns(2)
                            with action_col1:
                                if st.button(f"📝 Edit Contract", key=f"edit_{contract['id']}"):
                                    st.info("Edit functionality - Update contract terms")
                            with action_col2:
                                if st.button(f"❌ Cancel Contract", key=f"cancel_{contract['id']}"):
                                    contract['status'] = 'Cancelled'
                                    add_to_blockchain('CONTRACT_CANCELLED', {'contract_id': contract['id']})
                                    st.rerun()
                        elif contract['status'] == 'Accepted':
                            if st.button(f"✅ Mark as Completed", key=f"complete_{contract['id']}"):
                                contract['status'] = 'Completed'
                                contract['completion_date'] = datetime.now().isoformat()
                                add_to_blockchain('CONTRACT_COMPLETED', {'contract_id': contract['id']})
                                st.success("Contract marked as completed!")
                                st.rerun()
            else:
                st.info("📭 No contracts yet. Create your first contract in the 'Create Contract' tab.")
    elif menu == "💼 My Portfolio":
        st.header("💼 Portfolio Management Dashboard")
        portfolio = calculate_portfolio_metrics(st.session_state.username)
        overview_col1, overview_col2, overview_col3, overview_col4, overview_col5 = st.columns(5)
        overview_col1.metric("Total Portfolio Value", f"₹{portfolio['total_value']:,.0f}")
        overview_col2.metric("Total P&L", f"₹{portfolio['total_pnl']:,.0f}", f"{portfolio['avg_return']:.2f}%")
        overview_col3.metric("Active Positions", portfolio['num_positions'])
        overview_col4.metric("Account Balance", f"₹{st.session_state.users[st.session_state.username]['balance']:,.0f}")
        overview_col5.metric("Total Equity", f"₹{st.session_state.users[st.session_state.username]['balance'] + portfolio['total_pnl']:,.0f}")
        tab1, tab2, tab3 = st.tabs(["📈 Open Positions", "📊 Performance Analytics", "📜 Transaction History"])
        with tab1:
            st.subheader("📈 Active Positions")
            user_positions = [p for p in st.session_state.positions if p['username'] == st.session_state.username and p['status'] == 'Open']
            if user_positions:
                for idx, pos in enumerate(user_positions):
                    current_price = st.session_state.market_data.get(pos['commodity'], {}).get('price', pos['entry_price'])
                    pnl = (current_price - pos['entry_price']) * pos['quantity'] if pos['type'] == "Long (Buy)" else (pos['entry_price'] - current_price) * pos['quantity']
                    pnl_pct = (pnl / (pos['entry_price'] * pos['quantity'])) * 100
                    pnl_color = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                    with st.expander(f"{pnl_color} Position #{idx+1} - {pos['commodity']} {pos['type']} - P&L: ₹{pnl:,.0f} ({pnl_pct:+.2f}%)"):
                        pos_col1, pos_col2, pos_col3, pos_col4 = st.columns(4)
                        pos_col1.metric("Entry Price", f"₹{pos['entry_price']:.2f}")
                        pos_col2.metric("Current Price", f"₹{current_price:.2f}", f"{((current_price-pos['entry_price'])/pos['entry_price']*100):+.2f}%")
                        pos_col3.metric("Quantity", f"{pos['quantity']} qtl")
                        pos_col4.metric("P&L", f"₹{pnl:,.2f}", f"{pnl_pct:+.2f}%")
                        detail_col1, detail_col2, detail_col3 = st.columns(3)
                        with detail_col1:
                            st.markdown(f"**Expiry:** {pos['expiry_date']}")
                            st.markdown(f"**Leverage:** {pos.get('leverage', '1x')}")
                        with detail_col2:
                            st.markdown(f"**Stop Loss:** ₹{pos.get('stop_loss', 'N/A')}")
                            st.markdown(f"**Take Profit:** ₹{pos.get('take_profit', 'N/A')}")
                        with detail_col3:
                            st.markdown(f"**Margin:** ₹{pos.get('margin_required', 0):,.2f}")
                            st.markdown(f"**Opened:** {pos['timestamp'][:10]}")
                        if pos.get('notes'):
                            st.markdown(f"**Notes:** {pos['notes']}")
                        action_col1, action_col2, action_col3 = st.columns(3)
                        with action_col1:
                            if st.button(f"❌ Close Position", key=f"close_pos_{idx}", use_container_width=True):
                                pos['status'] = 'Closed'
                                pos['exit_price'] = current_price
                                pos['close_date'] = datetime.now().isoformat()
                                st.session_state.users[st.session_state.username]['balance'] += pnl + pos.get('margin_required', 0)
                                add_to_blockchain('POSITION_CLOSED', {'position': pos, 'pnl': pnl})
                                st.success(f"Position closed! P&L: ₹{pnl:,.2f}")
                                st.session_state.notifications.append(f"Position closed with P&L: ₹{pnl:,.0f}")
                                st.rerun()
                        with action_col2:
                            if st.button(f"📝 Modify", key=f"modify_{idx}", use_container_width=True):
                                st.info("Modify stop loss and take profit levels")
                        with action_col3:
                            if st.button(f"📊 Details", key=f"details_{idx}", use_container_width=True):
                                st.json(pos)
            else:
                st.info("📭 No open positions. Open a new position in the Virtual Hedging section.")
        with tab2:
            st.subheader("📊 Performance Analytics")
            closed_positions = [p for p in st.session_state.positions if p['username'] == st.session_state.username and p['status'] == 'Closed']
            if closed_positions:
                total_trades = len(closed_positions)
                winning_trades = len([p for p in closed_positions if p.get('exit_price', 0) > p['entry_price']])
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                total_pnl_closed = sum([
                    (p.get('exit_price', p['entry_price']) - p['entry_price']) * p['quantity'] if p['type'] == "Long (Buy)"
                    else (p['entry_price'] - p.get('exit_price', p['entry_price'])) * p['quantity']
                    for p in closed_positions
                ])
                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                perf_col1.metric("Total Trades", total_trades)
                perf_col2.metric("Win Rate", f"{win_rate:.1f}%")
                perf_col3.metric("Winning Trades", winning_trades)
                perf_col4.metric("Total Realized P&L", f"₹{total_pnl_closed:,.0f}")
                pnl_by_date = {}
                for p in closed_positions:
                    date = p.get('close_date', p['timestamp'])[:10]
                    pnl = (p.get('exit_price', p['entry_price']) - p['entry_price']) * p['quantity'] if p['type'] == "Long (Buy)" else (p['entry_price'] - p.get('exit_price', p['entry_price'])) * p['quantity']
                    pnl_by_date[date] = pnl_by_date.get(date, 0) + pnl
                if pnl_by_date:
                    pnl_df = pd.DataFrame(list(pnl_by_date.items()), columns=['Date', 'P&L'])
                    pnl_df['Cumulative P&L'] = pnl_df['P&L'].cumsum()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=pnl_df['Date'], y=pnl_df['Cumulative P&L'], mode='lines+markers', name='Cumulative P&L', line=dict(color='green', width=2)))
                    fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Cumulative P&L (₹)", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                commodity_pnl = {}
                for p in closed_positions:
                    pnl = (p.get('exit_price', p['entry_price']) - p['entry_price']) * p['quantity'] if p['type'] == "Long (Buy)" else (p['entry_price'] - p.get('exit_price', p['entry_price'])) * p['quantity']
                    commodity_pnl[p['commodity']] = commodity_pnl.get(p['commodity'], 0) + pnl
                if commodity_pnl:
                    fig_bar = px.bar(x=list(commodity_pnl.keys()), y=list(commodity_pnl.values()), labels={'x': 'Commodity', 'y': 'Total P&L (₹)'}, title='P&L by Commodity')
                    st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("📊 No closed positions yet. Performance analytics will appear after closing positions.")
        with tab3:
            st.subheader("📜 Transaction History")
            user_transactions = export_transaction_history(st.session_state.username)
            if not user_transactions.empty:
                st.dataframe(user_transactions, use_container_width=True, hide_index=True)
                csv = user_transactions.to_csv(index=False)
                st.download_button(
                    label="📥 Download Transaction History",
                    data=csv,
                    file_name=f"transactions_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("📭 No transaction history available.")
    elif menu == "🔗 Blockchain Ledger":
        st.header("🔗 Blockchain Transaction Ledger")
        st.write("All transactions are immutably recorded on the blockchain for complete transparency and security.")
        ledger_col1, ledger_col2, ledger_col3, ledger_col4 = st.columns(4)
        ledger_col1.metric("Total Blocks", len(st.session_state.blockchain))
        ledger_col2.metric("Total Users", len(st.session_state.users))
        ledger_col3.metric("Total Contracts", len(st.session_state.contracts))
        ledger_col4.metric("Blockchain Valid", "✅ Yes" if verify_blockchain() else "❌ No")
        st.subheader("🔍 Recent Transactions")
        filter_tx_type = st.multiselect("Filter by Type", ['All'] + list(set([b['transaction_type'] for b in st.session_state.blockchain])), default=['All'])
        show_blocks = st.slider("Number of blocks to display", 5, 50, 10)
        filtered_blocks = st.session_state.blockchain
        if 'All' not in filter_tx_type:
            filtered_blocks = [b for b in filtered_blocks if b['transaction_type'] in filter_tx_type]
        recent_blocks = list(reversed(filtered_blocks[-show_blocks:]))
        for block in recent_blocks:
            tx_type_emoji = {
                'USER_REGISTRATION': '👤',
                'FUTURES_POSITION': '💹',
                'FORWARD_CONTRACT': '📋',
                'CONTRACT_ACCEPTED': '✅',
                'CONTRACT_COMPLETED': '🎯',
                'CONTRACT_CANCELLED': '❌',
                'POSITION_CLOSED': '🔒'
            }
            emoji = tx_type_emoji.get(block['transaction_type'], '🔷')
            with st.expander(f"{emoji} Block #{block['index']} - {block['transaction_type']} - {block['timestamp'][:19]}"):
                block_col1, block_col2 = st.columns(2)
                with block_col1:
                    st.markdown(f"**Block Index:** {block['index']}")
                    st.markdown(f"**Transaction Type:** {block['transaction_type']}")
                    st.markdown(f"**Timestamp:** {block['timestamp']}")
                with block_col2:
                    st.markdown(f"**Block Hash:** `{block['hash'][:32]}...`")
                    st.markdown(f"**Previous Hash:** `{block['previous_hash'][:32]}...`")
                st.markdown("**Transaction Data:**")
                st.json(block['data'])
                if st.button(f"🔍 Verify Block #{block['index']}", key=f"verify_{block['index']}"):
                    block_copy = {k: v for k, v in block.items() if k != 'hash'}
                    computed_hash = hash_block(block_copy)
                    if computed_hash == block['hash']:
                        st.success("✅ Block verification successful - Hash matches!")
                    else:
                        st.error("❌ Block verification failed - Hash mismatch!")
        st.divider()
        st.subheader("🔐 Blockchain Verification")
        verify_col1, verify_col2 = st.columns(2)
        with verify_col1:
            if st.button("🔍 Verify Entire Blockchain", use_container_width=True):
                with st.spinner("Verifying blockchain integrity..."):
                    is_valid = verify_blockchain()
                    if is_valid:
                        st.success("✅ Blockchain integrity verified! All blocks are valid.")
                    else:
                        st.error("❌ Blockchain integrity compromised! Tampering detected.")
        with verify_col2:
            if st.button("📊 Blockchain Statistics", use_container_width=True):
                st.info(f"Total Blocks: {len(st.session_state.blockchain)}")
                st.info(f"Chain Length: {len(st.session_state.blockchain)}")
                if st.session_state.blockchain:
                    st.info(f"Latest Block Hash: {st.session_state.blockchain[-1]['hash'][:32]}...")
    elif menu == "📚 Learning Hub":
        st.header("📚 Educational Learning Hub")
        st.write("Master hedging strategies, risk management, and commodity trading through interactive content.")
        tab1, tab2, tab3, tab4 = st.tabs(["📖 Course Modules", "📺 Video Library", "📝 Quizzes", "📄 Articles"])
        with tab1:
            st.subheader("📖 Interactive Learning Modules")
            progress_modules = [m for m in st.session_state.educational_modules if m.get('completed', False)]
            completion_rate = (len(progress_modules) / len(st.session_state.educational_modules) * 100) if st.session_state.educational_modules else 0
            prog_col1, prog_col2, prog_col3 = st.columns(3)
            prog_col1.metric("Total Modules", len(st.session_state.educational_modules))
            prog_col2.metric("Completed", len(progress_modules))
            prog_col3.metric("Progress", f"{completion_rate:.0f}%")
            st.progress(completion_rate / 100)
            level_filter = st.multiselect("Filter by Level", ['All', 'Beginner', 'Intermediate', 'Advanced'], default=['All'])
            for idx, module in enumerate(st.session_state.educational_modules):
                if 'All' not in level_filter and module['level'] not in level_filter:
                    continue
                status_icon = "✅" if module.get('completed', False) else "📘"
                level_colors = {'Beginner': '🟢', 'Intermediate': '🟡', 'Advanced': '🔴'}
                level_icon = level_colors.get(module['level'], '⚪')
                with st.expander(f"{status_icon} {module['title']} {level_icon} {module['level']} - {module['duration']}"):
                    st.markdown(f"**{module['description']}**")
                    content = module.get('content', {})
                    if content:
                        st.markdown(f"### 📋 Overview")
                        st.info(content.get('overview', 'Content coming soon'))
                        if 'key_concepts' in content:
                            st.markdown("### 🔑 Key Concepts")
                            for concept in content['key_concepts']:
                                st.markdown(f"• **{concept.split(':')[0]}:** {':'.join(concept.split(':')[1:])}")
                        if 'examples' in content:
                            st.markdown("### 💡 Real-World Examples")
                            for i, example in enumerate(content['examples'], 1):
                                st.success(f"**Example {i}:** {example}")
                        if 'benefits' in content:
                            st.markdown("### ✨ Key Benefits")
                            for benefit in content['benefits']:
                                st.markdown(f"✓ {benefit}")
                        if 'trading_mechanics' in content:
                            st.markdown("### ⚙️ How It Works")
                            for mechanic in content['trading_mechanics']:
                                st.markdown(f"• {mechanic}")
                        if 'strategies' in content:
                            st.markdown("### 📊 Strategies")
                            for strategy in content['strategies']:
                                st.markdown(f"• {strategy}")
                        if 'contract_elements' in content:
                            st.markdown("### 📄 Contract Elements")
                            for element in content['contract_elements']:
                                st.markdown(f"• {element}")
                        if 'best_practices' in content:
                            st.markdown("### ⭐ Best Practices")
                            for practice in content['best_practices']:
                                st.markdown(f"✓ {practice}")
                        if 'risk_types' in content:
                            st.markdown("### ⚠️ Types of Risk")
                            for risk in content['risk_types']:
                                st.markdown(f"• {risk}")
                        if 'measurement_tools' in content:
                            st.markdown("### 📏 Measurement Tools")
                            for tool in content['measurement_tools']:
                                st.markdown(f"• {tool}")
                        if 'mitigation_strategies' in content:
                            st.markdown("### 🛡️ Risk Mitigation")
                            for strategy in content['mitigation_strategies']:
                                st.markdown(f"• {strategy}")
                        if 'advanced_concepts' in content:
                            st.markdown("### 🎓 Advanced Concepts")
                            for concept in content['advanced_concepts']:
                                st.markdown(f"• {concept}")
                        if 'contract_specifications' in content:
                            st.markdown("### 📋 Contract Specifications")
                            for spec in content['contract_specifications']:
                                st.markdown(f"• {spec}")
                        if 'trading_process' in content:
                            st.markdown("### 🔄 Trading Process")
                            for step in content['trading_process']:
                                st.markdown(f"{step}")
                        if 'delivery_process' in content:
                            st.markdown("### 🚚 Delivery Process")
                            for step in content['delivery_process']:
                                st.markdown(f"• {step}")
                    if not module.get('completed', False):
                        if st.button(f"✅ Mark as Complete", key=f"complete_{idx}", use_container_width=True, type="primary"):
                            module['completed'] = True
                            st.session_state.notifications.append(f"Completed: {module['title']}")
                            st.success(f"🎉 Module completed!")
                            st.balloons()
                            st.rerun()
                    else:
                        st.success("✅ Module Completed")
                        if st.button(f"🔄 Review Again", key=f"review_{idx}", use_container_width=True):
                            st.info("Reviewing module content above")
        with tab2:
            st.subheader("📺 Video Tutorial Library")
            st.markdown("**Educational videos on commodity trading and hedging strategies**")
            videos = [
                {"title": "Introduction to Commodity Hedging", "duration": "15:30", "level": "Beginner", "link": "Learn how hedging protects farmers from price volatility", "topics": ["Price risk", "Hedge basics", "Farmer benefits"]},
                {"title": "NCDEX Platform Complete Guide", "duration": "22:45", "level": "Beginner", "link": "Step-by-step guide to trading on NCDEX exchange", "topics": ["Account opening", "Order placement", "Margin system"]},
                {"title": "Futures vs Forward Contracts", "duration": "18:20", "level": "Intermediate", "link": "Understanding the differences and when to use each", "topics": ["Standardization", "Risk comparison", "Liquidity"]},
                {"title": "Technical Analysis for Commodities", "duration": "28:40", "level": "Intermediate", "link": "Chart patterns and indicators for commodity trading", "topics": ["Moving averages", "RSI", "Support/Resistance"]},
                {"title": "Advanced Hedging Strategies", "duration": "35:15", "level": "Advanced", "link": "Complex strategies for portfolio hedging", "topics": ["Delta hedging", "Spread strategies", "Ratio hedging"]},
                {"title": "Risk Management Masterclass", "duration": "42:10", "level": "Advanced", "link": "Comprehensive risk assessment and mitigation", "topics": ["VaR calculation", "Portfolio optimization", "Stress testing"]}
            ]
            video_level = st.selectbox("Filter by Level", ['All', 'Beginner', 'Intermediate', 'Advanced'], key="video_filter")
            filtered_videos = [v for v in videos if video_level == 'All' or v['level'] == video_level]
            cols = st.columns(2)
            for idx, video in enumerate(filtered_videos):
                with cols[idx % 2]:
                    st.markdown(f"### 🎬 {video['title']}")
                    st.caption(f"⏱️ {video['duration']} | 🎯 {video['level']}")
                    st.write(video['link'])
                    with st.expander("📋 Topics Covered"):
                        for topic in video['topics']:
                            st.markdown(f"• {topic}")
                    if st.button(f"▶️ Watch Now", key=f"video_{idx}", use_container_width=True):
                        st.success(f"🎥 Opening: {video['title']}")
                        st.info("💡 **Tip:** Take notes and practice with the Virtual Hedging simulator after watching!")
        with tab3:
            st.subheader("📝 Knowledge Assessment Quizzes")
            st.write("Test your understanding with interactive quizzes")
            quiz_topics = {
                "Hedging Basics": {
                    "level": "Beginner",
                    "questions": [
                        {
                            "q": "What is the primary purpose of hedging in commodity trading?",
                            "options": ["To maximize profits", "To reduce price risk", "To speculate on prices", "To increase trading volume"],
                            "correct": 1,
                            "explanation": "Hedging is primarily used to reduce or eliminate price risk, not to maximize profits. It provides price certainty."
                        },
                        {
                            "q": "Which market position would a farmer take to hedge against falling prices?",
                            "options": ["Long position (Buy)", "Short position (Sell)", "No position", "Both long and short"],
                            "correct": 1,
                            "explanation": "A farmer expecting to sell their harvest would take a short position (sell futures) to lock in current prices."
                        },
                        {
                            "q": "What is basis risk?",
                            "options": ["Risk of default", "Risk of delivery failure", "Difference between spot and futures prices", "Risk of margin call"],
                            "correct": 2,
                            "explanation": "Basis risk refers to the risk that the spot price and futures price don't move perfectly together."
                        }
                    ]
                },
                "Futures Trading": {
                    "level": "Intermediate",
                    "questions": [
                        {
                            "q": "What is the typical margin requirement for commodity futures in India?",
                            "options": ["1-2%", "5-10%", "20-30%", "50%"],
                            "correct": 1,
                            "explanation": "Initial margin for commodity futures is typically 5-10% of the contract value, varying by commodity and volatility."
                        },
                        {
                            "q": "What happens during mark-to-market settlement?",
                            "options": ["Contracts are cancelled", "Profits/losses are settled daily", "Delivery is made", "Margins are refunded"],
                            "correct": 1,
                            "explanation": "Mark-to-market means daily settlement where profits are credited and losses are debited to traders' accounts."
                        },
                        {
                            "q": "When should you square off a futures position?",
                            "options": ["Only at expiry", "Only on profitable days", "Before expiry to avoid delivery", "Never"],
                            "correct": 2,
                            "explanation": "Most traders square off positions before expiry by taking an opposite position to avoid physical delivery."
                        }
                    ]
                },
                "Risk Management": {
                    "level": "Advanced",
                    "questions": [
                        {
                            "q": "What does a hedge ratio of 0.8 mean?",
                            "options": ["80% profit guaranteed", "Hedge 80% of exposure", "20% margin required", "80% chance of success"],
                            "correct": 1,
                            "explanation": "A hedge ratio of 0.8 means you hedge 80% of your exposure, leaving 20% unhedged for potential upside."
                        },
                        {
                            "q": "What is Value at Risk (VaR)?",
                            "options": ["Maximum possible loss", "Maximum expected loss at confidence level", "Average daily loss", "Margin requirement"],
                            "correct": 1,
                            "explanation": "VaR estimates the maximum expected loss over a period at a given confidence level (e.g., 95%)."
                        },
                        {
                            "q": "Which strategy combines long and short positions in different contract months?",
                            "options": ["Arbitrage", "Spread trading", "Scalping", "Day trading"],
                            "correct": 1,
                            "explanation": "Spread trading involves simultaneous long and short positions in different months to profit from price differentials."
                        }
                    ]
                }
            }
            selected_quiz = st.selectbox("Choose Quiz Topic", list(quiz_topics.keys()))
            quiz_data = quiz_topics[selected_quiz]
            st.info(f"📊 Level: {quiz_data['level']} | Questions: {len(quiz_data['questions'])}")
            if f'quiz_{selected_quiz}' not in st.session_state.module_progress:
                st.session_state.module_progress[f'quiz_{selected_quiz}'] = {'started': False, 'answers': {}, 'submitted': False}
            quiz_state = st.session_state.module_progress[f'quiz_{selected_quiz}']
            if not quiz_state['started']:
                if st.button("🚀 Start Quiz", use_container_width=True, type="primary"):
                    quiz_state['started'] = True
                    quiz_state['answers'] = {}
                    quiz_state['submitted'] = False
                    st.rerun()
            else:
                if not quiz_state['submitted']:
                    st.markdown("### Answer the following questions:")
                    for i, question in enumerate(quiz_data['questions']):
                        st.markdown(f"**Q{i+1}. {question['q']}**")
                        answer = st.radio(
                            "Select your answer:",
                            question['options'],
                            key=f"q_{selected_quiz}_{i}",
                            index=quiz_state['answers'].get(i, 0)
                        )
                        quiz_state['answers'][i] = question['options'].index(answer)
                        st.divider()
                    if st.button("✅ Submit Quiz", use_container_width=True, type="primary"):
                        quiz_state['submitted'] = True
                        st.rerun()
                else:
                    correct_count = 0
                    for i, question in enumerate(quiz_data['questions']):
                        user_answer = quiz_state['answers'].get(i, -1)
                        is_correct = user_answer == question['correct']
                        if is_correct:
                            correct_count += 1
                        st.markdown(f"**Q{i+1}. {question['q']}**")
                        if is_correct:
                            st.success(f"✅ Correct! Your answer: {question['options'][user_answer]}")
                        else:
                            st.error(f"❌ Wrong. Your answer: {question['options'][user_answer]}")
                            st.info(f"Correct answer: {question['options'][question['correct']]}")
                        st.info(f"💡 {question['explanation']}")
                        st.divider()
                    score_pct = (correct_count / len(quiz_data['questions'])) * 100
                    st.markdown(f"### 📊 Your Score: {correct_count}/{len(quiz_data['questions'])} ({score_pct:.0f}%)")
                    if score_pct >= 80:
                        st.success("🎉 Excellent! You've mastered this topic!")
                        st.balloons()
                    elif score_pct >= 60:
                        st.info("👍 Good job! Review the concepts you missed.")
                    else:
                        st.warning("📚 Keep learning! Review the module content and try again.")
                    if st.button("🔄 Retake Quiz", use_container_width=True):
                        quiz_state['started'] = False
                        quiz_state['answers'] = {}
                        quiz_state['submitted'] = False
                        st.rerun()
        with tab4:
            st.subheader("📄 Expert Articles & Guides")
            articles = [
                {
                    "title": "Complete Guide to Oilseed Hedging in India",
                    "category": "Comprehensive Guide",
                    "level": "All Levels",
                    "reading_time": "12 min",
                    "url": "https://www.commoditiescontrol.com/eagritrader/commodityknowledge/futurstrading/hedging.htm",
                    "summary": "Everything you need to know about hedging oilseeds including soybean, mustard, and groundnut.",
                    "key_points": [
                        "Understanding Indian oilseed markets and seasonal patterns",
                        "Step-by-step hedging strategies for different crops",
                        "NCDEX contract specifications and trading hours",
                        "Real case studies from successful farmers"
                    ]
                },
                {
                    "title": "Understanding Futures Trading on NCDEX",
                    "category": "Trading Basics",
                    "level": "Beginner",
                    "reading_time": "10 min",
                    "url": "https://www.ncdex.com/market-pulse/knowledge-centre",
                    "summary": "Official NCDEX guide to futures trading covering contracts, margins, and settlement.",
                    "key_points": [
                        "Contract specifications for agricultural commodities",
                        "Margin requirements and mark-to-market process",
                        "Trading hours and settlement procedures",
                        "Delivery and quality standards"
                    ]
                },
                {
                    "title": "Managing Basis Risk in Agricultural Commodities",
                    "category": "Risk Management",
                    "level": "Intermediate",
                    "reading_time": "8 min",
                    "url": "https://www.investopedia.com/terms/b/basisrisk.asp",
                    "summary": "Learn how to minimize basis risk when hedging agricultural commodities.",
                    "key_points": [
                        "What causes basis to widen or narrow",
                        "Historical basis patterns in markets",
                        "Strategies to reduce basis risk exposure",
                        "Choosing the right delivery month"
                    ]
                },
                {
                    "title": "Forward Contracts vs Futures: Making the Right Choice",
                    "category": "Strategy",
                    "level": "Beginner",
                    "reading_time": "6 min",
                    "url": "https://www.investopedia.com/ask/answers/difference-between-forward-and-futures-contracts/",
                    "summary": "Compare forward and futures contracts to determine which hedging tool is right for your needs.",
                    "key_points": [
                        "Key differences in flexibility and standardization",
                        "Credit risk vs market risk considerations",
                        "Liquidity and transaction costs comparison",
                        "When to use each instrument effectively"
                    ]
                },
                {
                    "title": "Commodity Trading Strategies for Beginners",
                    "category": "Trading Basics",
                    "level": "Beginner",
                    "reading_time": "15 min",
                    "url": "https://economictimes.indiatimes.com/markets/commodities",
                    "summary": "Essential trading strategies and risk management tips for new commodity traders.",
                    "key_points": [
                        "Understanding market fundamentals",
                        "Position sizing and risk management",
                        "Technical analysis basics",
                        "Common mistakes to avoid"
                    ]
                },
                {
                    "title": "Seasonal Patterns in Oilseed Prices",
                    "category": "Market Analysis",
                    "level": "Intermediate",
                    "reading_time": "10 min",
                    "url": "https://agmarknet.gov.in/",
                    "summary": "Understand seasonal price movements to time your hedging decisions better.",
                    "key_points": [
                        "Harvest season impact on prices",
                        "Festival demand spikes",
                        "Import policy effects",
                        "Weather patterns and volatility"
                    ]
                },
                {
                    "title": "Technical Analysis in Commodity Markets",
                    "category": "Market Analysis",
                    "level": "Intermediate",
                    "reading_time": "12 min",
                    "url": "https://www.investopedia.com/articles/trading/09/commodity-technical-analysis.asp",
                    "summary": "Learn to use charts and indicators for better trading decisions.",
                    "key_points": [
                        "Key chart patterns",
                        "Moving averages and trends",
                        "Support and resistance",
                        "Volume analysis"
                    ]
                },
                {
                    "title": "Risk Management in Agricultural Futures",
                    "category": "Risk Management",
                    "level": "Advanced",
                    "reading_time": "14 min",
                    "url": "https://www.cmegroup.com/education/courses/risk-management.html",
                    "summary": "Comprehensive guide to managing risks in agricultural commodity trading.",
                    "key_points": [
                        "Value at Risk calculation",
                        "Portfolio diversification",
                        "Stop-loss strategies",
                        "Stress testing"
                    ]
                },
                {
                    "title": "Indian Agricultural Market Live Data",
                    "category": "Market Data",
                    "level": "All Levels",
                    "reading_time": "Live",
                    "url": "https://agmarknet.gov.in/PriceAndArrivals/CommodityWise.aspx",
                    "summary": "Real-time market prices and arrivals data from agricultural markets across India.",
                    "key_points": [
                        "Daily mandi prices by commodity",
                        "Arrival trends and supply",
                        "State-wise production data",
                        "Historical comparisons"
                    ]
                },
                {
                    "title": "Options Trading in Commodities",
                    "category": "Advanced Strategy",
                    "level": "Advanced",
                    "reading_time": "18 min",
                    "url": "https://www.investopedia.com/options-basics-tutorial-4583012",
                    "summary": "Use options for hedging and generating income from commodity positions.",
                    "key_points": [
                        "Call and put fundamentals",
                        "Protective puts for downside",
                        "Covered calls for income",
                        "Collar strategies"
                    ]
                }
            ]
            article_category = st.selectbox("Filter by Category", 
                ['All', 'Comprehensive Guide', 'Trading Basics', 'Risk Management', 'Strategy', 'Market Analysis', 'Market Data', 'Advanced Strategy'],
                key="article_filter")
            filtered_articles = [a for a in articles if article_category == 'All' or a['category'] == article_category]
            for idx, article in enumerate(filtered_articles):
                with st.expander(f"📰 {article['title']} - {article['reading_time']} read"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Category:** {article['category']}")
                        st.markdown(f"**Level:** {article['level']}")
                    with col2:
                        st.markdown(f"**⏱️ {article['reading_time']}**")
                    st.write(article['summary'])
                    st.markdown("**📌 Key Points Covered:**")
                    for point in article['key_points']:
                        st.markdown(f"• {point}")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"🔗 [**Read Full Article →**]({article['url']})")
                    with col_b:
                        if st.button(f"📋 Copy Link", key=f"copy_{idx}"):
                            st.code(article['url'], language=None)
    elif menu == "📊 Analytics":
        st.header("📊 Advanced Analytics & Insights")
        tab1, tab2, tab3 = st.tabs(["📈 Market Analysis", "🔥 Heat Maps", "📉 Volatility Analysis"])
        with tab1:
            st.subheader("📈 Comprehensive Market Analysis")
            analysis_commodity = st.selectbox("Select Commodity for Analysis", list(st.session_state.market_data.keys()), key="analysis_commodity")
            commodity_data = st.session_state.price_data[st.session_state.price_data['commodity'] == analysis_commodity]
            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
            current_price = commodity_data['price'].iloc[-1]
            avg_price = commodity_data['price'].mean()
            min_price = commodity_data['price'].min()
            max_price = commodity_data['price'].max()
            price_volatility = commodity_data['price'].std()
            metric_col1.metric("Current", f"₹{current_price:.2f}")
            metric_col2.metric("Average", f"₹{avg_price:.2f}")
            metric_col3.metric("Min (6M)", f"₹{min_price:.2f}")
            metric_col4.metric("Max (6M)", f"₹{max_price:.2f}")
            metric_col5.metric("Volatility (σ)", f"₹{price_volatility:.2f}")
            fig_candlestick = go.Figure()
            commodity_data_resampled = commodity_data.set_index('date').resample('W').agg({
                'price': ['first', 'max', 'min', 'last'],
                'volume': 'sum'
            })
            commodity_data_resampled.columns = ['open', 'high', 'low', 'close', 'volume']
            commodity_data_resampled = commodity_data_resampled.reset_index()
            fig_candlestick.add_trace(go.Candlestick(
                x=commodity_data_resampled['date'],
                open=commodity_data_resampled['open'],
                high=commodity_data_resampled['high'],
                low=commodity_data_resampled['low'],
                close=commodity_data_resampled['close'],
                name=analysis_commodity
            ))
            fig_candlestick.update_layout(
                title=f"{analysis_commodity} - Weekly Candlestick Chart",
                xaxis_title="Date",
                yaxis_title="Price (₹/quintal)",
                height=500,
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig_candlestick, use_container_width=True)
            st.subheader("📊 Volume Analysis")
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=commodity_data['date'],
                y=commodity_data['volume'],
                name='Volume',
                marker_color='lightblue'
            ))
            fig_volume.update_layout(
                title="Trading Volume Over Time",
                xaxis_title="Date",
                yaxis_title="Volume (quintals)",
                height=300
            )
            st.plotly_chart(fig_volume, use_container_width=True)
            st.subheader("📉 Technical Indicators")
            commodity_data_copy = commodity_data.copy()
            commodity_data_copy['SMA_20'] = commodity_data_copy['price'].rolling(20).mean()
            commodity_data_copy['SMA_50'] = commodity_data_copy['price'].rolling(50).mean()
            commodity_data_copy['EMA_12'] = commodity_data_copy['price'].ewm(span=12, adjust=False).mean()
            commodity_data_copy['EMA_26'] = commodity_data_copy['price'].ewm(span=26, adjust=False).mean()
            fig_technical = go.Figure()
            fig_technical.add_trace(go.Scatter(x=commodity_data_copy['date'], y=commodity_data_copy['price'], mode='lines', name='Price', line=dict(color='blue')))
            fig_technical.add_trace(go.Scatter(x=commodity_data_copy['date'], y=commodity_data_copy['SMA_20'], mode='lines', name='SMA 20', line=dict(color='orange', dash='dash')))
            fig_technical.add_trace(go.Scatter(x=commodity_data_copy['date'], y=commodity_data_copy['SMA_50'], mode='lines', name='SMA 50', line=dict(color='red', dash='dash')))
            fig_technical.update_layout(
                title="Price with Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price (₹/quintal)",
                height=400
            )
            st.plotly_chart(fig_technical, use_container_width=True)
        with tab2:
            st.subheader("🔥 Market Heat Map")
            heatmap_data = []
            for commodity, data in st.session_state.market_data.items():
                heatmap_data.append({
                    'Commodity': commodity,
                    'Price': data['price'],
                    'Change %': data['change'],
                    'Volume': data['volume'],
                    'Volatility': data['volatility']
                })
            heatmap_df = pd.DataFrame(heatmap_data)
            fig_heatmap = px.imshow(
                heatmap_df.set_index('Commodity')[['Change %', 'Volatility']].T,
                labels=dict(x="Commodity", y="Metric", color="Value"),
                x=heatmap_df['Commodity'],
                y=['Change %', 'Volatility'],
                color_continuous_scale='RdYlGn',
                aspect="auto"
            )
            fig_heatmap.update_layout(title="Market Performance Heat Map", height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            st.subheader("📊 Correlation Matrix")
            correlation_data = {}
            for commodity in st.session_state.market_data.keys():
                commodity_prices = st.session_state.price_data[st.session_state.price_data['commodity'] == commodity]['price'].values
                if len(commodity_prices) > 0:
                    correlation_data[commodity] = commodity_prices[:min(len(commodity_prices), 180)]
            max_len = max(len(v) for v in correlation_data.values())
            for k in correlation_data.keys():
                if len(correlation_data[k]) < max_len:
                    correlation_data[k] = np.pad(correlation_data[k], (0, max_len - len(correlation_data[k])), mode='edge')
            corr_df = pd.DataFrame(correlation_data)
            correlation_matrix = corr_df.corr()
            fig_corr = px.imshow(
                correlation_matrix,
                labels=dict(color="Correlation"),
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                color_continuous_scale='RdBu',
                aspect="auto",
                zmin=-1,
                zmax=1
            )
            fig_corr.update_layout(title="Commodity Price Correlation Matrix", height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
        with tab3:
            st.subheader("📉 Volatility Analysis")
            vol_commodity = st.selectbox("Select Commodity", list(st.session_state.market_data.keys()), key="vol_commodity")
            vol_window = st.slider("Rolling Window (days)", 7, 60, 30)
            vol_data = st.session_state.price_data[st.session_state.price_data['commodity'] == vol_commodity].copy()
            vol_data['returns'] = vol_data['price'].pct_change()
            vol_data['volatility'] = vol_data['returns'].rolling(vol_window).std() * np.sqrt(252) * 100
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=vol_data['date'],
                y=vol_data['volatility'],
                mode='lines',
                name='Volatility',
                fill='tonexty',
                line=dict(color='red')
            ))
            fig_vol.update_layout(
                title=f"{vol_commodity} - Rolling Volatility ({vol_window} days)",
                xaxis_title="Date",
                yaxis_title="Annualized Volatility (%)",
                height=400
            )
            st.plotly_chart(fig_vol, use_container_width=True)
            current_vol = vol_data['volatility'].iloc[-1]
            avg_vol = vol_data['volatility'].mean()
            max_vol = vol_data['volatility'].max()
            min_vol = vol_data['volatility'].min()
            vol_metric_col1, vol_metric_col2, vol_metric_col3, vol_metric_col4 = st.columns(4)
            vol_metric_col1.metric("Current Volatility", f"{current_vol:.2f}%")
            vol_metric_col2.metric("Average Volatility", f"{avg_vol:.2f}%")
            vol_metric_col3.metric("Max Volatility", f"{max_vol:.2f}%")
            vol_metric_col4.metric("Min Volatility", f"{min_vol:.2f}%")
            if current_vol > avg_vol * 1.5:
                st.error("⚠️ **High Volatility Alert!** Current volatility is significantly above average. Exercise caution in trading.")
            elif current_vol < avg_vol * 0.5:
                st.success("✅ **Low Volatility Period** - Stable market conditions. Good for conservative strategies.")
            else:
                st.info("📊 **Normal Volatility Range** - Market operating within typical volatility levels.")
    elif menu == "⚙️ Settings":
        st.header("⚙️ Account Settings & Preferences")
        tab1, tab2, tab3, tab4 = st.tabs(["👤 Profile", "🔔 Notifications", "🎨 Preferences", "🔐 Security"])
        with tab1:
            st.subheader("👤 Profile Information")
            user_info = st.session_state.users[st.session_state.username]
            profile_col1, profile_col2 = st.columns(2)
            with profile_col1:
                st.text_input("Username", value=st.session_state.username, disabled=True)
                new_location = st.text_input("Location", value=user_info.get('location', ''))
                new_phone = st.text_input("Phone Number", value=user_info.get('phone', ''))
            with profile_col2:
                st.text_input("Account Type", value=user_info.get('type', ''), disabled=True)
                kyc_status = st.selectbox("KYC Status", ["Not Submitted", "Pending", "Verified"], index=2)
                st.text_input("Member Since", value=user_info.get('created_at', '')[:10] if 'created_at' in user_info else 'N/A', disabled=True)
            if st.button("💾 Update Profile", use_container_width=True):
                user_info['location'] = new_location
                user_info['phone'] = new_phone
                st.success("✅ Profile updated successfully!")
                st.rerun()
        with tab2:
            st.subheader("🔔 Notification Preferences")
            enable_price_alerts = st.checkbox("Enable Price Movement Alerts", value=True)
            price_threshold = st.slider("Alert Threshold (%)", 1.0, 10.0, 3.0, 0.5)
            enable_contract_alerts = st.checkbox("Contract Status Notifications", value=True)
            enable_position_alerts = st.checkbox("Position P&L Alerts", value=True)
            enable_email = st.checkbox("Email Notifications", value=False)
            enable_sms = st.checkbox("SMS Notifications", value=False)
            if st.button("💾 Save Notification Settings", use_container_width=True):
                st.success("✅ Notification preferences saved!")
        with tab3:
            st.subheader("🎨 Display Preferences")
            theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
            language = st.selectbox("Language", ["English", "हिंदी", "मराठी", "ગુજરાતી"])
            currency = st.selectbox("Currency Display", ["INR (₹)", "USD ($)"])
            date_format = st.selectbox("Date Format", ["DD/MM/YYYY", "MM/DD/YYYY", "YYYY-MM-DD"])
            chart_type = st.selectbox("Default Chart Type", ["Line", "Candlestick", "Area"])
            if st.button("💾 Save Preferences", use_container_width=True):
                st.success("✅ Display preferences saved!")
        with tab4:
            st.subheader("🔐 Security Settings")
            st.markdown("#### Change Password")
            current_password = st.text_input("Current Password", type="password", key="current_pass")
            new_password = st.text_input("New Password", type="password", key="new_pass")
            confirm_new_password = st.text_input("Confirm New Password", type="password", key="confirm_new_pass")
            if st.button("🔑 Change Password", use_container_width=True):
                if new_password == confirm_new_password and len(new_password) >= 6:
                    hashed_pass = hashlib.sha256(new_password.encode()).hexdigest()
                    user_info['password'] = hashed_pass
                    st.success("✅ Password changed successfully!")
                else:
                    st.error("❌ Passwords don't match or too short!")
            st.divider()
            st.markdown("#### Two-Factor Authentication")
            enable_2fa = st.checkbox("Enable 2FA", value=False)
            if enable_2fa:
                st.info("📱 Scan QR code with authenticator app")
                st.text_input("Enter 2FA Code")
            st.divider()
            st.markdown("#### Active Sessions")
            st.markdown("**Current Session:** Desktop - Sangamner, Maharashtra")
            st.markdown("**Last Login:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            if st.button("🚪 Logout All Devices", use_container_width=True):
                st.warning("All sessions will be terminated")
st.markdown("---")
st.caption("© 2024 GrainShield Pro | Powered by AI & Blockchain | Contact: support@grainshield.in | Version 1.0.0")
