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
from deep_translator import GoogleTranslator
import kagglehub
from kagglehub import KaggleDatasetAdapter
@st.cache_data
def t(text, lang):
    if lang == 'en' or not text:
        return text
    try:
        translator = GoogleTranslator(source='en', target=lang)
        return translator.translate(text)
    except:
        return text
if 'language' not in st.session_state:
    st.session_state.language = 'en'
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
if 'module_progress' not in st.session_state:
    st.session_state.module_progress = {}
@st.cache_data
def load_kaggle_data():
    try:
        df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "santoshd3/crop-price-prediction", "")
        return df
    except:
        return None
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
lang = st.session_state.language
def translate_commodity(commodity_name, target_lang):
    if target_lang == 'en':
        return commodity_name
    commodity_translations = {
        'Soybean': {'hi': 'सोयाबीन', 'mr': 'सोयाबीन', 'gu': 'સોયાબીન', 'pa': 'ਸੋਇਆਬੀਨ', 'te': 'సోయాబీన్', 'kn': 'ಸೋಯಾಬೀನ್', 'ta': 'சோயாபீன்'},
        'Mustard': {'hi': 'सरसों', 'mr': 'मोहरी', 'gu': 'સરસવ', 'pa': 'ਸਰ੍ਹੋਂ', 'te': 'ఆవాలు', 'kn': 'ಸಾಸಿವೆ', 'ta': 'கடுகு'},
        'Groundnut': {'hi': 'मूंगफली', 'mr': 'भुईमूग', 'gu': 'મગફળી', 'pa': 'ਮੂੰਗਫਲੀ', 'te': 'వేరుశెనగ', 'kn': 'ಕಡಲೆಕಾಯಿ', 'ta': 'நிலக்கடலை'},
        'Sunflower': {'hi': 'सूरजमुखी', 'mr': 'सूर्यफूल', 'gu': 'સૂર્યમુખી', 'pa': 'ਸੂਰਜਮੁਖੀ', 'te': 'సూర్యకాంతి', 'kn': 'ಸೂರ್ಯಕಾಂತಿ', 'ta': 'சூரியகாந்தி'}
    }
    return commodity_translations.get(commodity_name, {}).get(target_lang, commodity_name)
st.title(t("🌾 GrainShield Pro - Oilseed Hedging Platform", lang))
st.caption(t("AI-Powered Risk Management & Blockchain-Secured Trading", lang))
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        tab1, tab2 = st.tabs([t("🔐 Login", lang), t("📝 Register", lang)])
        with tab1:
            st.subheader(t("Login to Your Account", lang))
            username = st.text_input(t("Username", lang), key="login_user")
            password = st.text_input(t("Password", lang), type="password", key="login_pass")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button(t("🔓 Login", lang), use_container_width=True):
                    if username in st.session_state.users:
                        hashed_pass = hashlib.sha256(password.encode()).hexdigest()
                        if st.session_state.users[username]['password'] == hashed_pass:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.notifications.append(f"Welcome back, {username}!")
                            st.rerun()
                        else:
                            st.error(t("❌ Invalid credentials", lang))
                    else:
                        st.error(t("❌ User not found", lang))
            with col_b:
                if st.button(t("Demo Login", lang), use_container_width=True):
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
            st.subheader(t("Create New Account", lang))
            new_username = st.text_input(t("Choose Username", lang), key="reg_user")
            new_password = st.text_input(t("Create Password", lang), type="password", key="reg_pass")
            confirm_password = st.text_input(t("Confirm Password", lang), type="password", key="confirm_pass")
            col1, col2 = st.columns(2)
            with col1:
                user_type = st.selectbox(t("Account Type", lang), ["Farmer", "FPO", "Trader", "Buyer"])
            with col2:
                location = st.text_input(t("Location/District", lang))
            phone = st.text_input(t("Mobile Number", lang))
            if st.button(t("✅ Create Account", lang), use_container_width=True):
                if new_password != confirm_password:
                    st.error(t("Passwords do not match!", lang))
                elif new_username in st.session_state.users:
                    st.error(t("Username already exists!", lang))
                elif len(new_username) < 3 or len(new_password) < 6:
                    st.error(t("Username must be 3+ chars and password 6+ chars", lang))
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
                    st.success(t("✅ Registration successful! Please login.", lang))
else:
    with st.sidebar:
        st.markdown("### 🌐 Language")
        languages = {
            "English 🇬🇧": "en (English)",
            "हिंदी 🇮🇳": "hi (Hindi)",
            "मराठी 🇮🇳": "mr (Marathi)",
            "ગુજરાતી 🇮🇳": "gu (Gujarati)",
            "ਪੰਜਾਬੀ 🇮🇳": "pa (Punjabi)",
            "తెలుగు 🇮🇳": "te (Telugu)",
            "ಕನ್ನಡ 🇮🇳": "kn (Kannada)",
            "தமிழ் 🇮🇳": "ta (Tamil)"    
        }
        current_lang_display = [k for k, v in languages.items() if v == st.session_state.language][0] if st.session_state.language in languages.values() else "English 🇬🇧"
        selected_lang = st.selectbox("", list(languages.keys()), index=list(languages.keys()).index(current_lang_display), label_visibility="collapsed")
        st.session_state.language = languages[selected_lang]
        lang = st.session_state.language
        st.divider()
        st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=80)
        st.title(f"👤 {st.session_state.username}")
        user_info = st.session_state.users[st.session_state.username]
        st.markdown(f"**{t('Type', lang)}:** {user_info['type']}")
        st.markdown(f"**{t('Location', lang)}:** {user_info.get('location', 'N/A')}")
        st.markdown(f"**{t('Balance', lang)}:** ₹{user_info['balance']:,.2f}")
        portfolio = calculate_portfolio_metrics(st.session_state.username)
        if portfolio['num_positions'] > 0:
            st.markdown(f"**{t('Active Positions', lang)}:** {portfolio['num_positions']}")
            st.markdown(f"**{t('Portfolio P&L', lang)}:** ₹{portfolio['total_pnl']:,.2f}")
            st.markdown(f"**{t('Avg Return', lang)}:** {portfolio['avg_return']:.2f}%")
        st.divider()
        unread_alerts = len([a for a in st.session_state.alerts if not a['read']])
        if unread_alerts > 0:
            st.warning(f"🔔 {unread_alerts} {t('New Alerts', lang)}")
        if st.session_state.notifications:
            with st.expander(f"📬 {t('Notifications', lang)}"):
                for notif in st.session_state.notifications[-5:]:
                    st.info(notif)
                if st.button(t("Clear Notifications", lang)):
                    st.session_state.notifications = []
                    st.rerun()
        st.divider()
        menu = st.selectbox(f"📍 {t('Navigation', lang)}", [
            t("🏠 Dashboard", lang),
            t("📈 Price Forecast", lang),
            t("💹 Virtual Hedging", lang),
            t("📋 Forward Contracts", lang),
            t("💼 My Portfolio", lang),
            t("🔗 Blockchain Ledger", lang),
            t("📊 Analytics", lang),
            t("⚙️ Settings", lang)
        ])
        st.divider()
        if st.button(t("🚪 Logout", lang), use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.notifications = []
            st.rerun()
    if t("🏠 Dashboard", lang) in menu:
        st.header(t("📊 Market Dashboard", lang))
        if st.session_state.notifications:
            for notif in st.session_state.notifications[-3:]:
                st.success(f"✅ {notif}")
        col1, col2, col3, col4, col5 = st.columns(5)
        current_price = st.session_state.price_data['price'].iloc[-1]
        price_change = current_price - st.session_state.price_data['price'].iloc[-2]
        col1.metric(t("Current Price", lang), f"₹{current_price:.2f}", f"{price_change:+.2f}")
        col2.metric(t("Daily Volume", lang), f"{st.session_state.price_data['volume'].iloc[-1]:,} qtl")
        col3.metric(t("Active Contracts", lang), len([c for c in st.session_state.contracts if c['status'] == 'Open']))
        col4.metric(t("Your Positions", lang), len([p for p in st.session_state.positions if p['username'] == st.session_state.username and p['status'] == 'Open']))
        portfolio = calculate_portfolio_metrics(st.session_state.username)
        col5.metric(t("Portfolio P&L", lang), f"₹{portfolio['total_pnl']:,.0f}", f"{portfolio['avg_return']:.1f}%")
        st.subheader(t("🌾 Live Commodity Prices", lang))
        cols = st.columns(4)
        for idx, (commodity, data) in enumerate(st.session_state.market_data.items()):
            with cols[idx]:
                delta_color = "normal" if data['change'] >= 0 else "inverse"
                st.metric(
                    translate_commodity(commodity, lang),
                    f"₹{data['price']:,.0f}",
                    f"{data['change']:+.1f}%",
                    delta_color=delta_color
                )
                st.caption(f"{t('Vol', lang)}: {data['volume']} | σ: {data['volatility']:.1f}%")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(t("📈 Historical Price Trend", lang))
            commodity_filter = st.selectbox(t("Select Commodity", lang), [t('All', lang)] + [translate_commodity(c, lang) for c in st.session_state.market_data.keys()], key="dash_commodity")
            if t('All', lang) in commodity_filter:
                plot_data = st.session_state.price_data
            else:
                original_commodity = [k for k, v in {k: translate_commodity(k, lang) for k in st.session_state.market_data.keys()}.items() if v == commodity_filter][0]
                plot_data = st.session_state.price_data[st.session_state.price_data['commodity'] == original_commodity]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_data['date'],
                y=plot_data['price'],
                mode='lines',
                name=t('Price', lang),
                line=dict(color='#1f77b4', width=2),
                fill='tonexty'
            ))
            fig.add_trace(go.Scatter(
                x=plot_data['date'],
                y=plot_data['price'].rolling(7).mean(),
                mode='lines',
                name=t('7-Day MA', lang),
                line=dict(color='orange', dash='dash')
            ))
            fig.update_layout(
                xaxis_title=t("Date", lang),
                yaxis_title=t("Price (₹/quintal)", lang),
                height=400,
                hovermode='x unified',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader(t("📊 Volume Distribution", lang))
            volume_by_commodity = st.session_state.price_data.groupby('commodity')['volume'].sum().reset_index()
            fig_pie = px.pie(
                volume_by_commodity,
                values='volume',
                names='commodity',
                title=t('Trading Volume by Commodity', lang)
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        st.subheader(t("🔔 Recent Market Alerts", lang))
        alert_cols = st.columns(3)
        for idx, alert in enumerate(st.session_state.alerts[:3]):
            with alert_cols[idx % 3]:
                time_diff = datetime.now() - alert['time']
                if time_diff.days > 0:
                    time_str = f"{time_diff.days}d {t('ago', lang)}"
                elif time_diff.seconds > 3600:
                    time_str = f"{time_diff.seconds // 3600}h {t('ago', lang)}"
                else:
                    time_str = f"{time_diff.seconds // 60}m {t('ago', lang)}"
                if alert['type'] == 'warning':
                    st.warning(f"⚠️ **{time_str}**\n\n{alert['message']}")
                elif alert['type'] == 'success':
                    st.success(f"✅ **{time_str}**\n\n{alert['message']}")
                else:
                    st.info(f"ℹ️ **{time_str}**\n\n{alert['message']}")
        if st.button(t("Mark All as Read", lang)):
            for alert in st.session_state.alerts:
                alert['read'] = True
            st.rerun()
        st.subheader(t("🎯 Quick Actions", lang))
        qcol1, qcol2, qcol3, qcol4 = st.columns(4)
        with qcol1:
            if st.button(t("📈 New Hedge Position", lang), use_container_width=True):
                st.session_state.quick_action = "hedge"
                st.rerun()
        with qcol2:
            if st.button(t("📋 Create Contract", lang), use_container_width=True):
                st.session_state.quick_action = "contract"
                st.rerun()
        with qcol3:
            if st.button(t("🔮 View Forecast", lang), use_container_width=True):
                st.session_state.quick_action = "forecast"
                st.rerun()
        with qcol4:
            if st.button(t("📚 Start Learning", lang), use_container_width=True):
                st.session_state.quick_action = "learn"
                st.rerun()
    elif t("📈 Price Forecast", lang) in menu:
        st.header(t("🔮 AI-Powered Price Forecasting", lang))
        col1, col2, col3 = st.columns(3)
        with col1:
            forecast_commodity = st.selectbox(t("Select Commodity", lang), [translate_commodity(c, lang) for c in st.session_state.market_data.keys()])
            forecast_commodity_en = [k for k, v in {k: translate_commodity(k, lang) for k in st.session_state.market_data.keys()}.items() if v == forecast_commodity][0]
        with col2:
            forecast_days = st.slider(t("Forecast Period (days)", lang), 7, 90, 30)
        with col3:
            confidence_level = st.selectbox(t("Confidence Level", lang), ["68%", "95%", "99%"])
        if st.button(t("🚀 Generate Forecast", lang), use_container_width=True):
            with st.spinner(t("🤖 AI analyzing market patterns...", lang)):
                predictions = predict_prices(forecast_days, forecast_commodity_en)
                historical = st.session_state.price_data[st.session_state.price_data['commodity'] == forecast_commodity_en].copy()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=historical['date'],
                    y=historical['price'],
                    mode='lines',
                    name=t('Historical Price', lang),
                    line=dict(color='blue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=predictions['date'],
                    y=predictions['predicted_price'],
                    mode='lines',
                    name=t('Predicted Price', lang),
                    line=dict(color='red', width=2, dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=predictions['date'].tolist() + predictions['date'].tolist()[::-1],
                    y=predictions['upper_bound'].tolist() + predictions['lower_bound'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=t('Confidence Interval', lang)
                ))
                fig.update_layout(
                    xaxis_title=t("Date", lang),
                    yaxis_title=t("Price (₹/quintal)", lang),
                    height=500,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                col1, col2, col3, col4 = st.columns(4)
                last_historical = historical['price'].iloc[-1]
                predicted_30d = predictions['predicted_price'].iloc[min(29, len(predictions)-1)]
                expected_change = ((predicted_30d - last_historical) / last_historical) * 100
                col1.metric(t("Current Price", lang), f"₹{last_historical:.2f}")
                col2.metric(t("30-Day Forecast", lang), f"₹{predicted_30d:.2f}", f"{expected_change:+.2f}%")
                col3.metric(t("Volatility", lang), f"{predictions['predicted_price'].std():.2f}")
                col4.metric(t("Trend", lang), t("📈 Bullish", lang) if expected_change > 0 else t("📉 Bearish", lang))
                st.subheader(t("📊 Forecast Statistics", lang))
                forecast_stats = pd.DataFrame({
                    t('Metric', lang): [t('Mean', lang), t('Median', lang), t('Min', lang), t('Max', lang), t('Std Dev', lang)],
                    t('Value', lang): [
                        f"₹{predictions['predicted_price'].mean():.2f}",
                        f"₹{predictions['predicted_price'].median():.2f}",
                        f"₹{predictions['predicted_price'].min():.2f}",
                        f"₹{predictions['predicted_price'].max():.2f}",
                        f"₹{predictions['predicted_price'].std():.2f}"
                    ]
                })
                st.dataframe(forecast_stats, use_container_width=True, hide_index=True)
                st.subheader(t("💡 AI Insights & Recommendations", lang))
                if expected_change > 5:
                    st.success(t("🎯 **Strong Uptrend Expected:** Consider long positions or holding existing stock", lang))
                elif expected_change > 2:
                    st.info(t("📈 **Moderate Uptrend:** Good time for partial hedging", lang))
                elif expected_change < -5:
                    st.error(t("⚠️ **Strong Downtrend Expected:** Consider hedging or forward contracts", lang))
                elif expected_change < -2:
                    st.warning(t("📉 **Moderate Downtrend:** Monitor closely and consider protective positions", lang))
                else:
                    st.info(t("➡️ **Stable Market:** Range-bound trading expected", lang))
                volatility_index = predictions['predicted_price'].std() / predictions['predicted_price'].mean() * 100
                if volatility_index > 10:
                    st.warning(t(f"⚡ High volatility detected ({volatility_index:.1f}%). Risk management strongly recommended.", lang))
    elif t("💹 Virtual Hedging", lang) in menu:
        st.header(t("💹 Virtual Futures Trading Simulator", lang))
        tab1, tab2, tab3 = st.tabs([t("📈 Open Position", lang), t("🧮 Calculator", lang), t("📚 Strategy Guide", lang)])
        with tab1:
            st.subheader(t("🎯 Open New Hedging Position", lang))
            col1, col2 = st.columns(2)
            with col1:
                position_type = st.selectbox(t("Position Type", lang), [t("Long (Buy)", lang), t("Short (Sell)", lang)])
                commodity = st.selectbox(t("Commodity", lang), [translate_commodity(c, lang) for c in st.session_state.market_data.keys()])
                commodity_en = [k for k, v in {k: translate_commodity(k, lang) for k in st.session_state.market_data.keys()}.items() if v == commodity][0]
                quantity = st.number_input(t("Quantity (quintals)", lang), min_value=1, max_value=10000, value=10)
            with col2:
                current_market_price = st.session_state.market_data[commodity_en]['price']
                entry_price = st.number_input(t("Entry Price (₹/quintal)", lang), value=float(current_market_price), min_value=1000.0)
                expiry_date = st.date_input(t("Expiry Date", lang), min_value=datetime.now().date() + timedelta(days=1), value=datetime.now().date() + timedelta(days=30))
                leverage = st.selectbox(t("Leverage", lang), ["1x", "2x", "3x", "5x"])
            col1, col2, col3 = st.columns(3)
            position_value = entry_price * quantity
            margin_required = position_value * 0.1
            potential_gain = entry_price * quantity * 0.1
            col1.metric(t("Position Value", lang), f"₹{position_value:,.2f}")
            col2.metric(t("Margin Required", lang), f"₹{margin_required:,.2f}")
            col3.metric(t("Potential 10% Gain", lang), f"₹{potential_gain:,.2f}")
            stop_loss = st.number_input(t("Stop Loss Price (₹/quintal)", lang), value=float(entry_price * 0.95), min_value=1000.0)
            take_profit = st.number_input(t("Take Profit Price (₹/quintal)", lang), value=float(entry_price * 1.1), min_value=1000.0)
            notes = st.text_area(t("Position Notes", lang), "")
            if st.button(t("🚀 Open Position", lang), use_container_width=True, type="primary"):
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
                    st.success(t("✅ Position opened successfully!", lang))
                    st.info(f"{t('📋 Transaction Hash:', lang)} {tx_hash[:32]}...")
                    st.session_state.notifications.append(t(f"New {position_type} position opened for {commodity}", lang))
                    st.balloons()
                    st.rerun()
                else:
                    st.error(t(f"❌ Insufficient balance! Required: ₹{margin_required:,.2f}, Available: ₹{st.session_state.users[st.session_state.username]['balance']:,.2f}", lang))
        with tab2:
            st.subheader(t("🧮 Position Calculator & Risk Assessment", lang))
            calc_col1, calc_col2 = st.columns(2)
            with calc_col1:
                calc_commodity = st.selectbox(t("Commodity", lang), [translate_commodity(c, lang) for c in st.session_state.market_data.keys()], key="fc_commodity")
                commodity_en = [k for k, v in {k: translate_commodity(k, lang) for k in st.session_state.market_data.keys()}.items() if v == commodity][0]
                calc_quantity = st.number_input(t("Quantity (quintals)", lang), min_value=1, value=10, key="calc_qty")
                calc_entry = st.number_input(t("Entry Price (₹/quintal)", lang), value=float(st.session_state.market_data[calc_commodity]['price']), key="calc_entry")
            with calc_col2:
                calc_target = st.number_input(t("Target Price (₹/quintal)", lang), value=float(calc_entry * 1.1), key="calc_target")
                calc_stoploss = st.number_input(t("Stop Loss Price (₹/quintal)", lang), value=float(calc_entry * 0.95), key="calc_sl")
                calc_type = st.selectbox(t("Position Type", lang), [t("Long", lang), t("Short", lang)], key="calc_type")
            if t("Long", lang) in calc_type:
                potential_profit = (calc_target - calc_entry) * calc_quantity
                potential_loss = (calc_entry - calc_stoploss) * calc_quantity
            else:
                potential_profit = (calc_entry - calc_target) * calc_quantity
                potential_loss = (calc_stoploss - calc_entry) * calc_quantity
            risk_reward_ratio = abs(potential_profit / potential_loss) if potential_loss != 0 else 0
            investment = calc_entry * calc_quantity
            roi_profit = (potential_profit / investment) * 100
            roi_loss = (potential_loss / investment) * 100
            st.subheader(t("📊 Risk-Reward Analysis", lang))
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric(t("Investment", lang), f"₹{investment:,.2f}")
            metric_col2.metric(t("Potential Profit", lang), f"₹{potential_profit:,.2f}", f"+{roi_profit:.1f}%")
            metric_col3.metric(t("Potential Loss", lang), f"₹{potential_loss:,.2f}", f"-{roi_loss:.1f}%")
            metric_col4.metric(t("Risk:Reward", lang), f"1:{risk_reward_ratio:.2f}")
            if risk_reward_ratio >= 2:
                st.success(t("✅ Excellent Risk-Reward Ratio (>2:1)", lang))
            elif risk_reward_ratio >= 1.5:
                st.info(t("📊 Good Risk-Reward Ratio (>1.5:1)", lang))
            else:
                st.warning(t("⚠️ Poor Risk-Reward Ratio (<1.5:1) - Consider adjusting targets", lang))
            fig = go.Figure()
            prices = np.linspace(calc_entry * 0.8, calc_entry * 1.2, 100)
            if t("Long", lang) in calc_type:
                pnl = (prices - calc_entry) * calc_quantity
            else:
                pnl = (calc_entry - prices) * calc_quantity
            fig.add_trace(go.Scatter(x=prices, y=pnl, mode='lines', name=t('P&L', lang), line=dict(color='blue', width=2)))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=calc_entry, line_dash="dash", line_color="orange", annotation_text=t("Entry", lang))
            fig.add_vline(x=calc_target, line_dash="dash", line_color="green", annotation_text=t("Target", lang))
            fig.add_vline(x=calc_stoploss, line_dash="dash", line_color="red", annotation_text=t("Stop Loss", lang))
            fig.update_layout(title=t("P&L Profile", lang), xaxis_title=t("Price (₹/quintal)", lang), yaxis_title=t("Profit/Loss (₹)", lang), height=400)
            st.plotly_chart(fig, use_container_width=True)
        with tab3:
            st.subheader(t("📚 Hedging Strategies Guide", lang))
            strategy_cols = st.columns(2)
            with strategy_cols[0]:
                st.markdown(f"### {t('🛡️ Protective Hedging Strategies', lang)}")
                st.markdown(t("""
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
                """, lang))
            with strategy_cols[1]:
                st.markdown(f"### {t('📈 Speculative Strategies', lang)}")
                st.markdown(t("""
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
                """, lang))
            st.info(t("💡 **Pro Tip:** Start with protective hedging strategies before attempting speculative trades. Always use stop losses!", lang))
    elif t("📋 Forward Contracts", lang) in menu:
        st.header(t("📋 Forward Contract Management System", lang))
        tab1, tab2, tab3 = st.tabs([t("✍️ Create Contract", lang), t("🔍 Browse Contracts", lang), t("📊 My Contracts", lang)])
        with tab1:
            st.subheader(t("✍️ Create New Forward Contract", lang))
            fc_col1, fc_col2 = st.columns(2)
            with fc_col1:
                contract_type = st.selectbox(t("Contract Type", lang), [t("Sell Forward", lang), t("Buy Forward", lang)])
                commodity = st.selectbox(t("Commodity", lang), list(st.session_state.market_data.keys()), key="fc_commodity")
                quantity = st.number_input(t("Quantity (quintals)", lang), min_value=1, max_value=100000, value=100, key="fc_qty")
                quality_grade = st.selectbox(t("Quality Grade", lang), [t("Premium", lang), t("Grade A", lang), t("Grade B", lang), t("Standard", lang)])
            with fc_col2:
                contract_price = st.number_input(t("Contract Price (₹/quintal)", lang), value=float(st.session_state.market_data[commodity]['price']), key="fc_price")
                delivery_date = st.date_input(t("Delivery Date", lang), min_value=datetime.now().date() + timedelta(days=30), value=datetime.now().date() + timedelta(days=60), key="fc_date")
                delivery_location = st.text_input(t("Delivery Location", lang), "")
                payment_terms = st.selectbox(t("Payment Terms", lang), [t("Advance 25%", lang), t("Advance 50%", lang), t("On Delivery", lang), t("30 Days Credit", lang)])
            terms = st.text_area(t("Contract Terms & Conditions", lang), t("1. Quality as per NCDEX standards\n2. Moisture content < 12%\n3. Packaging: New gunny bags\n4. Transportation: Buyer's responsibility\n5. Dispute resolution: Local arbitration", lang))
            advance_amount = st.number_input(t("Advance Amount (₹)", lang), min_value=0, value=int(contract_price * quantity * 0.1), key="fc_advance")
            if st.button(t("📝 Create Contract", lang), use_container_width=True, type="primary"):
                contract = {
                    'id': len(st.session_state.contracts) + 1,
                    'creator': st.session_state.username,
                    'type': contract_type,
                    'commodity': commodity_en,
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
                st.success(t(f"✅ Contract #{contract['id']} created successfully!", lang))
                st.info(t(f"📋 Blockchain TX: {tx_hash[:32]}...", lang))
                st.session_state.notifications.append(t(f"New forward contract created: {commodity} - {quantity} qtl", lang))
                st.balloons()
                st.rerun()
        with tab2:
            st.subheader(t("🔍 Browse Available Contracts", lang))
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                filter_commodity = st.multiselect(t("Filter by Commodity", lang), ['All'] + list(st.session_state.market_data.keys()), default=['All'])
            with filter_col2:
                filter_type = st.multiselect(t("Contract Type", lang), ['All', t('Sell Forward', lang), t('Buy Forward', lang)], default=['All'])
            with filter_col3:
                sort_by = st.selectbox(t("Sort By", lang), [t("Latest First", lang), t("Price: Low to High", lang), t("Price: High to Low", lang), t("Quantity: High to Low", lang)])
            open_contracts = [c for c in st.session_state.contracts if c['status'] == 'Open' and c['creator'] != st.session_state.username]
            if 'All' not in filter_commodity:
                open_contracts = [c for c in open_contracts if c['commodity'] in filter_commodity]
            if 'All' not in filter_type:
                open_contracts = [c for c in open_contracts if c['type'] in filter_type]
            if t("Price: Low to High", lang) in sort_by:
                open_contracts.sort(key=lambda x: x['price'])
            elif t("Price: High to Low", lang) in sort_by:
                open_contracts.sort(key=lambda x: x['price'], reverse=True)
            elif t("Quantity: High to Low", lang) in sort_by:
                open_contracts.sort(key=lambda x: x['quantity'], reverse=True)
            else:
                open_contracts.sort(key=lambda x: x['timestamp'], reverse=True)
            if open_contracts:
                st.info(t(f"📊 Found {len(open_contracts)} available contracts", lang))
                for contract in open_contracts:
                    contract['views'] = contract.get('views', 0) + 1
                    with st.expander(t(f"🔖 Contract #{contract['id']} - {contract['commodity']} - {contract['type']} - {contract['quantity']} qtl @ ₹{contract['price']:.2f}", lang)):
                        info_col1, info_col2, info_col3 = st.columns(3)
                        with info_col1:
                            st.markdown(f"**{t('👤 Seller:', lang)}** {contract['creator']}")
                            st.markdown(f"**{t('📦 Quantity:', lang)}** {contract['quantity']} {t('quintals', lang)}")
                            st.markdown(f"**{t('⭐ Quality:', lang)}** {contract['quality_grade']}")
                        with info_col2:
                            st.markdown(f"**{t('💰 Price:', lang)}** ₹{contract['price']:.2f}/{t('qtl', lang)}")
                            st.markdown(f"**{t('📅 Delivery:', lang)}** {contract['delivery_date']}")
                            st.markdown(f"**{t('📍 Location:', lang)}** {contract.get('delivery_location', 'N/A')}")
                        with info_col3:
                            total_value = contract['price'] * contract['quantity']
                            st.markdown(f"**{t('💵 Total Value:', lang)}** ₹{total_value:,.2f}")
                            st.markdown(f"**{t('💳 Payment:', lang)}** {contract.get('payment_terms', t('On Delivery', lang))}")
                            st.markdown(f"**{t('👁️ Views:', lang)}** {contract.get('views', 0)}")
                        st.markdown(f"**{t('📋 Terms & Conditions:', lang)}**")
                        st.text(contract['terms'])
                        action_col1, action_col2, action_col3 = st.columns([2,1,1])
                        with action_col1:
                            if st.button(t(f"✅ Accept Contract #{contract['id']}", lang), key=f"accept_{contract['id']}", use_container_width=True):
                                contract['status'] = 'Accepted'
                                contract['counterparty'] = st.session_state.username
                                contract['acceptance_date'] = datetime.now().isoformat()
                                tx_hash = add_to_blockchain('CONTRACT_ACCEPTED', {
                                    'contract_id': contract['id'],
                                    'counterparty': st.session_state.username,
                                    'acceptance_date': datetime.now().isoformat()
                                })
                                st.success(t(f"✅ Contract accepted! TX: {tx_hash[:16]}...", lang))
                                st.session_state.notifications.append(t(f"Contract #{contract['id']} accepted successfully", lang))
                                st.rerun()
                        with action_col2:
                            if st.button(t("💬 Contact", lang), key=f"contact_{contract['id']}", use_container_width=True):
                                st.info(t(f"Contact: {contract['creator']} via platform messaging", lang))
                        with action_col3:
                            if st.button(t("📊 Analyze", lang), key=f"analyze_{contract['id']}", use_container_width=True):
                                current_market = st.session_state.market_data[contract['commodity']]['price']
                                price_diff = contract['price'] - current_market
                                st.metric(t("vs Market Price", lang), f"₹{price_diff:+,.2f}", f"{(price_diff/current_market*100):+.2f}%")
            else:
                st.warning(t("⚠️ No contracts match your filters. Try adjusting the filters or create a new contract.", lang))
        with tab3:
            st.subheader(t("📊 My Contracts Dashboard", lang))
            my_contracts = [c for c in st.session_state.contracts if c['creator'] == st.session_state.username or c.get('counterparty') == st.session_state.username]
            if my_contracts:
                status_counts = {'Open': 0, 'Accepted': 0, 'Completed': 0, 'Cancelled': 0}
                total_value = 0
                for c in my_contracts:
                    status_counts[c['status']] = status_counts.get(c['status'], 0) + 1
                    if c['status'] in ['Open', 'Accepted']:
                        total_value += c['price'] * c['quantity']
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                metric_col1.metric(t("Total Contracts", lang), len(my_contracts))
                metric_col2.metric(t("Active Value", lang), f"₹{total_value:,.0f}")
                metric_col3.metric(t("Open", lang), status_counts['Open'])
                metric_col4.metric(t("Accepted", lang), status_counts['Accepted'])
                for contract in my_contracts:
                    is_creator = contract['creator'] == st.session_state.username
                    status_emoji = {"Open": "🟡", "Accepted": "🟢", "Completed": "✅", "Cancelled": "❌"}
                    with st.expander(f"{status_emoji.get(contract['status'], '⚪')} {t('Contract', lang)} #{contract['id']} - {contract['commodity']} - {contract['status']}"):
                        role_col1, role_col2 = st.columns(2)
                        with role_col1:
                            st.markdown(f"**{t('Your Role:', lang)}** {t('Seller', lang) if is_creator and t('Sell Forward', lang) in contract['type'] else t('Buyer', lang)}")
                            st.markdown(f"**{t('Creator:', lang)}** {contract['creator']}")
                            st.markdown(f"**{t('Counterparty:', lang)}** {contract.get('counterparty', t('Pending', lang))}")
                        with role_col2:
                            st.markdown(f"**{t('Commodity:', lang)}** {contract['commodity']}")
                            st.markdown(f"**{t('Quantity:', lang)}** {contract['quantity']} {t('qtl', lang)}")
                            st.markdown(f"**{t('Price:', lang)}** ₹{contract['price']:.2f}/{t('qtl', lang)}")
                        st.markdown(f"**{t('Delivery Date:', lang)}** {contract['delivery_date']}")
                        st.markdown(f"**{t('Status:', lang)}** {contract['status']}")
                        st.markdown(f"**{t('Total Value:', lang)}** ₹{contract['price'] * contract['quantity']:,.2f}")
                        if contract['status'] == 'Open' and is_creator:
                            action_col1, action_col2 = st.columns(2)
                            with action_col1:
                                if st.button(t("📝 Edit Contract", lang), key=f"edit_{contract['id']}"):
                                    st.info(t("Edit functionality - Update contract terms", lang))
                            with action_col2:
                                if st.button(t("❌ Cancel Contract", lang), key=f"cancel_{contract['id']}"):
                                    contract['status'] = 'Cancelled'
                                    add_to_blockchain('CONTRACT_CANCELLED', {'contract_id': contract['id']})
                                    st.rerun()
                        elif contract['status'] == 'Accepted':
                            if st.button(t("✅ Mark as Completed", lang), key=f"complete_{contract['id']}"):
                                contract['status'] = 'Completed'
                                contract['completion_date'] = datetime.now().isoformat()
                                add_to_blockchain('CONTRACT_COMPLETED', {'contract_id': contract['id']})
                                st.success(t("Contract marked as completed!", lang))
                                st.rerun()
            else:
                st.info(t("📭 No contracts yet. Create your first contract in the 'Create Contract' tab.", lang))
    elif t("💼 My Portfolio", lang) in menu:
        st.header(t("💼 Portfolio Management Dashboard", lang))
        portfolio = calculate_portfolio_metrics(st.session_state.username)
        overview_col1, overview_col2, overview_col3, overview_col4, overview_col5 = st.columns(5)
        overview_col1.metric(t("Total Portfolio Value", lang), f"₹{portfolio['total_value']:,.0f}")
        overview_col2.metric(t("Total P&L", lang), f"₹{portfolio['total_pnl']:,.0f}", f"{portfolio['avg_return']:.2f}%")
        overview_col3.metric(t("Active Positions", lang), portfolio['num_positions'])
        overview_col4.metric(t("Account Balance", lang), f"₹{st.session_state.users[st.session_state.username]['balance']:,.0f}")
        overview_col5.metric(t("Total Equity", lang), f"₹{st.session_state.users[st.session_state.username]['balance'] + portfolio['total_pnl']:,.0f}")
        tab1, tab2, tab3 = st.tabs([t("📈 Open Positions", lang), t("📊 Performance Analytics", lang), t("📜 Transaction History", lang)])
        with tab1:
            st.subheader(t("📈 Active Positions", lang))
            user_positions = [p for p in st.session_state.positions if p['username'] == st.session_state.username and p['status'] == 'Open']
            if user_positions:
                for idx, pos in enumerate(user_positions):
                    current_price = st.session_state.market_data.get(pos['commodity'], {}).get('price', pos['entry_price'])
                    pnl = (current_price - pos['entry_price']) * pos['quantity'] if pos['type'] == "Long (Buy)" else (pos['entry_price'] - current_price) * pos['quantity']
                    pnl_pct = (pnl / (pos['entry_price'] * pos['quantity'])) * 100
                    pnl_color = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                    with st.expander(f"{pnl_color} {t('Position', lang)} #{idx+1} - {pos['commodity']} {pos['type']} - {t('P&L', lang)}: ₹{pnl:,.0f} ({pnl_pct:+.2f}%)"):
                        pos_col1, pos_col2, pos_col3, pos_col4 = st.columns(4)
                        pos_col1.metric(t("Entry Price", lang), f"₹{pos['entry_price']:.2f}")
                        pos_col2.metric(t("Current Price", lang), f"₹{current_price:.2f}", f"{((current_price-pos['entry_price'])/pos['entry_price']*100):+.2f}%")
                        pos_col3.metric(t("Quantity", lang), f"{pos['quantity']} {t('qtl', lang)}")
                        pos_col4.metric(t("P&L", lang), f"₹{pnl:,.2f}", f"{pnl_pct:+.2f}%")
                        detail_col1, detail_col2, detail_col3 = st.columns(3)
                        with detail_col1:
                            st.markdown(f"**{t('Expiry:', lang)}** {pos['expiry_date']}")
                            st.markdown(f"**{t('Leverage:', lang)}** {pos.get('leverage', '1x')}")
                        with detail_col2:
                            st.markdown(f"**{t('Stop Loss:', lang)}** ₹{pos.get('stop_loss', 'N/A')}")
                            st.markdown(f"**{t('Take Profit:', lang)}** ₹{pos.get('take_profit', 'N/A')}")
                        with detail_col3:
                            st.markdown(f"**{t('Margin:', lang)}** ₹{pos.get('margin_required', 0):,.2f}")
                            st.markdown(f"**{t('Opened:', lang)}** {pos['timestamp'][:10]}")
                        if pos.get('notes'):
                            st.markdown(f"**{t('Notes:', lang)}** {pos['notes']}")
                        action_col1, action_col2, action_col3 = st.columns(3)
                        with action_col1:
                            if st.button(t("❌ Close Position", lang), key=f"close_pos_{idx}", use_container_width=True):
                                pos['status'] = 'Closed'
                                pos['exit_price'] = current_price
                                pos['close_date'] = datetime.now().isoformat()
                                st.session_state.users[st.session_state.username]['balance'] += pnl + pos.get('margin_required', 0)
                                add_to_blockchain('POSITION_CLOSED', {'position': pos, 'pnl': pnl})
                                st.success(t(f"Position closed! P&L: ₹{pnl:,.2f}", lang))
                                st.session_state.notifications.append(t(f"Position closed with P&L: ₹{pnl:,.0f}", lang))
                                st.rerun()
                        with action_col2:
                            if st.button(t("📝 Modify", lang), key=f"modify_{idx}", use_container_width=True):
                                st.info(t("Modify stop loss and take profit levels", lang))
                        with action_col3:
                            if st.button(t("📊 Details", lang), key=f"details_{idx}", use_container_width=True):
                                st.json(pos)
            else:
                st.info(t("📭 No open positions. Open a new position in the Virtual Hedging section.", lang))
        with tab2:
            st.subheader(t("📊 Performance Analytics", lang))
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
                perf_col1.metric(t("Total Trades", lang), total_trades)
                perf_col2.metric(t("Win Rate", lang), f"{win_rate:.1f}%")
                perf_col3.metric(t("Winning Trades", lang), winning_trades)
                perf_col4.metric(t("Total Realized P&L", lang), f"₹{total_pnl_closed:,.0f}")
                pnl_by_date = {}
                for p in closed_positions:
                    date = p.get('close_date', p['timestamp'])[:10]
                    pnl = (p.get('exit_price', p['entry_price']) - p['entry_price']) * p['quantity'] if p['type'] == "Long (Buy)" else (p['entry_price'] - p.get('exit_price', p['entry_price'])) * p['quantity']
                    pnl_by_date[date] = pnl_by_date.get(date, 0) + pnl
                if pnl_by_date:
                    pnl_df = pd.DataFrame(list(pnl_by_date.items()), columns=[t('Date', lang), t('P&L', lang)])
                    pnl_df[t('Cumulative P&L', lang)] = pnl_df[t('P&L', lang)].cumsum()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=pnl_df[t('Date', lang)], y=pnl_df[t('Cumulative P&L', lang)], mode='lines+markers', name=t('Cumulative P&L', lang), line=dict(color='green', width=2)))
                    fig.update_layout(title=t("Equity Curve", lang), xaxis_title=t("Date", lang), yaxis_title=t("Cumulative P&L (₹)", lang), height=400)
                    st.plotly_chart(fig, use_container_width=True)
                commodity_pnl = {}
                for p in closed_positions:
                    pnl = (p.get('exit_price', p['entry_price']) - p['entry_price']) * p['quantity'] if p['type'] == "Long (Buy)" else (p['entry_price'] - p.get('exit_price', p['entry_price'])) * p['quantity']
                    commodity_pnl[p['commodity']] = commodity_pnl.get(p['commodity'], 0) + pnl
                if commodity_pnl:
                    fig_bar = px.bar(x=list(commodity_pnl.keys()), y=list(commodity_pnl.values()), labels={'x': t('Commodity', lang), 'y': t('Total P&L (₹)', lang)}, title=t('P&L by Commodity', lang))
                    st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info(t("📊 No closed positions yet. Performance analytics will appear after closing positions.", lang))
        with tab3:
            st.subheader(t("📜 Transaction History", lang))
            user_transactions = export_transaction_history(st.session_state.username)
            if not user_transactions.empty:
                st.dataframe(user_transactions, use_container_width=True, hide_index=True)
                csv = user_transactions.to_csv(index=False)
                st.download_button(
                    label=t("📥 Download Transaction History", lang),
                    data=csv,
                    file_name=f"transactions_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info(t("📭 No transaction history available.", lang))
    elif t("🔗 Blockchain Ledger", lang) in menu:
        st.header(t("🔗 Blockchain Transaction Ledger", lang))
        st.write(t("All transactions are immutably recorded on the blockchain for complete transparency and security.", lang))
        ledger_col1, ledger_col2, ledger_col3, ledger_col4 = st.columns(4)
        ledger_col1.metric(t("Total Blocks", lang), len(st.session_state.blockchain))
        ledger_col2.metric(t("Total Users", lang), len(st.session_state.users))
        ledger_col3.metric(t("Total Contracts", lang), len(st.session_state.contracts))
        ledger_col4.metric(t("Blockchain Valid", lang), t("✅ Yes", lang) if verify_blockchain() else t("❌ No", lang))
        st.subheader(t("🔍 Recent Transactions", lang))
        filter_tx_type = st.multiselect(t("Filter by Type", lang), ['All'] + list(set([b['transaction_type'] for b in st.session_state.blockchain])), default=['All'])
        show_blocks = st.slider(t("Number of blocks to display", lang), 5, 50, 10)
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
            with st.expander(f"{emoji} {t('Block', lang)} #{block['index']} - {block['transaction_type']} - {block['timestamp'][:19]}"):
                block_col1, block_col2 = st.columns(2)
                with block_col1:
                    st.markdown(f"**{t('Block Index:', lang)}** {block['index']}")
                    st.markdown(f"**{t('Transaction Type:', lang)}** {block['transaction_type']}")
                    st.markdown(f"**{t('Timestamp:', lang)}** {block['timestamp']}")
                with block_col2:
                    st.markdown(f"**{t('Block Hash:', lang)}** `{block['hash'][:32]}...`")
                    st.markdown(f"**{t('Previous Hash:', lang)}** `{block['previous_hash'][:32]}...`")
                st.markdown(f"**{t('Transaction Data:', lang)}**")
                st.json(block['data'])
                if st.button(t(f"🔍 Verify Block #{block['index']}", lang), key=f"verify_{block['index']}"):
                    block_copy = {k: v for k, v in block.items() if k != 'hash'}
                    computed_hash = hash_block(block_copy)
                    if computed_hash == block['hash']:
                        st.success(t("✅ Block verification successful - Hash matches!", lang))
                    else:
                        st.error(t("❌ Block verification failed - Hash mismatch!", lang))
        st.divider()
        st.subheader(t("🔐 Blockchain Verification", lang))
        verify_col1, verify_col2 = st.columns(2)
        with verify_col1:
            if st.button(t("🔍 Verify Entire Blockchain", lang), use_container_width=True):
                with st.spinner(t("Verifying blockchain integrity...", lang)):
                    is_valid = verify_blockchain()
                    if is_valid:
                        st.success(t("✅ Blockchain integrity verified! All blocks are valid.", lang))
                    else:
                        st.error(t("❌ Blockchain integrity compromised! Tampering detected.", lang))
        with verify_col2:
            if st.button(t("📊 Blockchain Statistics", lang), use_container_width=True):
                st.info(t(f"Total Blocks: {len(st.session_state.blockchain)}", lang))
                st.info(t(f"Chain Length: {len(st.session_state.blockchain)}", lang))
                if st.session_state.blockchain:
                    st.info(t(f"Latest Block Hash: {st.session_state.blockchain[-1]['hash'][:32]}...", lang))
    elif t("📊 Analytics", lang) in menu:
        st.header(t("📊 Advanced Analytics & Insights", lang))
        tab1, tab2, tab3 = st.tabs([t("📈 Market Analysis", lang), t("🔥 Heat Maps", lang), t("📉 Volatility Analysis", lang)])
        with tab1:
            st.subheader(t("📈 Comprehensive Market Analysis", lang))
            analysis_commodity = st.selectbox(t("Select Commodity for Analysis", lang), list(st.session_state.market_data.keys()), key="analysis_commodity")
            commodity_data = st.session_state.price_data[st.session_state.price_data['commodity'] == analysis_commodity]
            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
            current_price = commodity_data['price'].iloc[-1]
            avg_price = commodity_data['price'].mean()
            min_price = commodity_data['price'].min()
            max_price = commodity_data['price'].max()
            price_volatility = commodity_data['price'].std()
            metric_col1.metric(t("Current", lang), f"₹{current_price:.2f}")
            metric_col2.metric(t("Average", lang), f"₹{avg_price:.2f}")
            metric_col3.metric(t("Min (6M)", lang), f"₹{min_price:.2f}")
            metric_col4.metric(t("Max (6M)", lang), f"₹{max_price:.2f}")
            metric_col5.metric(t("Volatility (σ)", lang), f"₹{price_volatility:.2f}")
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
                title=t(f"{analysis_commodity} - Weekly Candlestick Chart", lang),
                xaxis_title=t("Date", lang),
                yaxis_title=t("Price (₹/quintal)", lang),
                height=500,
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig_candlestick, use_container_width=True)
            st.subheader(t("📊 Volume Analysis", lang))
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=commodity_data['date'],
                y=commodity_data['volume'],
                name=t('Volume', lang),
                marker_color='lightblue'
            ))
            fig_volume.update_layout(
                title=t("Trading Volume Over Time", lang),
                xaxis_title=t("Date", lang),
                yaxis_title=t("Volume (quintals)", lang),
                height=300
            )
            st.plotly_chart(fig_volume, use_container_width=True)
            st.subheader(t("📉 Technical Indicators", lang))
            commodity_data_copy = commodity_data.copy()
            commodity_data_copy['SMA_20'] = commodity_data_copy['price'].rolling(20).mean()
            commodity_data_copy['SMA_50'] = commodity_data_copy['price'].rolling(50).mean()
            commodity_data_copy['EMA_12'] = commodity_data_copy['price'].ewm(span=12, adjust=False).mean()
            commodity_data_copy['EMA_26'] = commodity_data_copy['price'].ewm(span=26, adjust=False).mean()
            fig_technical = go.Figure()
            fig_technical.add_trace(go.Scatter(x=commodity_data_copy['date'], y=commodity_data_copy['price'], mode='lines', name=t('Price', lang), line=dict(color='blue')))
            fig_technical.add_trace(go.Scatter(x=commodity_data_copy['date'], y=commodity_data_copy['SMA_20'], mode='lines', name=t('SMA 20', lang), line=dict(color='orange', dash='dash')))
            fig_technical.add_trace(go.Scatter(x=commodity_data_copy['date'], y=commodity_data_copy['SMA_50'], mode='lines', name=t('SMA 50', lang), line=dict(color='red', dash='dash')))
            fig_technical.update_layout(
                title=t("Price with Moving Averages", lang),
                xaxis_title=t("Date", lang),
                yaxis_title=t("Price (₹/quintal)", lang),
                height=400
            )
            st.plotly_chart(fig_technical, use_container_width=True)
        with tab2:
            st.subheader(t("🔥 Market Heat Map", lang))
            heatmap_data = []
            for commodity, data in st.session_state.market_data.items():
                heatmap_data.append({
                    t('Commodity', lang): commodity,
                    t('Price', lang): data['price'],
                    t('Change %', lang): data['change'],
                    t('Volume', lang): data['volume'],
                    t('Volatility', lang): data['volatility']
                })
            heatmap_df = pd.DataFrame(heatmap_data)
            fig_heatmap = px.imshow(
                heatmap_df.set_index(t('Commodity', lang))[[t('Change %', lang), t('Volatility', lang)]].T,
                labels=dict(x=t("Commodity", lang), y=t("Metric", lang), color=t("Value", lang)),
                x=heatmap_df[t('Commodity', lang)],
                y=[t('Change %', lang), t('Volatility', lang)],
                color_continuous_scale='RdYlGn',
                aspect="auto"
            )
            fig_heatmap.update_layout(title=t("Market Performance Heat Map", lang), height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            st.subheader(t("📊 Correlation Matrix", lang))
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
                labels=dict(color=t("Correlation", lang)),
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                color_continuous_scale='RdBu',
                aspect="auto",
                zmin=-1,
                zmax=1
            )
            fig_corr.update_layout(title=t("Commodity Price Correlation Matrix", lang), height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
        with tab3:
            st.subheader(t("📉 Volatility Analysis", lang))
            vol_commodity = st.selectbox(t("Select Commodity", lang), list(st.session_state.market_data.keys()), key="vol_commodity")
            vol_window = st.slider(t("Rolling Window (days)", lang), 7, 60, 30)
            vol_data = st.session_state.price_data[st.session_state.price_data['commodity'] == vol_commodity].copy()
            vol_data['returns'] = vol_data['price'].pct_change()
            vol_data['volatility'] = vol_data['returns'].rolling(vol_window).std() * np.sqrt(252) * 100
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=vol_data['date'],
                y=vol_data['volatility'],
                mode='lines',
                name=t('Volatility', lang),
                fill='tonexty',
                line=dict(color='red')
            ))
            fig_vol.update_layout(
                title=t(f"{vol_commodity} - Rolling Volatility ({vol_window} days)", lang),
                xaxis_title=t("Date", lang),
                yaxis_title=t("Annualized Volatility (%)", lang),
                height=400
            )
            st.plotly_chart(fig_vol, use_container_width=True)
            current_vol = vol_data['volatility'].iloc[-1]
            avg_vol = vol_data['volatility'].mean()
            max_vol = vol_data['volatility'].max()
            min_vol = vol_data['volatility'].min()
            vol_metric_col1, vol_metric_col2, vol_metric_col3, vol_metric_col4 = st.columns(4)
            vol_metric_col1.metric(t("Current Volatility", lang), f"{current_vol:.2f}%")
            vol_metric_col2.metric(t("Average Volatility", lang), f"{avg_vol:.2f}%")
            vol_metric_col3.metric(t("Max Volatility", lang), f"{max_vol:.2f}%")
            vol_metric_col4.metric(t("Min Volatility", lang), f"{min_vol:.2f}%")
            if current_vol > avg_vol * 1.5:
                st.error(t("⚠️ **High Volatility Alert!** Current volatility is significantly above average. Exercise caution in trading.", lang))
            elif current_vol < avg_vol * 0.5:
                st.success(t("✅ **Low Volatility Period** - Stable market conditions. Good for conservative strategies.", lang))
            else:
                st.info(t("📊 **Normal Volatility Range** - Market operating within typical volatility levels.", lang))
    elif t("⚙️ Settings", lang) in menu:
        st.header(t("⚙️ Account Settings & Preferences", lang))
        tab1, tab2, tab3, tab4 = st.tabs([t("👤 Profile", lang), t("🔔 Notifications", lang), t("🎨 Preferences", lang), t("🔐 Security", lang)])
        with tab1:
            st.subheader(t("👤 Profile Information", lang))
            user_info = st.session_state.users[st.session_state.username]
            profile_col1, profile_col2 = st.columns(2)
            with profile_col1:
                st.text_input(t("Username", lang), value=st.session_state.username, disabled=True)
                new_location = st.text_input(t("Location", lang), value=user_info.get('location', ''))
                new_phone = st.text_input(t("Phone Number", lang), value=user_info.get('phone', ''))
            with profile_col2:
                st.text_input(t("Account Type", lang), value=user_info.get('type', ''), disabled=True)
                kyc_status = st.selectbox(t("KYC Status", lang), [t("Not Submitted", lang), t("Pending", lang), t("Verified", lang)], index=2)
                st.text_input(t("Member Since", lang), value=user_info.get('created_at', '')[:10] if 'created_at' in user_info else 'N/A', disabled=True)
            if st.button(t("💾 Update Profile", lang), use_container_width=True):
                user_info['location'] = new_location
                user_info['phone'] = new_phone
                st.success(t("✅ Profile updated successfully!", lang))
                st.rerun()
        with tab2:
            st.subheader(t("🔔 Notification Preferences", lang))
            enable_price_alerts = st.checkbox(t("Enable Price Movement Alerts", lang), value=True)
            price_threshold = st.slider(t("Alert Threshold (%)", lang), 1.0, 10.0, 3.0, 0.5)
            enable_contract_alerts = st.checkbox(t("Contract Status Notifications", lang), value=True)
            enable_position_alerts = st.checkbox(t("Position P&L Alerts", lang), value=True)
            enable_email = st.checkbox(t("Email Notifications", lang), value=False)
            enable_sms = st.checkbox(t("SMS Notifications", lang), value=False)
            if st.button(t("💾 Save Notification Settings", lang), use_container_width=True):
                st.success(t("✅ Notification preferences saved!", lang))
        with tab3:
            st.subheader(t("🎨 Display Preferences", lang))
            theme = st.selectbox(t("Theme", lang), [t("Light", lang), t("Dark", lang), t("Auto", lang)])
            language_pref = st.selectbox(t("Language", lang), ["English", "हिंदी", "मराठी", "ગુજરાતી"])
            currency = st.selectbox(t("Currency Display", lang), ["INR (₹)", "USD ($)"])
            date_format = st.selectbox(t("Date Format", lang), ["DD/MM/YYYY", "MM/DD/YYYY", "YYYY-MM-DD"])
            chart_type = st.selectbox(t("Default Chart Type", lang), [t("Line", lang), t("Candlestick", lang), t("Area", lang)])
            if st.button(t("💾 Save Preferences", lang), use_container_width=True):
                st.success(t("✅ Display preferences saved!", lang))
        with tab4:
            st.subheader(t("🔐 Security Settings", lang))
            st.markdown(f"#### {t('Change Password', lang)}")
            current_password = st.text_input(t("Current Password", lang), type="password", key="current_pass")
            new_password = st.text_input(t("New Password", lang), type="password", key="new_pass")
            confirm_new_password = st.text_input(t("Confirm New Password", lang), type="password", key="confirm_new_pass")
            if st.button(t("🔑 Change Password", lang), use_container_width=True):
                if new_password == confirm_new_password and len(new_password) >= 6:
                    hashed_pass = hashlib.sha256(new_password.encode()).hexdigest()
                    user_info['password'] = hashed_pass
                    st.success(t("✅ Password changed successfully!", lang))
                else:
                    st.error(t("❌ Passwords don't match or too short!", lang))
            st.divider()
            st.markdown(f"#### {t('Two-Factor Authentication', lang)}")
            enable_2fa = st.checkbox(t("Enable 2FA", lang), value=False)
            if enable_2fa:
                st.info(t("📱 Scan QR code with authenticator app", lang))
                st.text_input(t("Enter 2FA Code", lang))
            st.divider()
            st.markdown(f"#### {t('Active Sessions', lang)}")
            st.markdown(f"**{t('Current Session:', lang)}** Desktop - Sangamner, Maharashtra")
            st.markdown(f"**{t('Last Login:', lang)}** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            if st.button(t("🚪 Logout All Devices", lang), use_container_width=True):
                st.warning(t("All sessions will be terminated", lang))
st.markdown("---")
st.caption(t("© 2024 GrainShield Pro | Powered by AI & Blockchain | Contact: support@grainshield.in | Version 1.0.0", lang))
