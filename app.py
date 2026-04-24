import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration
st.set_page_config(page_title="AI Sales Dashboard", layout="wide")

# 2. Load Data and AI Assets
@st.cache_resource
def load_assets():
    # model_columns comes from your columns.pkl (X.columns)
    model = pickle.load(open('model.pkl', 'rb'))
    model_columns = pickle.load(open('columns.pkl', 'rb'))
    return model, model_columns

@st.cache_data
def load_data():
    return pd.read_csv('data.csv')

try:
    model, model_columns = load_assets()
    df = load_data()
except Exception as e:
    st.error("Missing files! Please ensure 'model.pkl', 'columns.pkl', and 'data.csv' are in your GitHub repo.")
    st.stop()

# 3. Sidebar Navigation
st.sidebar.title("🤖 AI Sales System")
page = st.sidebar.radio("Navigation", ["Dashboard", "Analytics", "AI Predictor"])

# --- PAGE 1: DASHBOARD ---
if page == "Dashboard":
    st.title("📊 Executive Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Avg Price", f"${df['Price_USD'].mean():,.0f}")
    col3.metric("Models Tracked", df['Model'].nunique())
    
    st.subheader("Data Overview")
    st.dataframe(df.head(10), use_container_width=True)

# --- PAGE 2: ANALYTICS ---
elif page == "Analytics":
    st.title("📈 Market Insights")
    c1, c2 = st.columns(2)
    with c1:
        st.write("#### Sales Performance")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Sales_Classification', palette='viridis', ax=ax)
        st.pyplot(fig)
    with c2:
        st.write("#### Price vs Mileage Distribution")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='Mileage_KM', y='Price_USD', hue='Sales_Classification', ax=ax)
        st.pyplot(fig)

# --- PAGE 3: AI PREDICTOR (With Improved Logic) ---
elif page == "AI Predictor":
    st.title("🔮 AI Sales Forecaster")
    st.info("Input specifications to calculate the probability of sales demand.")

    with st.form("prediction_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            year = st.selectbox("Year", sorted(df['Year'].unique(), reverse=True))
            engine = st.slider("Engine Size (L)", 1.0, 6.0, 2.5)
            mileage = st.number_input("Mileage (KM)", 0, 300000, 50000)
            price = st.number_input("Price (USD)", 5000, 200000, 45000)
        with col_b:
            v_model = st.selectbox("Model", sorted(df['Model'].unique()))
            region = st.selectbox("Region", sorted(df['Region'].unique()))
            fuel = st.selectbox("Fuel Type", sorted(df['Fuel_Type'].unique()))
            trans = st.selectbox("Transmission", sorted(df['Transmission'].unique()))
            color = st.selectbox("Color", sorted(df['Color'].unique()))
        
        submit = st.form_submit_button("Run AI Prediction")

    if submit:
        # --- DATA PREPARATION ---
        input_df = pd.DataFrame([{
            'Year': year, 'Engine_Size_L': engine, 'Mileage_KM': mileage,
            'Price_USD': price, 'Model': v_model, 'Region': region,
            'Fuel_Type': fuel, 'Transmission': trans, 'Color': color
        }])
        
        # 1. Encoding (Create dummies for the single user input)
        input_encoded = pd.get_dummies(input_df)
        
        # 2. Alignment (Add missing columns from training and set to 0)
        for col in model_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # 3. Final Ordering (Exact order model expects)
        input_final = input_encoded[model_columns]

        # --- PREDICTION & VISUALIZATION ---
        prediction = model.predict(input_final)[0]
        probabilities = model.predict_proba(input_final)[0] # [Prob_Low, Prob_High]
        
        result = "High" if prediction == 1 else "Low"
        confidence = max(probabilities) * 100
        
        st.divider()
        st.subheader(f"AI Result: {result} Demand")

        # Probability Bar Chart
        prob_df = pd.DataFrame({
            'Category': ['Low Demand', 'High Demand'],
            'Confidence (%)': [probabilities[0] * 100, probabilities[1] * 100]
        })
        
        fig_prob, ax_prob = plt.subplots(figsize=(10, 2.5))
        sns.barplot(data=prob_df, x='Confidence (%)', y='Category', palette=['#ff4b4b', '#29b09d'], ax=ax_prob)
        ax_prob.set_xlim(0, 100)
        st.pyplot(fig_prob)
        
        st.write(f"The AI is **{confidence:.1f}%** confident in this prediction.")
        
        if result == "High":
            st.balloons()
            st.success("This vehicle configuration is highly likely to sell!")
        else:
            st.warning("This vehicle configuration shows a lower market demand probability.")