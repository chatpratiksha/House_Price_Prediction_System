import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.title {
    font-size: 45px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 10px;
}

.subtitle {
    font-size: 20px;
    text-align: center;
    color: #666666;
    margin-bottom: 30px;
}

.card {
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.price {
    font-size: 38px;
    font-weight: bold;
    text-align: center;
}

.section-title {
    font-size: 30px;
    font-weight: 600;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# FILE PATHS
# =========================================================

MODEL_PATH = "house_price_model.pkl"
DATASET_PATH = "house_price_dataset.csv"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


model = load_model()


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():

    if not os.path.exists(DATASET_PATH):
        return None

    return pd.read_csv(DATASET_PATH)


df = load_dataset()


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"


# =========================================================
# PAGE NAVIGATION
# =========================================================

pages = [
    "Home",
    "About Project",
    "Predict Price",
    "Model Performance",
    "About Developer"
]


# Sidebar

st.sidebar.title("🏠 House Price App")

st.sidebar.markdown("---")

selected_page = st.sidebar.radio(
    "Navigate",
    pages,
    index=pages.index(st.session_state.page)
)

st.session_state.page = selected_page

st.sidebar.markdown("---")

st.sidebar.info(
    "AI/ML Based House Price Prediction System"
)


# =========================================================
# PAGE 1 — HOME
# =========================================================

if st.session_state.page == "Home":

    st.markdown(
        '<div class="title">🏠 House Price Prediction System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Predict house prices using Artificial Intelligence and Machine Learning'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

        <h2>Welcome 👋</h2>

        <p>
        Welcome to the House Price Prediction System.
        This application uses Machine Learning to estimate
        the price of a house based on its characteristics.
        </p>

        <p>
        Enter details such as area, bedrooms, bathrooms,
        parking, age and location to get an estimated price.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### ✨ What can you do?")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="card">

            <h3>📖 Learn</h3>

            <p>
            Understand how the House Price Prediction
            system works.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="card">

            <h3>🤖 Predict</h3>

            <p>
            Enter house details and get an estimated
            property price.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="card">

            <h3>📊 Analyze</h3>

            <p>
            View the performance of the Machine Learning
            model.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("")

    if st.button("🚀 Get Started", use_container_width=True):

        st.session_state.page = "About Project"
        st.rerun()


# =========================================================
# PAGE 2 — ABOUT PROJECT
# =========================================================

elif st.session_state.page == "About Project":

    st.markdown(
        '<div class="section-title">📖 About Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

        <h2>🏠 House Price Prediction System</h2>

        <p>
        House Price Prediction is a Machine Learning based
        system that predicts the estimated price of a house
        using different property features.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("🎯 Project Objectives")

    objectives = [
        "Predict house prices using Machine Learning.",
        "Provide an easy-to-use user interface.",
        "Reduce manual house price estimation.",
        "Use historical house data for prediction.",
        "Demonstrate practical use of Artificial Intelligence."
    ]

    for objective in objectives:
        st.write("✅", objective)

    st.subheader("🛠️ Technologies Used")

    technologies = pd.DataFrame({
        "Technology": [
            "Python",
            "Streamlit",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "Random Forest"
        ],
        "Purpose": [
            "Programming",
            "Front End / Web App",
            "Data Processing",
            "Numerical Processing",
            "Machine Learning",
            "Price Prediction"
        ]
    })

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        if st.button("← Home", use_container_width=True):

            st.session_state.page = "Home"
            st.rerun()

    with col2:

        if st.button("Start Prediction →", use_container_width=True):

            st.session_state.page = "Predict Price"
            st.rerun()


# =========================================================
# PAGE 3 — PREDICT PRICE
# =========================================================

elif st.session_state.page == "Predict Price":

    st.markdown(
        '<div class="section-title">🤖 Predict House Price</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the details of the house below."
    )

    if model is None:

        st.error(
            "Model file not found. Please check "
            "model/house_price_model.pkl"
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            area = st.number_input(
                "🏠 Area (sq.ft)",
                min_value=100,
                max_value=10000,
                value=1200,
                step=50
            )

            bedrooms = st.number_input(
                "🛏️ Bedrooms",
                min_value=1,
                max_value=10,
                value=2,
                step=1
            )

            bathrooms = st.number_input(
                "🚿 Bathrooms",
                min_value=1,
                max_value=10,
                value=2,
                step=1
            )

        with col2:

            parking = st.number_input(
                "🚗 Parking Spaces",
                min_value=0,
                max_value=10,
                value=1,
                step=1
            )

            age = st.number_input(
                "📅 House Age (years)",
                min_value=0,
                max_value=100,
                value=5,
                step=1
            )

            if df is not None and "Location" in df.columns:

                locations = sorted(
                    df["Location"].dropna().unique().tolist()
                )

            else:

                locations = ["City Center"]

            location = st.selectbox(
                "📍 Location",
                locations
            )

        st.markdown("---")

        if st.button(
            "🔮 Predict House Price",
            use_container_width=True
        ):

            input_data = pd.DataFrame({
                "Area": [area],
                "Bedrooms": [bedrooms],
                "Bathrooms": [bathrooms],
                "Parking": [parking],
                "Age": [age],
                "Location": [location]
            })

            try:

                prediction = model.predict(input_data)[0]

                st.session_state.predicted_price = prediction

                st.success("Prediction completed successfully! 🎉")

                st.markdown(
                    f"""
                    <div class="card">

                    <h2 style="text-align:center;">
                    🏠 Estimated House Price
                    </h2>

                    <div class="price">
                    ₹{prediction:,.0f}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.subheader("🏠 Property Summary")

                summary = pd.DataFrame({
                    "Feature": [
                        "Area",
                        "Bedrooms",
                        "Bathrooms",
                        "Parking",
                        "Age",
                        "Location"
                    ],
                    "Value": [
                        f"{area} sq.ft",
                        bedrooms,
                        bathrooms,
                        parking,
                        f"{age} years",
                        location
                    ]
                })

                st.dataframe(
                    summary,
                    use_container_width=True,
                    hide_index=True
                )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        if st.button("← About Project", use_container_width=True):

            st.session_state.page = "About Project"
            st.rerun()

    with col2:

        if st.button("View Model Performance →", use_container_width=True):

            st.session_state.page = "Model Performance"
            st.rerun()


# =========================================================
# PAGE 4 — MODEL PERFORMANCE
# =========================================================

elif st.session_state.page == "Model Performance":

    st.markdown(
        '<div class="section-title">📊 Model Performance</div>',
        unsafe_allow_html=True
    )

    if model is None or df is None:

        st.error(
            "Model or dataset could not be found."
        )

    else:

        features = [
            "Area",
            "Bedrooms",
            "Bathrooms",
            "Parking",
            "Age",
            "Location"
        ]

        target = "Price"

        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        mse = mean_squared_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test,
            y_pred
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "MAE",
                f"₹{mae:,.0f}"
            )

        with col2:

            st.metric(
                "MSE",
                f"{mse:,.0f}"
            )

        with col3:

            st.metric(
                "RMSE",
                f"₹{rmse:,.0f}"
            )

        with col4:

            st.metric(
                "R² Score",
                f"{r2:.4f}"
            )

        st.markdown("---")

        st.subheader("🌳 Algorithm Used")

        st.info(
            "Random Forest Regressor"
        )

        st.subheader("📈 Actual vs Predicted Price")

        chart_data = pd.DataFrame({
            "Actual Price": y_test.values,
            "Predicted Price": y_pred
        })

        st.scatter_chart(
            chart_data,
            x="Actual Price",
            y="Predicted Price"
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        if st.button("← Predict Price", use_container_width=True):

            st.session_state.page = "Predict Price"
            st.rerun()

    with col2:

        if st.button("About Developer →", use_container_width=True):

            st.session_state.page = "About Developer"
            st.rerun()


# =========================================================
# PAGE 5 — ABOUT DEVELOPER
# =========================================================

elif st.session_state.page == "About Developer":

    st.markdown(
        '<div class="section-title">👩‍💻 About Developer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

        <h2>🏠 House Price Prediction System</h2>

        <p>
        This project demonstrates the application of
        Artificial Intelligence and Machine Learning
        for predicting house prices.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("👩‍💻 Project Information")

    project_info = pd.DataFrame({
        "Details": [
            "Project Name",
            "Domain",
            "Programming Language",
            "Machine Learning Algorithm",
            "Front End",
            "Dataset"
        ],
        "Information": [
            "House Price Prediction System",
            "Artificial Intelligence / Machine Learning",
            "Python",
            "Random Forest Regressor",
            "Streamlit",
            "House Price Dataset"
        ]
    })

    st.dataframe(
        project_info,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        if st.button("← Model Performance", use_container_width=True):

            st.session_state.page = "Model Performance"
            st.rerun()

    with col2:

        if st.button("🏠 Go to Home", use_container_width=True):

            st.session_state.page = "Home"
            st.rerun()
