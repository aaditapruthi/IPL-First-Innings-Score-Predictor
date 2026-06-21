import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="IPL Score Predictor",
    layout="wide"
)

page_bg = """
<style>

[data-testid="stAppViewContainer"]{
background-image:url("https://i.pinimg.com/1200x/fe/7b/8b/fe7b8bcdd9900171085aa238c92bfc04.jpg");
background-size:cover;
background-position:center;
}

</style>
"""

st.markdown(page_bg,unsafe_allow_html=True)

st.markdown("""
<style>

h1{
    text-align:center;
    color:#FFD700;
}

.stButton > button {
    background-color: #FF4B4B;
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
    height: 60px;
    width: 100%;
}

</style>
""",unsafe_allow_html=True)


st.sidebar.title(
    "🏏 IPL Predictor"
)

left, center, right = st.columns([2,1,2])

with center:
    st.image(
        "assets/ipl_logo.jpg",
        width=250
    )

st.sidebar.markdown("""
### About

This project predicts IPL First Innings Scores using:

- Linear Regression
- Lasso Regression
- Decision Tree
- Random Forest
- SVR
- Neural Network

""")

st.markdown("""
<style>

h1{
text-align:center;
color:#FFD700;
}

.stButton>button{
    background-color:#FF4B4B;
    color:white;
    font-size:20px;
    border-radius:10px;
    height:60px;
}

</style>
""",
unsafe_allow_html=True)

model = pickle.load(
    open("best_model.pkl", "rb")
)

st.title(" IPL First Innings Score Predictor")

col1, col2 = st.columns(2)

with col1:

    venue = st.selectbox(
        " Venue",
        [
            "M Chinnaswamy Stadium",
            "Eden Gardens",
            "Feroz Shah Kotla",
            "MA Chidambaram Stadium",
            "Wankhede Stadium"
        ]
    )

    batting_team = st.selectbox(
        " Batting Team",
        [
            "Mumbai Indians",
            "Chennai Super Kings",
            "Royal Challengers Bangalore",
            "Kolkata Knight Riders",
            "Delhi Daredevils",
            "Kings XI Punjab",
            "Rajasthan Royals",
            "Sunrisers Hyderabad"
        ]
    )

    runs = st.number_input(
        "Current Runs",
        min_value=0,
        step=1
    )

    wickets = st.slider(
        "Wickets Fallen",
        min_value=0,
        max_value=10,
        value=0
    )

with col2:

    bowling_team = st.selectbox(
        " Bowling Team",
        [
            "Mumbai Indians",
            "Chennai Super Kings",
            "Royal Challengers Bangalore",
            "Kolkata Knight Riders",
            "Delhi Daredevils",
            "Kings XI Punjab",
            "Rajasthan Royals",
            "Sunrisers Hyderabad"
        ]
    )

    overs = st.slider(
        "Overs Completed",
        min_value=5.0,
        max_value=20.0,
        value=5.0,
        step=0.1
    )

    runs_last_5 = st.number_input(
        "Runs in Last 5 Overs",
        min_value=0,
        step=1
    )

    wickets_last_5 = st.slider(
        "Wickets in Last 5 Overs",
        min_value=0,
        max_value=5,
        value=0
    )

col3, col4 = st.columns(2)

with col3:

    striker = st.number_input(
        "Striker Runs",
        min_value=0,
        step=1
    )

with col4:

    non_striker = st.number_input(
        "Non-Striker Runs",
        min_value=0,
        step=1
    )

st.subheader(" Match Progress")

progress = int((overs / 20) * 100)

st.progress(progress)

st.write(f"Match Completion: {progress}%")

st.subheader("Current Match Situation")

st.info(f"""
Venue: {venue}

Batting Team: {batting_team}

Bowling Team: {bowling_team}

Score: {runs}/{wickets}

Overs: {overs}
""")

col1, col2, col3 = st.columns([3, 1, 3])

with col2:
    predict_button = st.button("🏏 Predict Score")

if predict_button:

    data = pd.DataFrame(
        {
            'venue':[venue],
            'batting_team':[batting_team],
            'bowling_team':[bowling_team],
            'runs':[runs],
            'wickets':[wickets],
            'overs':[overs],
            'runs_last_5':[runs_last_5],
            'wickets_last_5':[wickets_last_5],
            'striker':[striker],
            'non-striker':[non_striker]
        }
    )

    with st.spinner("Analyzing match situation..."):

        prediction = model.predict(data)

    score = round(prediction[0])

    lower = score - 5
    upper = score + 5

    st.success(
        f"🏏 Predicted Final Score: {score}"
    )

    st.metric(
        "Expected Range",
        f"{lower} - {upper}"
    )

    if score < 140:

        st.warning(
            " Low Scoring Total"
        )

    elif score < 180:

        st.info(
            " Competitive Total"
        )

    else:

        st.success(
            " Excellent Batting Total"
        )

st.divider()

st.subheader(" Model Performance Comparison")

results_df = pd.read_csv("model_results.csv")

results_df["MAE"] = results_df["MAE"].round(2)
results_df["RMSE"] = results_df["RMSE"].round(2)
results_df["R2"] = results_df["R2"].round(4)

best_model = results_df.loc[
    results_df["R2"].idxmax()
]

st.success(
    f"🏆 Best Model: {best_model['Model']} "
    f"(R² = {best_model['R2']:.4f})"
)

st.bar_chart(
    results_df.set_index("Model")["R2"]
)

st.subheader(" Detailed Model Metrics")

st.dataframe(
    results_df,
    width='stretch',
    hide_index=True
)