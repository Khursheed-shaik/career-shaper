
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "model_artifacts"

APP_DATA_PATH = DATA_DIR / "career_shaper_app_data.csv"
RELATED_DATA_PATH = DATA_DIR / "related_occupations_cleaned.csv"
MODEL_PATH = MODEL_DIR / "critical_thinking_model.pkl"
FEATURES_PATH = MODEL_DIR / "model_features.json"

st.set_page_config(
    page_title="Career Shaper",
    page_icon="🎯",
    layout="wide",
)


# ============================================================
# 2. LOAD DATA AND MODEL
# ============================================================

@st.cache_data
def load_occupation_data():
    data = pd.read_csv(APP_DATA_PATH)
    related = pd.read_csv(RELATED_DATA_PATH)
    return data, related


@st.cache_resource
def load_model():
    trained_model = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return trained_model, metadata


try:
    occupations, related_data = load_occupation_data()
    model, model_metadata = load_model()

except FileNotFoundError as error:
    st.error(f"Required file not found: {error}")
    st.info(
        "Check that the CSV files are inside data/processed/ "
        "and the model files are inside model_artifacts/."
    )
    st.stop()

except Exception as error:
    st.error(f"Could not load project files: {error}")
    st.stop()


# ============================================================
# 3. SKILL FEATURES AND OCCUPATION PROFILES
# ============================================================

skill_columns = [
    "Active Learning",
    "Active Listening",
    "Critical Thinking",
    "Learning Strategies",
    "Mathematics",
    "Monitoring",
    "Reading Comprehension",
    "Science",
    "Speaking",
    "Writing",
]

missing_skills = [
    skill for skill in skill_columns
    if skill not in occupations.columns
]

if missing_skills:
    st.error(f"Missing skill columns in dataset: {missing_skills}")
    st.stop()

profile_data = occupations[
    ["occupation_code", "occupation_title"] + skill_columns
].copy()

profile_data[skill_columns] = profile_data[skill_columns].apply(
    pd.to_numeric, errors="coerce"
)

profile_data = profile_data.dropna(subset=skill_columns).copy()

if profile_data.empty:
    st.error("No complete occupation skill profiles were found.")
    st.stop()

scaler = MinMaxScaler()

occupation_matrix = scaler.fit_transform(
    profile_data[skill_columns]
)


# ============================================================
# 4. RECOMMENDATION FUNCTIONS
# ============================================================

def recommend_careers(user_ratings, top_n=10):
    user_profile = pd.DataFrame(
        [[user_ratings[skill] for skill in skill_columns]],
        columns=skill_columns,
    )

    user_scaled = scaler.transform(user_profile)

    scores = cosine_similarity(
        user_scaled,
        occupation_matrix,
    ).ravel()

    results = profile_data[
        ["occupation_code", "occupation_title"]
    ].copy()

    results["similarity_score"] = scores
    results["similarity_percent"] = (scores * 100).round(2)

    results = results.sort_values(
        "similarity_score",
        ascending=False,
    ).head(top_n)

    return results.reset_index(drop=True)


def skill_gap_analysis(user_ratings, occupation_title):
    matches = profile_data[
        profile_data["occupation_title"] == occupation_title
    ]

    if matches.empty:
        return pd.DataFrame()

    occupation_profile = matches.iloc[0]
    rows = []

    for skill in skill_columns:
        user_rating = float(user_ratings[skill])
        importance = float(occupation_profile[skill])

        rows.append({
            "Skill": skill,
            "Your self-rating": user_rating,
            "Occupation importance": round(importance, 2),
            "Difference": round(importance - user_rating, 2),
        })

    result = pd.DataFrame(rows)

    return result.sort_values(
        "Difference",
        ascending=False,
    ).reset_index(drop=True)


def get_related_careers(occupation_code, limit=5):
    matches = related_data[
        (
            related_data["occupation_code"].astype(str)
            == str(occupation_code)
        )
        & related_data["Relatedness Tier"].isin(
            ["Primary-Short", "Primary-Long"]
        )
    ].copy()

    available_titles = set(
        profile_data["occupation_title"].astype(str)
    )

    matches = matches[
        matches["related_occupation_title"].isin(available_titles)
    ]

    if "Index" in matches.columns:
        matches = matches.sort_values("Index")

    return (
        matches[
            ["related_occupation_title", "Relatedness Tier"]
        ]
        .drop_duplicates()
        .head(limit)
        .reset_index(drop=True)
    )


# ============================================================
# 5. MODEL INPUT VALIDATION
# ============================================================

model_features = model_metadata.get("features", [])

if not model_features:
    st.error("No model features were found in model_features.json.")
    st.stop()

missing_model_features = [
    feature for feature in model_features
    if feature not in skill_columns
]

if missing_model_features:
    st.error(
        "The model feature list does not match the application skills: "
        f"{missing_model_features}"
    )
    st.stop()

# The model predicts Critical Thinking importance from the other skills.
if "Critical Thinking" in model_features:
    st.error(
        "Critical Thinking must not be an input feature because it is "
        "the model's prediction target."
    )
    st.stop()


# ============================================================
# 6. SHAP AND LIME EXPLANATION FUNCTIONS
# ============================================================

@st.cache_resource
def get_shap_explainer():
    return shap.TreeExplainer(model)


@st.cache_resource
def get_lime_explainer():
    # Use occupational ratings, ordered exactly as the model features.
    reference_data = profile_data[model_features].to_numpy(dtype=float)

    return LimeTabularExplainer(
        training_data=reference_data,
        feature_names=model_features,
        mode="regression",
        discretize_continuous=True,
        random_state=42,
    )


def explain_with_shap(prediction_input):
    explainer = get_shap_explainer()

    values = explainer.shap_values(prediction_input)

    # For a single-output regression model, SHAP returns one
    # contribution per feature.
    values = np.asarray(values)

    if values.ndim == 2:
        values = values[0]

    values = values.reshape(-1)

    explanation_df = pd.DataFrame({
        "Feature": model_features,
        "SHAP contribution": values,
    })

    explanation_df["Absolute contribution"] = (
        explanation_df["SHAP contribution"].abs()
    )

    explanation_df = explanation_df.sort_values(
        "Absolute contribution",
        ascending=False,
    )

    return explanation_df


def explain_with_lime(prediction_input):
    explainer = get_lime_explainer()

    def predict_named_features(values):
        values_df = pd.DataFrame(
            values,
            columns=model_features,
        )
        return model.predict(values_df)

    explanation = explainer.explain_instance(
        data_row=prediction_input.iloc[0].to_numpy(dtype=float),
        predict_fn=predict_named_features,
        num_features=len(model_features),
    )

    explanation_df = pd.DataFrame(
        explanation.as_list(),
        columns=["Feature condition", "Local contribution"],
    )

    return explanation, explanation_df


# ============================================================
# 7. APPLICATION INTERFACE
# ============================================================

st.title("🎯 Career Shaper")

st.subheader(
    "Data-Driven Career Path Recommendation Analysis"
)

st.write(
    "Explore occupations using your self-rated skills, compare "
    "skill importance, discover related career options, and examine "
    "a supporting machine-learning model."
)

st.warning(
    "Career recommendations are exploratory matches, not predictions "
    "of hiring, salary, job performance, or employment success. "
    "Your skill ratings are self-reported, while occupational ratings "
    "describe general occupational requirements."
)


# ============================================================
# 8. SKILL INPUT FORM
# ============================================================

st.header("1. Rate your skills")

st.write(
    "Rate each skill from 1 (beginner) to 5 (very strong). "
    "Use your own assessment rather than trying to obtain a particular "
    "recommendation."
)

with st.form("career_skill_form"):
    left, right = st.columns(2)

    user_ratings = {}

    for index, skill in enumerate(skill_columns):
        target_column = left if index % 2 == 0 else right

        with target_column:
            user_ratings[skill] = st.slider(
                skill,
                min_value=1,
                max_value=5,
                value=3,
                key=f"rating_{skill}",
            )

    submitted = st.form_submit_button(
        "Find matching careers",
        type="primary",
    )

if submitted:
    st.session_state["user_ratings"] = user_ratings

    st.session_state["recommendations"] = recommend_careers(
        user_ratings,
        top_n=10,
    )


# ============================================================
# 9. DISPLAY RECOMMENDATIONS AND CAREER DETAILS
# ============================================================

if "recommendations" not in st.session_state:
    st.info(
        "Set your skill ratings and click "
        "'Find matching careers' to begin."
    )

else:
    user_ratings = st.session_state["user_ratings"]
    recommendations = st.session_state["recommendations"]

    st.header("2. Recommended careers")

    st.caption(
        "Similarity scores compare skill profiles. They are not "
        "probabilities of being hired or succeeding in a role."
    )

    display_recommendations = recommendations[
        ["occupation_title", "similarity_percent"]
    ].rename(columns={
        "occupation_title": "Career",
        "similarity_percent": "Similarity score (%)",
    })

    st.dataframe(
        display_recommendations,
        use_container_width=True,
        hide_index=True,
    )

    selected_career = st.selectbox(
        "Choose a career to explore",
        recommendations["occupation_title"].tolist(),
        key="selected_career",
    )

    selected_row = recommendations[
        recommendations["occupation_title"] == selected_career
    ].iloc[0]

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric("Selected career", selected_career)

    with metric_col2:
        st.metric(
            "Skill-profile similarity",
            f"{selected_row['similarity_percent']:.2f}%",
        )

    st.caption(
        f"Occupation code: {selected_row['occupation_code']}"
    )


    # --------------------------------------------------------
    # Skill-gap analysis
    # --------------------------------------------------------

    st.header("3. Skill-gap comparison")

    gaps = skill_gap_analysis(
        user_ratings,
        selected_career,
    )

    st.dataframe(
        gaps,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "A positive difference means the occupation's importance rating "
        "is higher than your self-rating. This comparison is for "
        "reflection; it does not prove that you lack a skill."
    )

    positive_gaps = gaps[gaps["Difference"] > 0]

    if not positive_gaps.empty:
        st.write("Skills to explore for development:")

        st.write(
            ", ".join(positive_gaps["Skill"].head(5).tolist())
        )
    else:
        st.info(
            "No positive skill-importance differences were found "
            "for this profile."
        )


    # --------------------------------------------------------
    # Related careers
    # --------------------------------------------------------

    st.header("4. Related career options")

    related = get_related_careers(
        selected_row["occupation_code"],
        limit=5,
    )

    if related.empty:
        st.info(
            "No matching related occupations are available in "
            "the current dataset."
        )
    else:
        related_display = related.rename(columns={
            "related_occupation_title": "Related career",
            "Relatedness Tier": "Relationship category",
        })

        st.dataframe(
            related_display,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Relationship categories describe occupational relationships; "
        "they are not guaranteed promotion paths."
    )


    # --------------------------------------------------------
    # Education information
    # --------------------------------------------------------

    st.header("5. Education information")

    selected_occupation = occupations[
        occupations["occupation_title"] == selected_career
    ]

    education_available = False

    if (
        not selected_occupation.empty
        and "education_data_available" in occupations.columns
    ):
        education_available = bool(
            selected_occupation.iloc[0]["education_data_available"]
        )

    if education_available:
        education_columns = [
            col for col in occupations.columns
            if col.startswith("education_category_")
            and col.endswith("_percent")
        ]

        education_row = selected_occupation.iloc[0]

        education_values = pd.DataFrame({
            "Education category": [
                col.replace(
                    "education_category_", ""
                ).replace("_percent", "")
                for col in education_columns
            ],
            "Reported percentage": [
                pd.to_numeric(
                    pd.Series([education_row[col]]),
                    errors="coerce",
                ).iloc[0]
                for col in education_columns
            ],
        })

        education_values = education_values.dropna()

        education_values = education_values.sort_values(
            "Reported percentage",
            ascending=False,
        )

        st.dataframe(
            education_values,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Category numbers are displayed as recorded in the source "
            "dataset. They are not translated into degree names here."
        )

    else:
        st.info(
            "Education distribution is unavailable for this occupation."
        )


    # --------------------------------------------------------
    # Supporting ML prediction
    # --------------------------------------------------------

    st.header("6. Supporting ML analysis")

    st.write(
        "The Random Forest model estimates an occupational Critical "
        "Thinking importance rating using the other nine skill ratings. "
        "It does not estimate your personal ability or job success."
    )

    prediction_input = pd.DataFrame(
        [[user_ratings[feature] for feature in model_features]],
        columns=model_features,
    )

    predicted_importance = float(
        model.predict(prediction_input)[0]
    )

    st.metric(
        "Estimated Critical Thinking importance",
        f"{predicted_importance:.2f} / 5",
    )

    st.caption(
        "This is a model estimate based on the entered ratings, not an "
        "official occupational requirement or a personal skills test."
    )


    # --------------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------------

    with st.expander("Explain this prediction with SHAP"):
        try:
            shap_df = explain_with_shap(prediction_input)

            st.write(
                "SHAP contribution values show how each input feature "
                "moves this prediction relative to the model baseline."
            )

            st.dataframe(
                shap_df[
                    ["Feature", "SHAP contribution"]
                ],
                use_container_width=True,
                hide_index=True,
            )

            plot_data = shap_df.sort_values(
                "SHAP contribution",
                ascending=True,
            )

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.barh(
                plot_data["Feature"],
                plot_data["SHAP contribution"],
            )

            ax.axvline(0, linewidth=0.8)
            ax.set_xlabel("SHAP contribution")
            ax.set_title(
                "Feature contributions to the predicted rating"
            )

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as error:
            st.error(f"Could not generate the SHAP explanation: {error}")


    # --------------------------------------------------------
    # LIME explanation
    # --------------------------------------------------------

    with st.expander("Explain this prediction with LIME"):
        try:
            lime_explanation, lime_df = explain_with_lime(
                prediction_input
            )

            st.write(
                "LIME approximates the model near this particular input. "
                "Its feature contributions are local explanations."
            )

            st.dataframe(
                lime_df,
                use_container_width=True,
                hide_index=True,
            )

            fig = lime_explanation.as_pyplot_figure()
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        except Exception as error:
            st.error(f"Could not generate the LIME explanation: {error}")


    st.success(
        "Career recommendations, skill-gap comparison, and supporting "
        "ML analysis are ready."
    )


# ============================================================
# 10. FOOTER
# ============================================================

st.divider()

st.caption(
    "Career Shaper | Occupational data-based career exploration | "
    "For educational and exploratory use"
)
