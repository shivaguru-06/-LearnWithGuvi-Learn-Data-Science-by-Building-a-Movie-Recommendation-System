import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


#  Title & Description
st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title("🎬 Movie Recommendation System")
st.markdown("Find movies similar to your favorites using TF-IDF and cosine similarity.")


#  Load Data and Model (Cached)
@st.cache_data(show_spinner=True)
def load_data_and_model():
    df = pd.read_csv("movies.csv")


    # Automatically detect the text column
    possible_cols = ["overview", "description", "tags", "genres"]
    text_col = next((col for col in possible_cols if col in df.columns), None)


    if not text_col:
        st.error("No suitable text column found. Please ensure your CSV has 'overview', 'description', 'tags', or 'genres'.")
        st.stop()


    df = df.dropna(subset=["title", text_col])
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df[text_col].astype(str))


    indices = pd.Series(df.index, index=df["title"].str.lower()).drop_duplicates()
    return df, tfidf_matrix, indices




#  Load Cached Objects
df, tfidf_matrix, indices = load_data_and_model()


#  Recommendation Function
def get_recommendations(title, df, tfidf_matrix, indices, top_n=10):
    title = title.lower().strip()
    if title not in indices:
        st.error("Movie not found. Please check spelling or try another title.")
        return pd.DataFrame()


    idx = indices[title]
    sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[-top_n-1:][::-1]
    sim_indices = [i for i in sim_indices if i != idx]


    # Identify which column was used for TF-IDF
    possible_cols = ["overview", "description", "tags", "genres"]
    text_col = next((col for col in possible_cols if col in df.columns), None)


    # Safely return recommendations
    if text_col:
        return df.iloc[sim_indices][["title", text_col]]
    else:
        return df.iloc[sim_indices][["title"]]




#  Streamlit UI
st.subheader("Search for a Movie")
user_input = st.text_input("Enter a movie title:", "")


if st.button("Recommend"):
    if user_input.strip():
        recommendations = get_recommendations(user_input, df, tfidf_matrix, indices)
        if len(recommendations) > 0:
            st.success(f"Top {len(recommendations)} movies similar to '{user_input}':")
            text_col = [col for col in ["overview", "description", "tags", "genres"] if col in recommendations.columns]
            text_col = text_col[0] if text_col else None


            for _, row in recommendations.iterrows():
                st.markdown(f"### {row['title']}")
                if text_col:
                    st.write(row[text_col])
                st.markdown("---")


    else:
        st.warning("Please enter a movie title first.")


#  Footer
st.markdown("---")
st.caption("Built with Streamlit & scikit-learn")

