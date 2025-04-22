import streamlit as st
#from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load the embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Sample KB articles
kb_articles = [
    {
        "id": "KB1023",
        "title": "iOS Devices Failing to Connect to Wi-Fi After Policy Push",
        "content": "Resolve iOS Wi-Fi issues post policy update by re-pushing Wi-Fi profiles via MaaS360."
    },
    {
        "id": "KB1078",
        "title": "Troubleshooting Wi-Fi Profiles in MaaS360",
        "content": "Steps to validate Wi-Fi certificate settings and policy sync on iOS and Android devices."
    },
    {
        "id": "KB1005",
        "title": "How to Enroll a New Device in MaaS360",
        "content": "Walkthrough for device enrollment using QR code or email link through MaaS360 portal."
    }
]

st.title("📚 MaaS360 KB Match Assistant")
st.write("Paste a raw ticket issue description to get suggested KB articles.")

# Input from user
issue_input = st.text_area("🔍 Ticket Issue Description", height=150)

if st.button("Find KB Articles") and issue_input:
    # Encode issue and KB content
    kb_texts = [kb["content"] for kb in kb_articles]
    kb_embeddings = model.encode(kb_texts)
    issue_embedding = model.encode([issue_input])

    # Compute similarity
    scores = cosine_similarity(issue_embedding, kb_embeddings).flatten()
    results = sorted(zip(kb_articles, scores), key=lambda x: x[1], reverse=True)

    # Display results
    st.subheader("🔗 Suggested KB Articles:")
    for article, score in results:
        st.markdown(f"**{article['title']}**  ")
        st.markdown(f"*Article ID:* `{article['id']}`  ")
        st.markdown(f"*Confidence:* `{round(score, 2)}`  ")
        st.markdown(f"_Summary_: {article['content']}")
        st.markdown("---")

    st.success("If none of the articles solve the issue, escalate to Tier 2 support.")
