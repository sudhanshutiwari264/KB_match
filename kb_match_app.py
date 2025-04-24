import streamlit as st
# Set page config must be the first Streamlit command
st.set_page_config(page_title="MaaS360 KB Match Assistant", page_icon="📚", layout="wide")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Initialize session state
if 'solved' not in st.session_state:
    st.session_state.solved = False
if 'search_performed' not in st.session_state:
    st.session_state.search_performed = False
if 'issue_input' not in st.session_state:
    st.session_state.issue_input = ""
if 'solved_article' not in st.session_state:
    st.session_state.solved_article = ""

# Use TF-IDF instead of sentence transformers to avoid PyTorch issues
@st.cache_resource
def create_vectorizer():
    return TfidfVectorizer(stop_words='english')

vectorizer = create_vectorizer()

# Sample KB articles with categories
kb_articles = [
    {
        "id": "KB1023",
        "title": "iOS Devices Failing to Connect to Wi-Fi After Policy Push",
        "content": "Resolve iOS Wi-Fi issues post policy update by re-pushing Wi-Fi profiles via MaaS360. Check that the device has received the latest policy and verify certificate validity.",
        "category": "Connectivity"
    },
    {
        "id": "KB1078",
        "title": "Troubleshooting Wi-Fi Profiles in MaaS360",
        "content": "Steps to validate Wi-Fi certificate settings and policy sync on iOS and Android devices. Includes verification of SSID configuration and authentication methods.",
        "category": "Connectivity"
    },
    {
        "id": "KB1005",
        "title": "How to Enroll a New Device in MaaS360",
        "content": "Walkthrough for device enrollment using QR code or email link through MaaS360 portal. Covers both iOS and Android enrollment procedures.",
        "category": "Enrollment"
    },
    {
        "id": "KB1112",
        "title": "MaaS360 Container App Installation Failures",
        "content": "Troubleshooting steps for MaaS360 container app installation issues on mobile devices. Includes checking device compatibility, network connectivity, and storage requirements.",
        "category": "Applications"
    },
    {
        "id": "KB1245",
        "title": "Resolving MaaS360 VPN Connection Issues",
        "content": "Guide to diagnosing and fixing VPN connection problems with MaaS360 managed devices. Covers certificate validation, network settings, and policy configuration.",
        "category": "Connectivity"
    },
    {
        "id": "KB1367",
        "title": "Managing App Catalogs in MaaS360",
        "content": "Instructions for creating, updating, and distributing app catalogs to managed devices. Includes best practices for app version management and distribution strategies.",
        "category": "Applications"
    },
    {
        "id": "KB1489",
        "title": "MaaS360 Email Configuration Troubleshooting",
        "content": "Steps to resolve email configuration issues in MaaS360 managed devices. Covers Exchange, Gmail, and other email providers with detailed server settings.",
        "category": "Email"
    },
    {
        "id": "KB1523",
        "title": "Device Compliance Policy Management",
        "content": "Guide to creating and enforcing device compliance policies in MaaS360. Includes setting up automated actions for non-compliant devices and reporting options.",
        "category": "Policy"
    },
    {
        "id": "KB1678",
        "title": "MaaS360 Portal Access and Permission Issues",
        "content": "Troubleshooting guide for administrator access problems with the MaaS360 management portal. Includes role-based access control configuration and common permission errors.",
        "category": "Administration"
    },
    {
        "id": "KB1742",
        "title": "Secure Document Sharing with MaaS360",
        "content": "Best practices for secure document distribution and management using MaaS360 Document Catalog. Covers DLP settings, sharing restrictions, and content encryption options.",
        "category": "Security"
    },
    {
        "id": "KB1856",
        "title": "MaaS360 Mobile Threat Defense Configuration",
        "content": "Setup guide for enabling and configuring Mobile Threat Defense features in MaaS360. Includes malware detection settings and threat response automation.",
        "category": "Security"
    },
    {
        "id": "KB1921",
        "title": "iOS App Deployment Troubleshooting",
        "content": "Solutions for common iOS app deployment issues in MaaS360. Covers app signing problems, installation failures, and Apple Business Manager integration.",
        "category": "Applications"
    },
    {
        "id": "KB2034",
        "title": "Android Enterprise Enrollment Guide",
        "content": "Step-by-step instructions for setting up Android Enterprise enrollment in MaaS360. Covers work profile, fully managed, and dedicated device scenarios.",
        "category": "Enrollment"
    },
    {
        "id": "KB2156",
        "title": "MaaS360 Reporting and Analytics",
        "content": "Guide to generating and interpreting reports in MaaS360. Includes compliance reporting, security analytics, and custom report creation.",
        "category": "Administration"
    },
    {
        "id": "KB2287",
        "title": "Troubleshooting MaaS360 Browser Issues",
        "content": "Solutions for common problems with the MaaS360 secure browser. Includes connectivity issues, content filtering, and bookmark management.",
        "category": "Applications"
    }
]

# Extract unique categories for filtering
categories = ["All"] + sorted(list(set(article["category"] for article in kb_articles)))

# Add custom CSS for improved UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .subheader {
        font-size: 1.2rem;
        color: #424242;
        margin-bottom: 1.5rem;
    }
    .stExpander {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stButton>button {
        border-radius: 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .category-tag {
        background-color: #f0f2f6;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .high-match {
        color: #2e7d32;
        font-weight: 600;
    }
    .medium-match {
        color: #f57c00;
        font-weight: 600;
    }
    .low-match {
        color: #c62828;
        font-weight: 600;
    }
    .article-title {
        font-weight: 600;
        color: #1976D2;
    }
    .sidebar .stButton>button {
        width: 100%;
    }
    .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .stSlider>div>div>div>div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# App header with improved styling
st.markdown('<div style="text-align: center; padding: 1rem 0 2rem 0;">', unsafe_allow_html=True)
st.markdown('<p class="main-header">📚 MaaS360 KB Match Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Find the right knowledge base articles to solve your support tickets quickly</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Create a sidebar for filters with improved styling
with st.sidebar:
    st.markdown("## 🔍 Search Filters")
    
    # Category filter with improved styling
    st.markdown("### Category")
    selected_category = st.selectbox("Select a category", categories, index=0)
    
    # Confidence threshold with improved styling
    st.markdown("### Confidence Threshold")
    min_confidence = st.slider("Minimum match score", 0.0, 1.0, 0.1, 0.05)
    
    # Add a reset filters button
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.session_state.search_performed = False
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    st.markdown(f"**Total KB Articles:** {len(kb_articles)}")
    st.markdown(f"**Categories:** {len(categories)-1}")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    This tool uses TF-IDF vectorization to match support tickets with relevant knowledge base articles.
    
    1. Enter your issue description
    2. Click 'Find KB Articles'
    3. Review the suggested articles
    """)
    
    st.markdown("---")
    st.markdown("**Version:** 1.0.0")
    st.markdown("**Last Updated:** April 2025")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Input from user with improved styling
    st.markdown("### 🎫 Ticket Issue Description")
    issue_input = st.text_area(
        "Describe the issue in detail for better matching results...",
        height=180,
        placeholder="Example: My company iPhone is unable to connect to our corporate Wi-Fi network after the latest MaaS360 policy update. I've tried restarting the device but it still shows 'Unable to join network'...",
        key="issue_input",
        value=st.session_state.get('issue_input', '')
    )
    
    # Add a small helper text
    st.caption("💡 Tip: The more details you provide, the better the matching results will be")

with col2:
    if st.button("🔍 Find KB Articles", use_container_width=True) and issue_input:
        st.session_state.search_performed = True
    
    # Add some example queries with improved styling
    st.markdown("### 💡 Try an Example Query")
    
    example_queries = [
        "My iOS device won't connect to Wi-Fi after the latest update",
        "How do I enroll a new Android device?",
        "Email configuration not working on iPhone",
        "Need help with MaaS360 container app installation",
        "VPN connection issues on managed devices"
    ]
    
    # Create a more visually appealing layout for example queries
    for i, query in enumerate(example_queries):
        if st.button(f"📝 {query}", key=f"example_{i}", use_container_width=True):
            st.session_state.issue_input = query
            st.session_state.search_performed = True
            # Rerun to update the text area with the example query
            st.rerun()
            
    # Add a small note about examples
    st.caption("Click any example to see how the matching works")

# Store the issue input in session state when changed
if issue_input:
    st.session_state.issue_input = issue_input

# Process search if triggered
if st.session_state.get('search_performed', False) and st.session_state.get('issue_input', ''):
    issue_input = st.session_state.issue_input
    
    # Filter KB articles by category if needed
    filtered_kb = kb_articles
    if selected_category != "All":
        filtered_kb = [article for article in kb_articles if article["category"] == selected_category]
    
    if not filtered_kb:
        st.warning(f"No KB articles found in the '{selected_category}' category. Try another category.")
    else:
        # Vectorize issue and KB content using TF-IDF
        kb_texts = [kb["content"] for kb in filtered_kb]
        all_texts = kb_texts + [issue_input]
        
        # Fit and transform all texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Get the issue vector (last one) and KB vectors
        issue_vector = tfidf_matrix[-1:]
        kb_vectors = tfidf_matrix[:-1]
        
        # Compute similarity
        scores = cosine_similarity(issue_vector, kb_vectors).flatten()
        results = sorted(zip(filtered_kb, scores), key=lambda x: x[1], reverse=True)
        
        # Filter by minimum confidence
        results = [(article, score) for article, score in results if score >= min_confidence]
        
        if not results:
            st.warning("No matching KB articles found with the current confidence threshold. Try lowering the minimum confidence level.")
        else:
            # Display results
            st.subheader(f"🔗 Suggested KB Articles ({len(results)} matches)")
            
            # Create columns for better layout
            for i, (article, score) in enumerate(results):
                # Create an expander for each article with improved styling
                with st.expander(f"<span class='article-title'>{i+1}. {article['title']}</span> (Match: {int(score * 100)}%)", expanded=(i==0)):
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        st.markdown(f"**Article ID:**")
                        st.markdown(f"**Category:**")
                        st.markdown(f"**Relevance:**")
                    
                    with col2:
                        st.markdown(f"`{article['id']}`")
                        st.markdown(f"<span class='category-tag'>{article['category']}</span>", unsafe_allow_html=True)
                        # Create a colored confidence indicator with improved styling
                        if score >= 0.7:
                            st.markdown(f"<span class='high-match'>🟢 High ({round(score, 2)})</span>", unsafe_allow_html=True)
                        elif score >= 0.4:
                            st.markdown(f"<span class='medium-match'>🟡 Medium ({round(score, 2)})</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<span class='low-match'>🔴 Low ({round(score, 2)})</span>", unsafe_allow_html=True)
                    
                    with col3:
                        # Add a button to indicate this article solved the issue
                        if st.button(f"✅ This solved my issue", key=f"solved_{i}", use_container_width=True):
                            st.session_state.solved = True
                            st.session_state.solved_article = article['id']
                    
                    st.markdown("### Summary")
                    st.markdown(f"{article['content']}")
                    
                    # Add a divider for better visual separation
                    st.markdown("---")
            
            # Show a message if an article was marked as solving the issue
            if st.session_state.get('solved', False):
                st.success(f"""
                ### ✅ Issue Resolved!
                
                The issue has been successfully resolved using KB article **{st.session_state.solved_article}**.
                
                **Next Steps:**
                - The solution has been recorded for future reference
                - Similar tickets will be matched with this article
                """)
                
                # Add a button to start a new search
                if st.button("🔄 Start New Search", use_container_width=True):
                    st.session_state.solved = False
                    st.session_state.search_performed = False
                    st.session_state.issue_input = ""
                    st.rerun()
            else:
                # Create a better layout for the escalation section
                st.markdown("### 🆘 Need More Help?")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.warning("""
                    If none of the articles solve your issue, you can escalate to Tier 2 support.
                    
                    A specialist will contact you within 4 business hours.
                    """)
                with col2:
                    if st.button("� Escalate to Tier 2 Support", use_container_width=True):
                        st.error("""
                        ### 🚨 Issue Escalated
                        
                        Your issue has been escalated to Tier 2 support.
                        
                        **Ticket ID:** ESC-{}-{}
                        
                        A specialist will contact you shortly.
                        """.format(st.session_state.get('solved_article', 'NEW'), 
                                  pd.Timestamp.now().strftime('%Y%m%d%H%M')))
