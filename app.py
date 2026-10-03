from pathlib import Path
import streamlit as st

# --------------------------------------------------
# PATHS
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="AI Smart Disaster Relief System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# PAGE NAVIGATION
# --------------------------------------------------
pages = [
    st.Page(
        ROOT / "pages" / "Home.py",
        title="Home",
        icon="🏠",
        default=True
    ),

    st.Page(
        ROOT / "pages" / "Dashboard.py",
        title="Dashboard",
        icon="📊"
    ),

    st.Page(
        ROOT / "pages" / "Severity_Prediction.py",
        title="Severity Prediction",
        icon="⚠️"
    ),

    st.Page(
        ROOT / "pages" / "Resource_Prediction.py",
        title="Resource Prediction",
        icon="📦"
    ),

    st.Page(
        ROOT / "pages" / "Resource_Allocation.py",
        title="Resource Allocation",
        icon="🚚"
    ),

    st.Page(
        ROOT / "pages" / "Maps.py",
        title="Maps",
        icon="🗺️"
    ),

    st.Page(
        ROOT / "pages" / "Reports.py",
        title="Reports",
        icon="📑"
    ),
]

# --------------------------------------------------
# START NAVIGATION
# --------------------------------------------------
navigation = st.navigation(
    pages,
    position="sidebar"
)

navigation.run()