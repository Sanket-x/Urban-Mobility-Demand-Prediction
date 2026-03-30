import sys
import re

file_path = r'd:\College\DS\TS\frontend\app.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Globalize area_map so REVERSE_AREA_MAP logic actually works
# We will extract it from load_data and put it above it.

area_map_code = """    # Area mapping
    area_map = {
        "Area-1": "Indiranagar", "Area-2": "Koramangala", "Area-3": "Whitefield", "Area-4": "HSR Layout",
        "Area-5": "Electronic City", "Area-6": "Jayanagar", "Area-7": "JP Nagar", "Area-8": "BTM Layout",
        "Area-9": "Marathahalli", "Area-10": "Bellandur", "Area-11": "MG Road", "Area-12": "Malleshwaram",
        "Area-13": "Hebbal", "Area-14": "Yelahanka", "Area-15": "Banashankari", "Area-16": "Basavanagudi",
        "Area-17": "Rajajinagar", "Area-18": "RT Nagar", "Area-19": "CV Raman Nagar", "Area-20": "Kengeri",
        "Area-21": "Peenya", "Area-22": "Yeshwanthpur", "Area-23": "Malleswaram", "Area-24": "Vijayanagar",
        "Area-25": "Sadashivanagar", "Area-26": "Frazer Town", "Area-27": "Ulsoor", "Area-28": "Richmond Town",
        "Area-29": "Sanjay Nagar", "Area-30": "Mathikere", "Area-31": "Mahadevapura", "Area-32": "KR Puram",
        "Area-33": "Devanahalli", "Area-34": "Banaskankari", "Area-35": "Kumaraswamy Layout", "Area-36": "Nagawara",
        "Area-37": "Hennur", "Area-38": "Kalyan Nagar", "Area-39": "Horamavu", "Area-40": "Ramamurthy Nagar",
        "Area-41": "Kasturi Nagar", "Area-42": "Domlur", "Area-43": "Ejipura", "Area-44": "Adugodi",
        "Area-45": "Benson Town", "Area-46": "Cox Town", "Area-47": "Shivajinagar", "Area-48": "Vasanth Nagar",
        "Area-49": "Shanthi Nagar", "Area-50": "Wilson Garden"
    }"""

global_area_map = """
# Define mapping globally so API prediction module can securely do reverse lookups!
GLOBAL_AREA_MAP = {
    "Area-1": "Indiranagar", "Area-2": "Koramangala", "Area-3": "Whitefield", "Area-4": "HSR Layout",
    "Area-5": "Electronic City", "Area-6": "Jayanagar", "Area-7": "JP Nagar", "Area-8": "BTM Layout",
    "Area-9": "Marathahalli", "Area-10": "Bellandur", "Area-11": "MG Road", "Area-12": "Malleshwaram",
    "Area-13": "Hebbal", "Area-14": "Yelahanka", "Area-15": "Banashankari", "Area-16": "Basavanagudi",
    "Area-17": "Rajajinagar", "Area-18": "RT Nagar", "Area-19": "CV Raman Nagar", "Area-20": "Kengeri",
    "Area-21": "Peenya", "Area-22": "Yeshwanthpur", "Area-23": "Malleswaram", "Area-24": "Vijayanagar",
    "Area-25": "Sadashivanagar", "Area-26": "Frazer Town", "Area-27": "Ulsoor", "Area-28": "Richmond Town",
    "Area-29": "Sanjay Nagar", "Area-30": "Mathikere", "Area-31": "Mahadevapura", "Area-32": "KR Puram",
    "Area-33": "Devanahalli", "Area-34": "Banaskankari", "Area-35": "Kumaraswamy Layout", "Area-36": "Nagawara",
    "Area-37": "Hennur", "Area-38": "Kalyan Nagar", "Area-39": "Horamavu", "Area-40": "Ramamurthy Nagar",
    "Area-41": "Kasturi Nagar", "Area-42": "Domlur", "Area-43": "Ejipura", "Area-44": "Adugodi",
    "Area-45": "Benson Town", "Area-46": "Cox Town", "Area-47": "Shivajinagar", "Area-48": "Vasanth Nagar",
    "Area-49": "Shanthi Nagar", "Area-50": "Wilson Garden"
}

"""

if "GLOBAL_AREA_MAP" not in content:
    # Remove old local area_map definition block
    content = content.replace(area_map_code, "    # Reference global mapping\n    area_map = GLOBAL_AREA_MAP")
    
    # insert global mapping just before load_data()
    content = content.replace("def load_data():", global_area_map + "def load_data():")

# Fix the predict API logic
predict_api_old = "REVERSE_AREA_MAP = {v: k for k, v in area_map.items()} if 'area_map' in globals() else {}"
predict_api_new = "REVERSE_AREA_MAP = {v: k for k, v in GLOBAL_AREA_MAP.items()}"
content = content.replace(predict_api_old, predict_api_new)


# 2. Add Footer Logic
footer_logic = """
# ------------------------------------------------------------------------------
# 7. FOOTER LOGIC
# ------------------------------------------------------------------------------
def render_footer():
    st.markdown(\"""
        <style>
        .custom-footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: rgba(14, 17, 23, 0.85); /* Matches dark UI with opacity */
            color: #64748B;
            text-align: center;
            padding: 12px 0;
            font-size: 0.85rem;
            border-top: 1px solid rgba(255, 255, 255, 0.05); /* Slight separation line */
            z-index: 9999;
            backdrop-filter: blur(8px);
            text-shadow: 0 0 5px rgba(255, 255, 255, 0.05); /* Glow effect */
        }
        
        /* Ensure streamlit content doesn't get hidden behind the sticky footer */
        .block-container {
            padding-bottom: 70px !important;
        }
        </style>
        
        <div class="custom-footer">
            🚖 Urban Mobility AI System • Built by Sanket Thakore & Mridul Goswami • Powered by Machine Learning & FastAPI
        </div>
    \""", unsafe_allow_html=True)

# Call the footer at the very end of app execution
render_footer()
"""

# Only append if not already there
if "def render_footer():" not in content:
    content = content + "\n\n" + footer_logic


# 3. Remove Sidebar version text
version_text = "    st.markdown(\"<p style='color:#64748B; font-size:0.75rem; text-align:center;'>Version 2.0 • Data refreshed</p>\", unsafe_allow_html=True)"
content = content.replace(version_text, "")
# there might be another duplicate, or it might use single quote/double quotes differently. Let's make sure using regex just in case
content = re.sub(r'^[ \t]*st\.markdown\("<p[^>]*>Version 2.0 • Data refreshed</p>".*\n', '', content, flags=re.MULTILINE)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch successful!")
