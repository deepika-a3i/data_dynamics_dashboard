import streamlit as st

# def style_sidebar():
#     st.markdown("""
#         <style>
#             /* Sidebar container */
#             section[data-testid="stSidebar"] {
#                 background-color: transparent !important; /* Fully transparent */
#                 padding: 20px;
#                 width: 500px !important; /* Slightly wider */
#                 box-shadow: none !important; /* No shadow */
#                 border-right: 2px solid rgba(255, 75, 75, 0.8); /* Red accent border */
#             }

#             /* Hide the sidebar toggle (>, <) buttons */
#             button[title="Expand sidebar"], 
#             button[title="Collapse sidebar"] {
#                 display: none !important;
#             }

#             /* Sidebar text */
#             section[data-testid="stSidebar"] * {
#                 font-size: 14px !important; /* Reduced font size */
#                 font-weight: 400 !important;
#                 color: #262730 !important; /* Dark gray text for readability */
#             }

#             /* Sidebar headers */
#             section[data-testid="stSidebar"] h1, 
#             section[data-testid="stSidebar"] h2, 
#             section[data-testid="stSidebar"] h3 {
#                 color: #FF4444 !important; /* Streamlit red */
#                 font-weight: bold;
#                 font-size: 14px !important;
#             }

#             /* Sidebar buttons */
#             section[data-testid="stSidebar"] button {
#                 /* background-color: #FF4B4B !important;  Streamlit red */
#                 color: white !important;
#                 border-radius: 6px !important;
#                 border: none !important;
#                 font-size: 14px !important;
#                 padding: 8px 14px !important;
#             }

#             /* Hover effect for buttons */
#             section[data-testid="stSidebar"] button:hover {
#                 background-color: #4A4A4A !important;
#             }

#             /* Make segmented control background transparent */
#             div[data-baseweb="segmented-control"] {
#                 background-color: transparent !important;
#                 border: none !important;
#             }

#             /* Ensure segmented control buttons match theme */
#             div[data-baseweb="segmented-control"] div {
#                 background-color: transparent !important;
#                 color: #4A4A4A !important; /* Dark text */
#                 font-weight: bold !important;
#             }

#             /* Highlight active segmented control button */
#             div[data-baseweb="segmented-control"] div[aria-selected="true"] {
#                 background-color: rgba(255, 75, 75, 0.4) !important; /* Darker red tint */
#                 border: 2px solid rgba(255, 75, 75, 0.8) !important; /* Darker red border */
#                 border-radius: 6px !important;
#             }
#             /* Make selected multiselect options a lighter shade */
#             div[data-baseweb="tag"] {
#                 background-color: rgba(255, 75, 75, 0.2) !important; /* Lighter red */
#                 color: black !important;
#                 border-radius: 4px !important;
#             }

#         </style>
#     """, unsafe_allow_html=True)

def style_dashboard():
    st.markdown("""
        <style>
        /* --- GLOBAL STYLING FOR PASTEL IT-THEMED DASHBOARD --- */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: transparent !important; /* Simple white background */
            color: #374151 !important; /* Dark gray for contrast */
            font-family: "Inter", sans-serif !important;
        }

        /* --- SIDEBAR STYLING --- */
        section[data-testid="stSidebar"] {
            background-color: transparent !important; /* Soft gray-blue */
            padding: 20px;
            width: 450px !important;
            border-right: 3px solid #A5B4FC !important; /* Soft purple accent */
        }

        /* Sidebar Text */
        section[data-testid="stSidebar"] * {
            font-size: 15px !important;
            font-weight: 500 !important;
            color: #000000 !important; /* Dark gray for readability */
        }

        /* Sidebar Headers */
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
            color: #000000 !important; /* midnight blue */
            font-weight: bold;
        }

        /* Sidebar Buttons */
        section[data-testid="stSidebar"] button {
            background-color: transparent !important; /* Light pastel purple */
            color: #4A4A4A !important;
            font-weight: 700;
            border-radius: 6px;
            padding: 10px 20px;
            transition: all 0.3s ease-in-out;
        }

        /* Sidebar Button Hover */
        section[data-testid="stSidebar"] button:hover {
            background-color: #C0C0C0 !important; /* Bright pastel blue */
            color: #FFFFFF !important; /* Light gray */
        }

        div[data-testid="stButton"] button {
            width: 300px !important;  /* Fixed button width */
            height: 40px !important;  /* Fixed button height */
            font-size: 16px !important;  /* Readable font size */
            font-weight: 600 !important;
            background-color: transparent !important;  /* Light gray background */
            color: #333 !important;  /* Dark text for contrast */
            border: 3px solid #E0E0E0 !important;
            border-radius: 8px !important;
            transition: none !important;  /* Removes fade effect */
        }

        /* Change hover background to gray instead of fading */
        div[data-testid="stButton"] button:hover {
            background-color: #C0C0C0 !important;  /* Darker gray on hover */
            color: black !important;
        }
        div[data-testid="stDownloadButton"] button {
            width: 300px !important;  /* Same fixed width */
            height: 40px !important;  /* Same fixed height */
            font-size: 16px !important;
            font-weight: 600 !important;
            background-color: transparent !important;  /* Light gray */
            color: #333 !important;  /* Dark text */
            border: 3px solid #E0E0E0 !important;
            border-radius: 8px !important;
            transition: none !important;  /* Removes hover fade */
        }

        /* Change hover background to darker gray */
        div[data-testid="stDownloadButton"] button:hover {
            background-color: #C0C0C0 !important;
            color: black !important;
        }
        </style>
    """, unsafe_allow_html=True)
