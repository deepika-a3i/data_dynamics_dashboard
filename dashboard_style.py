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
#                 background-color: #FFBBBB !important;
#             }

#             /* Make segmented control background transparent */
#             div[data-baseweb="segmented-control"] {
#                 background-color: transparent !important;
#                 border: none !important;
#             }

#             /* Ensure segmented control buttons match theme */
#             div[data-baseweb="segmented-control"] div {
#                 background-color: transparent !important;
#                 color: #262730 !important; /* Dark text */
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
            background-color: white !important; /* Simple white background */
            color: #374151 !important; /* Dark gray for contrast */
            font-family: "Inter", sans-serif !important;
        }

        /* --- SIDEBAR STYLING --- */
        section[data-testid="stSidebar"] {
            background-color: #F2F2F2 !important; /* Soft gray-blue */
            padding: 20px;
            width: 450px !important;
            border-right: 3px solid #A5B4FC !important; /* Soft purple accent */
        }

        /* Sidebar Text */
        section[data-testid="stSidebar"] * {
            font-size: 15px !important;
            font-weight: 500 !important;
            color: #374151 !important; /* Dark gray for readability */
        }

        /* Sidebar Headers */
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
            color: #191970 !important; /* midnight blue */
            font-weight: bold;
        }

        /* Sidebar Buttons */
        section[data-testid="stSidebar"] button {
            background-color: #FAFAFA !important; /* Light pastel purple */
            color: #1A1A1A !important;
            font-weight: 700;
            border-radius: 6px;
            padding: 10px 20px;
            transition: all 0.3s ease-in-out;
        }

        /* Sidebar Button Hover */
        section[data-testid="stSidebar"] button:hover {
            background-color: #D8D8D8 !important; /* Bright pastel blue */
            color: #F1F5F9 !important; /* Light gray */
        }

        /* --- WIDGET STYLING --- */

        /* Buttons */
        button {
            background-color: #DDDDDD !important; /* Soft blue */
            color: #1E293B !important;
            font-weight: bold !important;
            border-radius: 8px !important;
            padding: 10px 16px !important;
            transition: all 0.3s ease-in-out;
        }

        /* Button Hover */
        button:hover {
            background-color: #1A1A1A !important; /* Deeper pastel blue */
            color: #F8FAFC !important;
        }

        /* Segmented Control */
        div[data-baseweb="segmented-control"] {
            background-color: transparent !important;
            border: none !important;
        }

        /* Segmented Control Default */
        div[data-baseweb="segmented-control"] div {
            background-color: #FFE4FF !important; /* Soft pastel peach */
            color: #1E293B !important;
            font-weight: 600 !important;
            border-radius: 6px !important;
        }

        /* Segmented Control Selected */
        div[data-baseweb="segmented-control"] div[aria-selected="true"] {
            background-color: #FF69AA !important; /* Soft pastel pink */
            color: #FF69AA !important;
            font-weight: bold !important;
            border: 5px solid #FF69B4 !important;
            border-width: thick !important;
            border-radius: 8px !important;
        }

        /* Multi-Select Default */
        div[data-baseweb="select"] div {
            background-color: #F3F6FF !important; /* Soft pastel green */
            color: #374151 !important;
            font-weight: 600;
        }

        /* Multi-Select Selected */
        div[data-baseweb="tag"] {
            background-color: #98FAFF !important; /* Soft light green */
            color: white !important;
            font-weight: 700;
            border: 2px solid #2E8B57 !important;
            border-radius: 6px !important;
            padding: 6px 10px !important;
        }

        /* Slicer Default */
        div[data-baseweb="slider"] {
            color: #374151 !important;
        }

        /* Slicer Active */
        div[data-baseweb="slider"] div[role="slider"] {
            background-color: #1A1A1A !important; /* Soft pastel purple */
            border-radius: 50% !important;
            border: 3px solid white !important;
        }

        /* --- TABLE STYLING --- */
        .stDataFrame {
            border: 2px solid #A5B4FC !important;
            border-radius: 10px !important;
        }

        /* --- CHART STYLING --- */
        .stChart {
            background-color: #F1F1F1 !important;
            border-radius: 10px !important;
            padding: 15px !important;
            box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.2);
        }
        </style>
    """, unsafe_allow_html=True)
