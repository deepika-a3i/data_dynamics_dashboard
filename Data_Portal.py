import streamlit as st

import pandas as pd
import polars as pl

from data_processing import load_sales_data
from data_processing import find_plot_start_and_end_date
from data_processing import style_sidebar
import pycountry
from holidays import country_holidays
from holidays import utils as holidays_utils

supported_countries = ["Austria", "France", "Germany"]

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Data Dynamics! 👋")




def is_running_on_streamlit_cloud():
    # Check if "is_cloud" exists in secrets and is set to true
    running_on_cloud = st.secrets.get("general", {}).get("is_cloud", False)

    if running_on_cloud:
        pass
    else:
        st.warning("running on local")
    return running_on_cloud



def get_start_and_end_dates_all():
    if st.session_state.start_date_all is None:
        df = st.session_state.df_dict[list(st.session_state.df_dict.keys())[0]]
        df = df[list(df.keys())[0]]
        timestamp_column = [f for f in df.columns if "Timestamp" in f][0]
        st.session_state.start_date_all = df[timestamp_column].min()
        st.session_state.end_date_all = df[timestamp_column].max()


def select_period_options():

    if st.session_state.period_selection_mode is not None:
        default = st.session_state.period_selection_mode
    else:
        default ="Last Months"

    st.session_state.period_selection_mode = st.segmented_control(
        "Select period by",
        ["Dates", "Last Months"],
        default=default,
        
    )
    


    number_of_months_max = int((st.session_state.end_date_all - st.session_state.start_date_all).days / 30) + 1

    if st.session_state.period_selection_mode  == "Last Months":
        if st.session_state.number_of_months is not None:
            default = st.session_state.number_of_months
        else:
            default =5

        st.session_state.number_of_months = st.slider(
                "number_of_months",
                min_value=1,
                max_value=number_of_months_max,
                step=1,
                value=default,
            )
        st.session_state.start_date, st.session_state.end_date = find_plot_start_and_end_date(st.session_state.end_date_all,
                                                                                              st.session_state.number_of_months)
   
    else:

        
        if st.session_state.start_date is not None:
            default_start = st.session_state.start_date
            default_end = st.session_state.end_date
        else:
            default_start = st.session_state.start_date_all
            default_end = st.session_state.end_date_all
        col11, col12 = st.columns(2)
        with col11:
            st.session_state.start_date = st.date_input(
                "Start Date", min_value=st.session_state.start_date_all, 
                max_value=st.session_state.end_date_all, value=default_start
            )
        with col12:
            st.session_state.end_date = st.date_input(
                "End Date", min_value=st.session_state.start_date_all, 
                max_value=st.session_state.end_date_all, value=default_end
            )




def select_country_options():
    st.write(f'Country/Region: {st.session_state.country_name} / {st.session_state.region_name} ')
    choose_country = st.button('Choose country and region')

    if choose_country:

        st.session_state.country_name = st.selectbox(
            "Choose your country", sorted(supported_countries), index=0
        )
        st.session_state.country_code = pycountry.countries.get(name=st.session_state.country_name).alpha_2

        country_regions = holidays_utils.list_supported_countries()[st.session_state.country_code]

        if len(country_regions) > 0:

            region_code_to_name = {
                item.name: item.code.replace(f"{st.session_state.country_code}-", "")
                for item in pycountry.subdivisions.get(country_code=st.session_state.country_code)
            }

            region_code_to_name = {
                k: v for k, v in region_code_to_name.items() if v in country_regions
            }

            st.session_state.region_name = st.selectbox(
                "Choose your region",
                [None] + sorted(list(region_code_to_name.keys())),
                index=0,
            )
            if st.session_state.region_name is not None:
                st.session_state.region_code = region_code_to_name[st.session_state.region_name]
            else:
                st.session_state.region_code = None

        else:
            st.session_state.region_name = None
            st.session_state.region_code = None

        

def get_holidays_list(min_date,max_date):

    st.session_state.holidays_list = country_holidays(
        st.session_state.country_code, subdiv=None#st.session_state.region_code
    )[min_date:max_date]
    return st.session_state.holidays_list



def select_plot_options_common(forecasting=False):
    st.success("Data Formatted Successfully!")
    if not forecasting:
        st.markdown("#### Choose your plot options")
    else:
        st.markdown("#### Choose your forecasting options")
    
    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.product_or_product_group is not None:
            default = st.session_state.product_or_product_group
        else: 
            default = st.session_state.categorical_columns_of_interest[0]
        st.session_state.product_or_product_group = st.segmented_control(
            "Filter by",
            st.session_state.categorical_columns_of_interest,
            default=default,
        )
   

        if st.session_state.value_to_plot is not None:
            default = st.session_state.value_to_plot
        else: 
            default = st.session_state.numeric_columns_of_interest[0]

        st.session_state.value_to_plot = st.segmented_control(
            "Choose a value to display",
            st.session_state.numeric_columns_of_interest,
            default=default,
        )

    with col2:

    

        if st.session_state.selection_mode is not None:
            default_selection_mode = st.session_state.selection_mode
        else: 
            default_selection_mode = "Manually"

        st.session_state.selection_mode = st.segmented_control(
            "Select Elements to Plot",
            ["Manually", "Top or bottom"],
            default=default_selection_mode,

        )


        if st.session_state.selection_mode == "Top or bottom":

            if st.session_state.number_of_products_to_plot is not None:
                default_number_of_products_to_plot = st.session_state.number_of_products_to_plot
            else: 
                default_number_of_products_to_plot = 5

            if st.session_state.top_or_bottom is not None:
                default_top_or_bottom = ["top", "bottom"].index(st.session_state.top_or_bottom)
            else: 
                default_top_or_bottom = 0
            st.session_state.number_of_products_to_plot = st.slider(
                "Number of Elements to Plot",
                min_value=1,
                max_value=100,
                step=1,
                value=default_number_of_products_to_plot,
            )
            st.session_state.top_or_bottom = st.selectbox(
                "Top or Bottom", ["top", "bottom"], index=default_top_or_bottom
            )
            st.session_state.categories_to_plot = None
        else:
            df = st.session_state.df_dict[list(st.session_state.df_dict.keys())[0]]
            df = df[st.session_state.product_or_product_group]
          
            categories_to_select_from = st.session_state.pl_df.select(
                    [st.session_state.product_or_product_group]
                ).unique(
                    subset=[st.session_state.product_or_product_group],
                    maintain_order=True,
                ).to_series().to_list()
            
            categories_to_select_from = ['Total'] + categories_to_select_from


            if st.session_state.categories_to_plot is not None:

                default_categories_to_plot = st.session_state.categories_to_plot
                default_categories_to_plot = [cat for cat in default_categories_to_plot  
                                              if cat in categories_to_select_from ]
            else: 
                default_categories_to_plot = categories_to_select_from[0]

            st.session_state.categories_to_plot = st.multiselect(
                "Choose Elements to plot",
                categories_to_select_from,
                default=default_categories_to_plot,
            )
            st.session_state.number_of_products_to_plot = None
            st.session_state.top_or_bottom = "top"

        #if not forecasting:
        #    st.session_state.drop_products_with_zero_sales = st.checkbox(
        #    "Hide products with zero sales", value=True
        #    )
    st.session_state.resolution = st.segmented_control(
        "Choose a resolution",
        st.session_state.df_dict.keys(),
        default=list(st.session_state.df_dict.keys())[0],
    )
    col1, col2 = st.columns(2)
    with col1:
        select_period_options()

        #st.session_state.markers = st.checkbox("Show_markers", value=True)
        #st.session_state.line_shape = st.segmented_control(
        #    "type of selection",
        #    ["Step Plot", "Linear Plot"],
        #    default="Linear Plot",
        #    label_visibility="hidden",
        #)
        
        #if st.session_state.line_shape == "Step Plot":
        #    st.session_state.line_shape = "hv"
        #else:
        #    st.session_state.line_shape = "linear"

    with col2:



        bank_holidays = st.checkbox("Display Bank Holidays", value=True)

        if bank_holidays:
            select_country_options()
            get_holidays_list(st.session_state.start_date_all,st.session_state.end_date_all)

def init_ddd():
    if is_running_on_streamlit_cloud():
        st.session_state.use_athentication = True
        st.session_state.select_data = True
    else:
        st.session_state.use_athentication = False
        st.session_state.select_data = False
        

    if "authenfied_user" not in st.session_state:
        st.session_state.authenfied_user = False
    if "data_selected" not in st.session_state:
        st.session_state.data_selected = False
    if "pl_df" not in st.session_state:
        st.session_state.pl_df = None

    if "df_dict" not in st.session_state:
        st.session_state.df_dict = None
    if "trend_df_dict" not in st.session_state:
        st.session_state.trend_df_dict = None
    if "product_or_product_group" not in st.session_state:
        st.session_state.product_or_product_group = None
    if "resolution" not in st.session_state:
        st.session_state.resolution = None
    if "value_to_plot" not in st.session_state:
        st.session_state.value_to_plot = None
    if "number_of_products_to_plot" not in st.session_state:
        st.session_state.number_of_products_to_plot = None
    if "categories_to_plot" not in st.session_state:
        st.session_state.categories_to_plot = None
    if "top_or_bottom" not in st.session_state:
        st.session_state.top_or_bottom = None
    if "drop_products_with_zero_sales" not in st.session_state:
        st.session_state.drop_products_with_zero_sales = None
    if "start_date" not in st.session_state:
        st.session_state.start_date = None
    if "end_date" not in st.session_state:
        st.session_state.end_date = None
    if "holidays_list" not in st.session_state:
        st.session_state.holidays_list = None
    if "markers" not in st.session_state:
        st.session_state.markers = None
    if "line_shape" not in st.session_state:
        st.session_state.line_shape = None

    if "new_data_set" not in st.session_state:
        st.session_state.new_data_set = False
    if "trend_resolution" not in st.session_state:
        st.session_state.trend_resolution = "Weekday vs Weekend"
    if "trend_option" not in st.session_state:
        st.session_state.trend_option = None
    if "trend_year" not in st.session_state:
        st.session_state.trend_year = None
    if "selection_mode" not in st.session_state:
        st.session_state.selection_mode = None
    if "forecast_dict" not in st.session_state:
        st.session_state.forecast_dict = {}
        st.session_state.forecast_validation_dict = {}


        st.session_state.forecast_df = pd.DataFrame([])
        st.session_state.forecast_validation_df = pd.DataFrame([])

    if "country_name" not in st.session_state:
        st.session_state.country_name = 'Austria'
        st.session_state.country_code = 'AT'
        st.session_state.region_name = 'Wien'
        st.session_state.region_code = 'W'
    if 'start_date_all' not in st.session_state:

        st.session_state.start_date_all = None
        st.session_state.end_date_all = None
    if 'number_of_months'not in st.session_state:
        st.session_state.number_of_months = None
        st.session_state.period_selection_mode  = None
    if 'categorical_columns_of_interest' not in st.session_state:
        st.session_state.categorical_columns_of_interest = None
        st.session_state.numeric_columns_of_interest = None
    if 'is_running_on_streamlit_cloud'  not in st.session_state:
        st.session_state.is_running_on_streamlit_cloud = is_running_on_streamlit_cloud()
    if 'selected_plot' not in st.session_state:
        st.session_state.selected_plot = None
    if 'selected_box_plot' not in st.session_state:
        st.session_state.selected_box_plot = None   



def authenticate_user():
    correct_username = "dd-admin"
    correct_password = "dd-admin-2025"

    # Prompt the user for username and password
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    st.session_state.authenfied_user = (
        username == correct_username and password == correct_password
    )
    st.markdown("Please enter your Username and Password")
    if st.button("Login"):
        if st.session_state.authenfied_user:
            st.success("Login successful!")
        else:
            st.error("Invalid username or password. Please try again.")


def data_selection():
    st.markdown("## Data Selection")
    st.session_state.new_data_set = False
    if not st.session_state.select_data:
        st.session_state.pl_df = pl.read_csv("sales_data_example.csv")
        st.session_state.new_data_set = True

    option = st.radio("Choose data source", ["Upload a CSV file", "Select a dataset"])

    if option == "Upload a CSV file":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            st.session_state.pl_df = pl.read_csv(uploaded_file)
            st.session_state.new_data_set = True
    elif option == "Select a dataset":
        st.write("or select one of our data sets")
        data_set = st.selectbox(
            "Choose an option",
            ["Pizza Restaurant", "Coffee Shop", "Pizza Restaurant Raw Data"],
            index=1,
        )
        st.session_state.new_data_set = True

        if data_set == "Pizza Restaurant":
            st.session_state.pl_df = pl.read_csv("Pizza_Sales.csv")
        elif data_set == "Coffee Shop":
            st.session_state.pl_df = pl.read_csv("sales_data_example.csv")
        elif data_set == "Pizza Restaurant Raw Data":
            st.session_state.pl_df = pl.read_csv("Data Model - Pizza Sales.csv")
    elif st.session_state.pl_df is not None:
        st.session_state.new_data_set = False
        pass


def data_loading(pl_df):
    pl_df, numeric_columns, categorical_columns, timestamp_column = load_sales_data(
        pl_df
    )

    if timestamp_column is None:
        st.warning(f"Warning : No timestamp column in your data file!")
        return None, None, None
    if len(numeric_columns) == 0:
        st.warning(
            f"Warning : No numerical column (price, quantity, ...) in your data file!"
        )
        return None, None, None
    if len(categorical_columns) == 0:
        st.warning(
            f"Warning : No categorical column (product, product group, ...) in your data file!"
        )
        return None, None, None
    st.success("Data Loaded Successfully!")

    return pl_df, numeric_columns, categorical_columns



def main():
    style_sidebar()
    init_ddd()
    if not st.session_state.authenfied_user and st.session_state.use_athentication:

        authenticate_user()
    else:
        st.session_state.authenfied_user = True
        # st.sidebar.success("Select a dashboard above.")

        st.success("Login successful!")
        data_selection()
        if st.session_state.new_data_set:
            (
                st.session_state.pl_df,
                st.session_state.all_numeric_columns,
                st.session_state.all_categorical_columns,
            ) = data_loading(st.session_state.pl_df)

            col1, col2 = st.columns(2)


            with col1:

                if st.session_state.numeric_columns_of_interest is not None:
                    default = st.session_state.numeric_columns_of_interest
                else:
                    default =  st.session_state.all_numeric_columns
            #     st.session_state.numeric_columns_of_interest = st.segmented_control(
            #     "Choose your Numeric Values of Interest",
            #     st.session_state.all_numeric_columns,
            #     selection_mode='multi',
            #     default=default,
            # )
                st.session_state.numeric_columns_of_interest = default
            with col2:

                if st.session_state.categorical_columns_of_interest is not None:
                    default = st.session_state.categorical_columns_of_interest
                else:
                    default =  st.session_state.all_categorical_columns

            #     st.session_state.categorical_columns_of_interest = st.segmented_control(
            #     "Choose your grouping Categories of Interest",
            #     st.session_state.all_categorical_columns,
            #     selection_mode='multi',
            #     default=default,
            # )
                st.session_state.categorical_columns_of_interest = default



main()
