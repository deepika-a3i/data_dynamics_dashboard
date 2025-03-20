import pandas as pd
from datetime import timedelta

import streamlit as st


from data_processing import group_by_day_week_month
from data_processing import add_holidays_count
from data_processing import get_sorted_list_of_products
from dashboard_style import style_dashboard
from Data_Portal import init_ddd
from Data_Portal import select_plot_options_common
from Data_Portal import select_country_options
from Data_Portal import get_holidays_list
from Data_Portal import get_start_and_end_dates_all

from forecasting import forecast_sales
from forecasting import plot_forecast


def select_forecast_options():


    st.session_state.resolution = st.segmented_control(
        "Choose a resolution",
        st.session_state.df_dict.keys(),
        default=list(st.session_state.df_dict.keys())[0],
    )

    df = st.session_state.df_dict[list(st.session_state.df_dict.keys())[0]]
    df = df[list(df.keys())[0]]
    timestamp_column = [f for f in df.columns if "Timestamp" in f][0]
    min_date = df[timestamp_column].min()
    max_date = df[timestamp_column].max()

    select_country_options()
    get_holidays_list(min_date,max_date)

def main():
    style_dashboard()
    init_ddd()

    st.markdown("# Sales Forecast: Analysis & Accuracy")

    if not (st.session_state.authenfied_user):
        st.warning("Please login to access the dashboard")

    else:

        if st.session_state.pl_df is None:
            st.warning("Please select a data set in the Data Portal")

        else:

            if st.session_state.df_dict is None or st.session_state.new_data_set:

                st.session_state.df_dict = group_by_day_week_month(
                    st.session_state.pl_df,
                    st.session_state.numeric_columns_of_interest,
                    st.session_state.categorical_columns_of_interest,
                )
            if st.session_state.df_dict is not None:
                get_start_and_end_dates_all()
                with st.sidebar:
                    select_plot_options_common()


                sales_df = st.session_state.df_dict[st.session_state.resolution][st.session_state.product_or_product_group]
                
                
                sales_df = add_holidays_count(sales_df=sales_df,
                                              resolution=st.session_state.resolution,
                                              holidays_list= st.session_state.holidays_list)

                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                   run_forecasting = st.button("Run Forecasting")
                with col2:
                    run_forecasting_validation = st.button("Run Forecasting Validation")


                st.divider()

                if st.session_state.selection_mode == 'Top or bottom':
                    sorted_product_list = get_sorted_list_of_products(sales_df,
                                                st.session_state.start_date,
                                                st.session_state.end_date,
                                                f'Timestamp_{st.session_state.resolution}',
                                                st.session_state.product_or_product_group,
                                                st.session_state.value_to_plot,
                                                st.session_state.top_or_bottom,

                                                )
                    st.session_state.categories_to_plot = sorted_product_list[:st.session_state.number_of_products_to_plot]

                products_to_forecast = [product_element for product_element in st.session_state.categories_to_plot 
                                        if ((st.session_state.resolution,
                                st.session_state.product_or_product_group,
                                product_element,
                                st.session_state.value_to_plot) 
                            not in st.session_state.forecast_dict.keys())]

                if run_forecasting:
                    with st.spinner("Wait for it...", show_time=True):
                        for i,product_element in enumerate(products_to_forecast):
                            st.write(f"Forecasting for {product_element}, {i+1} of {len(products_to_forecast)}")
                            forecast_df = forecast_sales(sales_df, 
                                    st.session_state.product_or_product_group, 
                                    product_element,
                                    st.session_state.value_to_plot,
                                    st.session_state.resolution)
                                
            
                            st.session_state.forecast_dict[(st.session_state.resolution,
                                                                st.session_state.product_or_product_group,
                                                                product_element,
                                                                st.session_state.value_to_plot)] = forecast_df
                            st.session_state.forecast_df = pd.concat([st.session_state.forecast_df,
                                                                        forecast_df])
                        st.session_state.forecast_fig  = plot_forecast(sales_df, 
                                    st.session_state.forecast_dict, 
                                    st.session_state.categories_to_plot, 
                                    st.session_state.product_or_product_group, 
                                    st.session_state.resolution, 
                                    st.session_state.value_to_plot,
                                    st.session_state.start_date,
                                    st.session_state.end_date)

                    st.divider()                
                if run_forecasting_validation:

                    products_to_forecast_validation = [product_element for product_element in st.session_state.categories_to_plot 
                                        if ((st.session_state.resolution,
                                st.session_state.product_or_product_group,
                                product_element,
                                st.session_state.value_to_plot) 
                            not in st.session_state.forecast_validation_dict.keys())]

                    with st.spinner("Wait for it...", show_time=True):
                        for i,product_element in enumerate(products_to_forecast_validation):
                            st.write(f"Forecasting for {product_element}, {i+1} of {len(products_to_forecast)}")
                            days_delta = st.session_state.end_date - st.session_state.start_date + timedelta(days = 1)
                            sales_df[f'Timestamp_{st.session_state.resolution}'] = pd.to_datetime(sales_df[f'Timestamp_{st.session_state.resolution}'])
                            # st.write(type(sales_df[f'Timestamp_{st.session_state.resolution}'].values[0]))
                            forecast_validation_df = forecast_sales(sales_df[sales_df[f'Timestamp_{st.session_state.resolution}'] < st.session_state.start_date ], 
                                        st.session_state.product_or_product_group, 
                                        product_element,
                                        st.session_state.value_to_plot,
                                        st.session_state.resolution,
                                        forecast_horizon = (days_delta.days if st.session_state.resolution == "day" else (
                                            (days_delta.days // 7) if st.session_state.resolution == "week" else (days_delta.days // 30))),
                                        validation_forecast=True
                                        )
                            st.session_state.forecast_validation_dict[(st.session_state.resolution,
                                                                st.session_state.product_or_product_group,
                                                                product_element,
                                                                st.session_state.value_to_plot)] = forecast_validation_df
                            st.session_state.forecast_validation_df = pd.concat([st.session_state.forecast_validation_df,
                                                                        forecast_validation_df])

                        st.session_state.validate_fig = plot_forecast(
                                                                sales_df = (sales_df[(sales_df[f'Timestamp_{st.session_state.resolution}'] >= st.session_state.start_date) &
                                                                                (sales_df[f'Timestamp_{st.session_state.resolution}'] <= st.session_state.end_date)]),  
                                                                forecast_dfs = st.session_state.forecast_validation_dict, 
                                                                products_to_plot = st.session_state.categories_to_plot, 
                                                                product_or_product_group = st.session_state.product_or_product_group, 
                                                                resolution = st.session_state.resolution, 
                                                                value_to_plot = st.session_state.value_to_plot,
                                                                start_date = st.session_state.start_date,
                                                                end_date = st.session_state.end_date,
                                                                validation_plot =True)
                if len(st.session_state.forecast_df)>0:
                    
                    st.plotly_chart(st.session_state.forecast_fig, use_container_width=True)
                    st.markdown("#### Forecasted Reports")
                    st.dataframe(st.session_state.forecast_df)
                    


                if len(st.session_state.forecast_validation_df)>0:    
                    st.markdown("#### Forecast Validation")
                    st.plotly_chart(st.session_state.validate_fig, use_container_width=True)

main()
