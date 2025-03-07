
import streamlit as st

from data_processing import create_weekend_sales_figure
from data_processing import group_by_day_week_month
from data_processing import style_sidebar

from Data_Portal import init_ddd
from Data_Portal import select_plot_options_common

supported_countries = ["Austria", "France", "Germany"]




def select_trend_options():

    st.session_state.trend_resolution = st.segmented_control(
            "Choose a trend resolution",
            ["Weekday vs Weekend", "Seasons"],
            default="Weekday vs Weekend",)
    st.divider()
    st.markdown("### Choose your Trend options")
    if st.session_state.trend_resolution == "Seasons":
        st.session_state.trend_option = st.segmented_control(
            f"{st.session_state.trend_resolution}",
            ["Spring", "Summer", "Autumn", "Wineter"],
            default="Spring",
        )
    else:
        st.session_state.trend_option = st.segmented_control(
            f"{st.session_state.trend_resolution}",
            ["Weekday", "Weekend"],
            default="Weekday",
        )

    st.markdown("#### Choose an year")
    st.session_state.trend_year = st.selectbox(
        "Year", st.session_state.trend_df.year.unique(), index=0
    )


def show_best_products_in_trend():
    if st.session_state.trend_resolution == "Seasons":
        trend_df = st.session_state.trend_df[
            (st.session_state.trend_df.season == st.session_state.trend_option)
            & (st.session_state.trend_df.year == st.session_state.trend_year)
        ]
        trend_df = trend_df.drop(columns=["season", "year"])
    else:
        trend_df = st.session_state.trend_df[
            (st.session_state.trend_df.is_weekend == st.session_state.trend_option)
            & (st.session_state.trend_df.year == st.session_state.trend_year)
        ]
        trend_df = trend_df.drop(columns=["is_weekend", "year"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### Top 5 in {st.session_state.trend_year}")
        st.dataframe(
            trend_df.sort_values([st.session_state.value_to_plot], ascending=False)
            .head(5)
            .reset_index(drop=True)
        )
    with col2:
        st.markdown(f"### Bottom 5 in {st.session_state.trend_year}")
        st.dataframe(
            trend_df.sort_values([st.session_state.value_to_plot], ascending=True)
            .head(5)
            .reset_index(drop=True)
        )


def main():
    style_sidebar()
    init_ddd()

    if st.session_state.is_running_on_streamlit_cloud:
        st.markdown("# Dashboard Under Construction")

    else:
        st.markdown("# Sales Patterns & Seasonality")

        if not (st.session_state.authenfied_user):
            st.warning("Please login to access the dashboard")

        else:

            if st.session_state.pl_df is None:
                st.warning("Please select a data set in the Data Portal")

            else:

                if st.session_state.trend_df_dict is None or st.session_state.new_data_set:

                    st.session_state.trend_df_dict = group_by_day_week_month(
                        st.session_state.pl_df,
                        st.session_state.numeric_columns_of_interest,
                        st.session_state.categorical_columns_of_interest,
                    )
                if st.session_state.trend_df_dict is not None:
                    with st.sidebar:
                        select_plot_options_common()
                        
                    fig, st.session_state.trend_df = create_weekend_sales_figure(
                        st.session_state.trend_df_dict,
                        trend_resolution=st.session_state.trend_resolution,
                        value_to_plot=st.session_state.value_to_plot,
                        product_or_product_group=st.session_state.product_or_product_group,
                        number_of_products_to_plot=st.session_state.number_of_products_to_plot,
                        categories_to_plot=st.session_state.categories_to_plot,
                        top_or_bottom=st.session_state.top_or_bottom,
                        drop_products_with_zero_sales=st.session_state.drop_products_with_zero_sales,
                    )
                    
                    with st.sidebar:
                        select_trend_options()
                    st.plotly_chart(fig, use_container_width=True)

                    show_best_products_in_trend()


main()
