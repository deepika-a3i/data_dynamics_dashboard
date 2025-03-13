import streamlit as st


from data_processing import group_by_day_week_month
from data_processing import create_weekdays_box_plot

from data_processing import style_dashboard


from Data_Portal import init_ddd
from Data_Portal import select_plot_options_common
from Data_Portal import get_start_and_end_dates_all


def show_box_plots():
    box_plot_figs_dict = create_weekdays_box_plot(
        st.session_state.df_dict,
        value_to_plot=st.session_state.value_to_plot,
        product_or_product_group=st.session_state.product_or_product_group,
        categories_to_plot=st.session_state.categories_to_plot,
        start_date=st.session_state.start_date,
        end_date=st.session_state.end_date,
        )
    box_plot_category = st.segmented_control("",st.session_state.categories_to_plot, default=st.session_state.categories_to_plot[0] )
    st.session_state.selected_box_plot = box_plot_category
    st.plotly_chart(box_plot_figs_dict[st.session_state.selected_box_plot], use_container_width=True)

# def show_additional_plots():
#     st.markdown("#### Toggle Between Box Plot and Pie Charts")
#     selected_plot = st.segmented_control("", ["Box Plot", "Pie Chart"], default="Box Plot")
#     st.session_state.selected_plot = 'box_plot' if selected_plot == "Box Plot" else 'pie_chart'
#     if st.session_state.selected_plot == 'pie_chart':
#         show_pie_charts()
#     elif st.session_state.selected_plot == 'box_plot':
#         show_box_plots()
        

def main():
    style_dashboard()
    init_ddd()

    st.markdown("# Sales Variability: Monday to Sunday")

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


                show_box_plots()


main()
