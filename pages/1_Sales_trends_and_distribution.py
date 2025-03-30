import pandas as pd


import streamlit as st

from data_processing import create_plot_sales_figure
from data_processing import group_by_day_week_month
from dashboard_style import style_dashboard
import plotly.graph_objects as go



from Data_selection import init_ddd
from Data_selection import select_plot_options_common
from Data_selection import get_start_and_end_dates_all



def compute_others(sum_df):

    total_df = sum_df[sum_df[st.session_state.product_or_product_group] == "Total"]
    total_row = total_df.iloc[0]
    sum_df = sum_df[sum_df[st.session_state.product_or_product_group] != "Total"]

    col_to_compute_others = st.session_state.numeric_columns_of_interest + [
        col for col in sum_df.columns if (col).endswith("_%")
    ]
    row_sum = total_row - sum_df[col_to_compute_others].sum()

    df_other = pd.DataFrame([row_sum], columns=(sum_df.columns))
    df_other.loc[:, st.session_state.product_or_product_group] = "Others"

    # df_total[st.session_state.product_or_product_group] = "Other"  # Add a label for the sum row (optional)

    sum_df = pd.concat([sum_df, df_other, total_df], ignore_index=True)

    return sum_df


def show_csvs():
    st.markdown("## Trends & Distribution Reports")
    
    sum_df = (
        st.session_state.displayed_df[
            [col for col in st.session_state.displayed_df if "Timestamp" not in col]
        ]
        .groupby(by=st.session_state.product_or_product_group)
        .sum()
        .reset_index()
    )
    sum_df = sum_df.drop(columns=[col for col in sum_df.columns if "index" in col])

    sum_df = compute_others(sum_df)

    for numerical_feat in st.session_state.numeric_columns_of_interest:
        sum_df.loc[:, f"{numerical_feat}_%"] = (
            sum_df[numerical_feat]
            / sum_df[sum_df[st.session_state.product_or_product_group] == "Total"][
                numerical_feat
            ].iloc[0]
        ) * 100

    sum_df.loc[:, "start_date"] = st.session_state.start_date
    sum_df.loc[:, "end_date"] = st.session_state.end_date
    st.session_state.sum_df = sum_df

    df_to_show = sum_df[
        [st.session_state.product_or_product_group]
        + st.session_state.numeric_columns_of_interest
        + ["start_date", "end_date", "holidays_count"]
    ]
    st.markdown("#### Sales Performance & Growth Report")
    df_to_show = df_to_show.merge(
            pd.DataFrame(st.session_state.trend_dict.items(), 
                         columns=[st.session_state.product_or_product_group, 'trend_prct']),
            on=st.session_state.product_or_product_group,
            how="left",
        )
    df_to_show["start_date"] = pd.to_datetime(df_to_show["start_date"]).dt.strftime('%Y-%m-%d')
    df_to_show["end_date"] = pd.to_datetime(df_to_show["end_date"]).dt.strftime('%Y-%m-%d')
    desired_order = ["start_date", "end_date"] + [col for col in df_to_show.columns if col not in ["start_date", "end_date"]]
    st.dataframe(df_to_show[desired_order])
    # st.markdown("*Note: Others represent everything other than selected option.*")
    st.markdown("#### Sales Proportion Report")
    df_to_show = sum_df[["start_date", "end_date"] +
        [st.session_state.product_or_product_group]
        + [col for col in sum_df.columns if col.endswith("_%")]
        + [ "holidays_count"]
    ]
    df_to_show["start_date"] = pd.to_datetime(df_to_show["start_date"]).dt.strftime('%Y-%m-%d')
    df_to_show["end_date"] = pd.to_datetime(df_to_show["end_date"]).dt.strftime('%Y-%m-%d')
    
    st.dataframe(df_to_show)
    st.markdown("*Note: Others represent everything other than selected option.*")


def show_pie_charts():
    st.markdown("## Sales Composition")

    sum_df_to_plot = st.session_state.sum_df.copy()
    sum_df_to_plot = sum_df_to_plot[
        sum_df_to_plot[st.session_state.product_or_product_group] != "Total"
    ]
    # sum_df_to_plot.loc[:,st.session_state.product_or_product_group] = sum_df_to_plot.loc[:,st.session_state.product_or_product_group].fillna('Others')

    col1, col2 = st.columns(2)
    with col1:

        for col in sum_df_to_plot.columns[0::2]:
            if col.endswith("_%"):
                fig = go.Figure(
                    go.Pie(
                        labels=sum_df_to_plot[
                            st.session_state.product_or_product_group
                        ],  # Use the category column for labels
                        values=sum_df_to_plot[
                            col
                        ],  # Use the percentage column for values
                        hole=0.3,  # Optional: Makes it a donut chart
                    )
                )
                fig.update_layout(title=f"Share of {col} Across {st.session_state.product_or_product_group}")
                st.plotly_chart(fig)
    with col2:

        for col in sum_df_to_plot.columns[1::2]:
            if col.endswith("_%"):
                fig = go.Figure(
                    go.Pie(
                        labels=sum_df_to_plot[
                            st.session_state.product_or_product_group
                        ],  # Use the category column for labels
                        values=sum_df_to_plot[
                            col
                        ],  # Use the percentage column for values
                        hole=0.3,  # Optional: Makes it a donut chart
                    )
                )
                fig.update_layout(title=f"Share of {col} Across {st.session_state.product_or_product_group}")
                st.plotly_chart(fig)


def download_csv_files():

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download displayed data as a CSV",
            data=st.session_state.displayed_df.to_csv(index=False).encode("utf-8"),
            file_name="displayed_data.csv",
        )
    with col2:
        st.download_button(
            label="Download sum data as a CSV",
            data=st.session_state.sum_df.to_csv(index=False).encode("utf-8"),
            file_name="sum_data.csv",
        )


        

def main():
    style_dashboard()
    init_ddd()

    st.markdown("# Sales Trends & Distribution")

    if not (st.session_state.authenfied_user):
        st.warning("Please login to access the dashboard")

    else:

        if st.session_state.pl_df is None:
            st.warning("Please select a data set in the Data selection")

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
                (
                    fig,
                    st.session_state.displayed_df,
                    st.session_state.start_date,
                    st.session_state.end_date,
                    st.session_state.trend_dict,
                    st.session_state.categories_to_plot
                ) = create_plot_sales_figure(
                    st.session_state.df_dict,
                    resolution=st.session_state.resolution,
                    value_to_plot=st.session_state.value_to_plot,
                    product_or_product_group=st.session_state.product_or_product_group,
                    number_of_products_to_plot=st.session_state.number_of_products_to_plot,
                    categories_to_plot=st.session_state.categories_to_plot,
                    top_or_bottom=st.session_state.top_or_bottom,
                    drop_products_with_zero_sales=st.session_state.drop_products_with_zero_sales,
                    start_date=st.session_state.start_date,
                    end_date=st.session_state.end_date,
                    holidays_list=st.session_state.holidays_list,
                    markers=st.session_state.markers,
                    line_shape=st.session_state.line_shape,
                    show_holidays = st.session_state.bank_holidays,
                )
                
                st.plotly_chart(fig, use_container_width=True)

                show_csvs()
                show_pie_charts()
                download_csv_files()


main()
