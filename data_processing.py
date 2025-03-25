import polars as pl
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import streamlit as st
import plotly.colors as pc

resolution_map = {
    "Daily" : "day",
    "Weekly" : "week",
    "Monthly" : "month",
}
def get_lin_regression_trend(sales_df,
                            resolution,
                            product,
                            product_or_product_group,
                            value_to_plot,
                            regression_window):

    timestamp_col = f'Timestamp_{resolution}'

    df_reg = sales_df[sales_df[product_or_product_group] == product].copy()
    # df_reg.fillna({value_to_plot:0}, inplace = True)
    regression_window = [pd.to_datetime(regression_window[0]), pd.to_datetime(regression_window[1])]
    

    df_reg = df_reg[pd.to_datetime(df_reg[timestamp_col]) >= regression_window[0]]
    df_reg = df_reg[(pd.to_datetime(df_reg[timestamp_col]) <= regression_window[1])]

    # Convert timestamp to numeric (days since start)
    df_reg['x_axis'] = (df_reg[timestamp_col] - df_reg[timestamp_col].min()).dt.days


    # Fit Linear Regression Model
    X = df_reg[['x_axis']]
    y = df_reg[value_to_plot].values

    if X.shape[0] == 0 or y.shape[0] == 0:
        st.error("⚠️ Error: Insufficient data for regression. Try adjusting filters.")
        return None, None
    model = LinearRegression()
    model.fit(X, y)

    # Generate predicted values for the trendline
    df_reg[f"{value_to_plot}_trend"] = model.predict(X)

    trend_prct = (df_reg[f"{value_to_plot}_trend"].iloc[-1] 
                  - df_reg[f"{value_to_plot}_trend"].iloc[0])/df_reg[f"{value_to_plot}_trend"].iloc[0]

    
    return df_reg,0 if np.isnan(trend_prct) else int(trend_prct*100)

def add_holidays_count(sales_df,holidays_list,resolution):

    timestamp_column = f'Timestamp_{resolution}'

    holidays_df = pd.DataFrame({timestamp_column: holidays_list})

    # Ensure the dates in df2 are sorted
    holidays_df = holidays_df.sort_values(timestamp_column)
    sales_dates_df = sales_df[timestamp_column].unique()

    sales_dates_df = pd.DataFrame({timestamp_column: sales_dates_df})
    holidays_df[timestamp_column] = pd.to_datetime(holidays_df[timestamp_column])
    sales_dates_df[timestamp_column] = pd.to_datetime(sales_dates_df[timestamp_column])

    # Find the indices of rows in df2 that fall before/after each date in df1
    start_indices = holidays_df[timestamp_column].searchsorted(
        sales_dates_df[timestamp_column], side="left"
    )  # Inclusive start
    end_indices = holidays_df[timestamp_column].searchsorted(
        sales_dates_df[timestamp_column].shift(-1, fill_value=pd.Timestamp.max),
        side="left",
    )  # Exclusive end
    # Compute counts as the difference between indices
    sales_dates_df["holidays_count"] = end_indices - start_indices

    sales_df = sales_df.merge(sales_dates_df, on=timestamp_column, how="left")
    sales_df["holidays_count"] = sales_df["holidays_count"].fillna(0)

    return sales_df


possible_timestamp_column_names = [
    "Timestamp",
    "Time",
    "Date",
    "Day",
    "Week",
    "Month",
    "Year",
]

def find_timestamp_column(pl_df):
    df_columns = pl_df.columns

    print(" -------- Checking Columns : ")
    for feature in df_columns:
        print(" -------- Checking Column : ", feature)
        if feature.title() in possible_timestamp_column_names:
            try:
                converted_pl_df = pl_df.with_columns(
                    pl.col(feature).str.to_datetime().alias("Timestamp")
                )
                if feature != "Timestamp":
                    converted_pl_df = converted_pl_df.drop(feature)

                return converted_pl_df
            except:
                pass
            try:

                converted_pl_df = pl_df.with_columns(
                    pl.col(feature)
                    .str.strptime(pl.Datetime, format="%m/%d/%Y")
                    .alias("Timestamp")
                )

                if feature != "Timestamp":
                    converted_pl_df = converted_pl_df.drop(feature)

                return converted_pl_df
            except:
                print("+++++++ Not a proper timestamp column: ", feature)
    return None


def load_sales_data(pl_df):

    pl_df = find_timestamp_column(pl_df)
    if pl_df is None:
        print("*********** No Valid Timestamp")
        return None, None, None, None

    numeric_columns = pl_df.select(pl.col(pl.NUMERIC_DTYPES)).columns
    categorical_columns = [
        f
        for f in pl_df.columns
        if f not in numeric_columns
        and f not in possible_timestamp_column_names
        and f.title() not in possible_timestamp_column_names
    ]
    extra_timestamp_columns = []
    for cat in categorical_columns:
        try:
            pl.col(cat).str.to_datetime()
            categorical_columns.pop(cat)
            extra_timestamp_columns.append(cat)
        except:
            print("Not a ts column")

    pl_df = pl_df.drop(extra_timestamp_columns)

    dummy_pl_df = pl.Series(
        pl.datetime_range(
            pl_df["Timestamp"].dt.min(), pl_df["Timestamp"].dt.max(), "1d", eager=True
        ).alias("Timestamp")
    ).to_frame()
    pl_df = pl.concat([pl_df, dummy_pl_df], how="diagonal").sort("Timestamp")

    pl_df = pl_df.with_columns(
        pl.col(numeric_columns).fill_null(strategy="zero"),
        pl.col(categorical_columns).fill_null(strategy="forward"),
    )
    pl_df = pl_df.with_columns(
        pl.col(categorical_columns).fill_null(strategy="backward")
    )

    return pl_df, numeric_columns, categorical_columns, "Timestamp"

def add_Total_line_to_sales_df(df,
                               numeric_columns,
                               category_column,
                               resolution):

    df_total = (
                    df.select(numeric_columns + [category_column, resolution])
                    .group_by([resolution], maintain_order=True)
                    .sum()
                )
    df_total = df_total.with_columns(pl.col(category_column).fill_null('Total'))
    df_total = df_total.select(sorted(df_total.columns))
    df = df.select(sorted(df.columns))


    df = df.extend(df_total).sort(resolution)
    return df

def add_zeros_for_missing_timestamps(df_res_prod,resolution,product_or_product_group,numeric_columns):

    # Determine the interval based on user input
    if resolution == "day":
        interval = "1d"  # Daily
    elif resolution == "week":
        interval = "1w"  # Weekly
    elif resolution == "month":
        interval = "1mo"  # Monthly
    else:
        raise ValueError("Invalid aggregation level. Choose 'day', 'week', or 'month'.")

    # Find min and max timestamp
    min_date = df_res_prod.select(pl.col(resolution).min()).item()
    max_date = df_res_prod.select(pl.col(resolution).max()).item()

    # Generate a dynamic date range
    all_timestamps = pl.date_range(
        start=min_date, 
        end=max_date, 
        interval=interval, 
        eager=True  # Ensures immediate execution
    ).cast(pl.Datetime).to_frame().rename({"literal":resolution})  # Convert to DataFrame

    # Get unique product names
    all_products = df_res_prod.select(pl.col(product_or_product_group)).unique()

    # Create the full grid using a cross-join
    grid = all_timestamps.join(all_products, how="cross")

    # Merge with the original sales data
    df_res_prod = grid.join(df_res_prod, on=[resolution, product_or_product_group], how="left")

    # Fill missing values with 0
    df_res_prod = df_res_prod.with_columns(pl.col(value_to_plot).fill_null(0) for value_to_plot in numeric_columns )

    # Display the output
    return df_res_prod

def drop_incomplete_periods(resolution,pl_df):
            # Assuming `day` is a Polars datetime column
    first_day = pl_df["day"].min()
    last_day = pl_df["day"].max()
    import streamlit as st 

    if resolution == 'week':
        # Find first Monday after or equal to first_day
        first_day = (first_day + timedelta(days=(7 - first_day.weekday()) % 7))

        # Find last Sunday before or equal to last_day
        last_day = last_day - timedelta(days=(last_day.weekday() % 7) )- timedelta(seconds=1) 


    elif resolution == 'month':

        # Compute the first day of the month AFTER first_day
        if first_day.month == 12:  # Handle December case
            first_day_month = first_day.replace(year=first_day.year + 1, month=1, day=1)
        else:
            first_day_month = first_day.replace(month=first_day.month + 1, day=1)

        # Compute last day of previous month
        last_day_prev_month = last_day.replace(day=1) - timedelta(seconds=1)

        #print('first_day,last_day  month: ', first_day_month,last_day_prev_month)
        first_day ,  last_day = first_day_month , last_day_prev_month


    return pl_df.filter((pl.col("day") >= first_day) & (pl.col("day") <= last_day))


def group_by_day_week_month(pl_df, numeric_columns, categorical_columns):

    pl_df = pl_df.with_columns(
        [
            pl.col("Timestamp").dt.round("1d").alias("day"),  
            # Extract the day in 'YYYY-MM-DD' format
            (pl.col("Timestamp") - pl.duration(days=pl.col("Timestamp").dt.weekday()-1)).dt.truncate("1d")
    .alias("week"),
            pl.col("Timestamp").dt.truncate("1mo").alias("month"),
        ])
    

    df_dict = {}
    for resolution in ["day", "week", "month"]:
        df_dict[resolution] = {}
        for category_column in categorical_columns:            
            df_res_prod = drop_incomplete_periods(resolution,pl_df.clone())

            df_res_prod = (
                df_res_prod.select(numeric_columns + [category_column, resolution])
                .group_by([resolution, category_column], maintain_order=True)
                .sum()
            )
            
            
            #df_res_prod = df_res_prod.to_pandas().reset_index()
            df_res_prod = add_Total_line_to_sales_df(df_res_prod, 
                                                     numeric_columns,
                                                     category_column,   
                                                     resolution)
            
            df_res_prod = add_zeros_for_missing_timestamps(df_res_prod,
                                                           resolution,
                                                           category_column,
                                                           numeric_columns)

            

            df_res_prod = df_res_prod.rename({resolution:f"Timestamp_{resolution}" })

            df_res_prod = df_res_prod.to_pandas().reset_index()
            
            df_dict[resolution][category_column] = df_res_prod
            
    if df_dict == {}:
        return None

    return df_dict


# Function to determine season
def get_season(date):
    Y = date.year  # Extract year for leap year handling if needed
    seasons = {
        "Winter": ((pd.Timestamp(Y, 12, 21), pd.Timestamp(Y + 1, 3, 19))),
        "Spring": ((pd.Timestamp(Y, 3, 20), pd.Timestamp(Y, 6, 20))),
        "Summer": ((pd.Timestamp(Y, 6, 21), pd.Timestamp(Y, 9, 21))),
        "Autumn": ((pd.Timestamp(Y, 9, 22), pd.Timestamp(Y, 12, 20))),
    }

    for season, (start, end) in seasons.items():
        if start <= date <= end:
            return season
    return "Winter"  # Default case (Winter spans across years)


def create_weekend_sales_figure(
    trend_df_dict,
    value_to_plot="Gross Price",
    trend_resolution="Weekday vs Weekend",
    product_or_product_group="ProductGroup",
    number_of_products_to_plot=5,
    categories_to_plot=None,
    top_or_bottom="top",
    drop_products_with_zero_sales=True,
):
    sales_df = trend_df_dict["day"][product_or_product_group]
    sales_df["year"] = pd.to_datetime(sales_df["Timestamp_day"]).dt.year

    if trend_resolution == "Weekday vs Weekend":
        sales_df["is_weekend"] = pd.to_datetime(sales_df["Timestamp_day"]).dt.weekday.isin([5, 6])
        sales_df["is_weekend"] = sales_df["is_weekend"].map(
            {True: "Weekend", False: "Weekday"}
        )
        facet_col = "is_weekend"
    elif trend_resolution == "Seasons":
        sales_df["season"] = pd.to_datetime(sales_df["Timestamp_day"]).apply(get_season)
        facet_col = "season"

    sales_summary = (
        sales_df.groupby([facet_col, "year", product_or_product_group])[value_to_plot]
        .sum()
        .reset_index()
    )
    product_totals = (
        sales_df.groupby(product_or_product_group)[value_to_plot].sum().reset_index()
    )

    if top_or_bottom == "bottom":
        ascending = True
    else:
        ascending = False

    product_totals = product_totals.sort_values(value_to_plot, ascending=ascending)
    if drop_products_with_zero_sales:
        product_totals = product_totals[product_totals[value_to_plot] > 0]

    if categories_to_plot is None:
        products_to_plot = product_totals.head(number_of_products_to_plot)[
            product_or_product_group
        ].to_list()
    else:
        products_to_plot = categories_to_plot

    plot_sales_summary = sales_summary[
        sales_summary[product_or_product_group].isin(products_to_plot)
    ]

    fig = px.bar(
        plot_sales_summary,
        x="year",
        y=value_to_plot,
        color=product_or_product_group,
        facet_col=facet_col,  # Separate charts by year
        title=f"Sales Comparison:{trend_resolution} (Yearly per {product_or_product_group})",
        text_auto=True,
        barmode="group",
    )

    return fig, sales_summary

def get_timestamp_col(df):
    timestamp_col = [col for col in df.columns if col.startswith('Timestamp_')]
    if len(timestamp_col)>0:
        return timestamp_col[0]
    else:
        print('No Timestamp Column')
        return 'No_Timestamp_Column'


def find_plot_start_and_end_date(end_date_all,number_of_months):
    
    start_date = end_date_all - timedelta(days=30 * number_of_months)
    return start_date , end_date_all



def get_sorted_list_of_products(sales_df,
                                start_date,
                                end_date,
                                timestamp_column,
                                product_or_product_group,
                                value_to_plot,
                                top_or_bottom,
                                drop_products_with_zero_sales=True,
                                ):

    # Filter data for the last 3 months
    last_3_months = sales_df[
        (sales_df[timestamp_column] >= pd.to_datetime(start_date))
        & (sales_df[timestamp_column] <= pd.to_datetime(end_date))
    ]

    # Calculate total sales per product in the last 3 months
    product_totals = (
        last_3_months.groupby(product_or_product_group)[value_to_plot]
        .sum()
        .reset_index()
    )
    if drop_products_with_zero_sales:
        product_totals = product_totals[product_totals[value_to_plot] > 0]
    # Sort products by total sales (descending order)
    if top_or_bottom == "Bottom":
        ascending_flag = True
    else:
        ascending_flag = False

    product_totals = product_totals.sort_values(by=value_to_plot, ascending=ascending_flag)[product_or_product_group].to_list()
    #product_totals.remove('Total')
    return product_totals

def get_final_list_of_categories_to_plot(sales_df,
                                        start_date,
                                        end_date,
                                        timestamp_column,
                                        product_or_product_group,
                                        value_to_plot,
                                        top_or_bottom,
                                        number_of_products_to_plot,
                                        categories_to_plot,
                                        drop_products_with_zero_sales
                                        ):
    

    if categories_to_plot is None:
        sorted_product_totals = get_sorted_list_of_products(sales_df,
                                        start_date,
                                        end_date,
                                        timestamp_column,
                                        product_or_product_group,
                                        value_to_plot,
                                        top_or_bottom,
                                        drop_products_with_zero_sales)
        categories_to_plot = sorted_product_totals[:number_of_products_to_plot]
    return categories_to_plot
        


def create_weekdays_box_plot(
    df_dict,
    resolution="day",
    value_to_plot="gross_price",
    product_or_product_group="product",
    categories_to_plot= None,
    number_of_products_to_plot=5,
    top_or_bottom="top",
    drop_products_with_zero_sales=True,
    start_date = None,
    end_date = None,
):
    sales_df = df_dict[resolution][product_or_product_group]
    timestamp_column = f'Timestamp_{resolution}'
    sales_df[timestamp_column] = pd.to_datetime(sales_df[timestamp_column])
    sales_df = sales_df[
        (sales_df[timestamp_column] >= pd.to_datetime(start_date))
        & (sales_df[timestamp_column] <= pd.to_datetime(end_date))
    ]
    from_date = sales_df[timestamp_column].min().strftime('%Y-%m-%d')
    to_date = sales_df[timestamp_column].max().strftime('%Y-%m-%d')
    sales_df["weekday"] = sales_df[timestamp_column].dt.day_name()
    
    box_plot_figs_dict = {}
    categories_to_return = get_final_list_of_categories_to_plot(sales_df,
                                                            start_date,
                                                            end_date,
                                                            timestamp_column,
                                                            product_or_product_group,
                                                            value_to_plot,
                                                            top_or_bottom,
                                                            number_of_products_to_plot,
                                                            categories_to_plot,
                                                            drop_products_with_zero_sales
                                                            )
    sales_df = sales_df[sales_df[product_or_product_group].isin(categories_to_return)]
    for category in categories_to_return:
        # Step 1: Aggregate total sales per day per weekday
        cat_sales_df = sales_df[sales_df[product_or_product_group] == category]
        df_grouped = cat_sales_df.groupby(["weekday", f"{timestamp_column}"])[value_to_plot].sum().reset_index()

        # Step 2: Remove outliers using IQR method
        Q1 = df_grouped[value_to_plot].quantile(0.25)
        Q3 = df_grouped[value_to_plot].quantile(0.75)
        IQR = Q3 - Q1
        df_filtered = df_grouped[(df_grouped[value_to_plot] >= Q1 - 1.5 * IQR) & (df_grouped[value_to_plot] <= Q3 + 1.5 * IQR)]

        # Step 3: Create improved box plot
        fig = px.box(df_filtered, x="weekday", y=value_to_plot,
                    title=f"Sales Variability by Weekday {value_to_plot} for {category} | {from_date} - {to_date}",
                    labels={"weekday": "Weekday"},
                    category_orders={"weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]})

        # Adjust figure size
        fig.update_layout(width=1200, height=600)
        # st.plotly_chart(fig, use_container_width=True)
        box_plot_figs_dict.update({category : fig})
    return box_plot_figs_dict, categories_to_return


def create_plot_sales_figure(
    df_dict,
    resolution="week",
    value_to_plot="gross_price",
    product_or_product_group="product",
    number_of_products_to_plot=5,
    categories_to_plot=None,
    top_or_bottom="top",
    drop_products_with_zero_sales=True,
    start_date=None,
    end_date=None,
    holidays_list=[],
    line_shape="linear",
    markers=True,
    show_holidays = True
):

    sales_df = df_dict[resolution][product_or_product_group]

    timestamp_column = f'Timestamp_{resolution}'

    sales_df = sales_df[
        (sales_df[timestamp_column] >= pd.to_datetime(start_date))
        & (sales_df[timestamp_column] <= pd.to_datetime(end_date))
    ]
    # Ensure 'day' is sorted and convert to datetime (if not already)
    sales_df = sales_df.sort_values(timestamp_column)

    sales_df = add_holidays_count(sales_df,holidays_list,resolution)

    # Determine the last 3 months
    

    
    categories_to_plot = get_final_list_of_categories_to_plot(sales_df,
                                                            start_date,
                                                            end_date,
                                                            timestamp_column,
                                                            product_or_product_group,
                                                            value_to_plot,
                                                            top_or_bottom,
                                                            number_of_products_to_plot,
                                                            categories_to_plot,
                                                            drop_products_with_zero_sales
                                                            )
    if 'Total' not in categories_to_plot:
        categories_to_return = ['Total'] + categories_to_plot

    else:
        categories_to_return = categories_to_plot
    sales_df_to_return = sales_df[
        sales_df[product_or_product_group].isin(categories_to_return)
    ]

    sales_df = sales_df[sales_df[product_or_product_group].isin(categories_to_plot)]
    from_date = sales_df[timestamp_column].min().strftime('%Y-%m-%d')
    to_date = sales_df[timestamp_column].max().strftime('%Y-%m-%d')
    # Update the product column as a categorical type with the sorted order
    sales_df[product_or_product_group] = pd.Categorical(
        sales_df[product_or_product_group],
        categories=categories_to_plot,
        ordered=True,
    )

    # max_values_df = sales_df.groupby(timestamp_column).apply(group_by_function).reset_index()

    fig = px.line(
        sales_df,
        x=timestamp_column,
        y=value_to_plot,
        line_shape=line_shape,
        color=product_or_product_group,
        title=f"{value_to_plot} Evolution by {product_or_product_group} | {from_date} - {to_date}".title(),
        category_orders={
            product_or_product_group: categories_to_plot
        },
        markers=markers,
    )

    # Update layout to focus on the last 3 months
    fig.update_layout(
        xaxis=dict(range=[start_date, end_date]),  # Set initial range
        xaxis_title="",
        yaxis_title=value_to_plot.title(),
    )

    # Extract color mapping from the original figure
    color_map = {trace.name: trace.line.color for trace in fig.data if trace.name in categories_to_plot}
    # all_colors = pc.qualitative.Plotly
    # holiday_color = next(c for c in all_colors if c not in color_map.values())
    holiday_color = "gold"
    positive_super_count = sales_df.drop_duplicates(subset=[timestamp_column])
    positive_super_count = positive_super_count[
        positive_super_count["holidays_count"] > 0
    ]
    # List of specific timestamps to add vertical lines
    product_or_product_group
    specific_timestamps = holidays_list
    trend_dict = {}
    for product in categories_to_plot:
        df_reg,slope = get_lin_regression_trend(sales_df,
                                resolution,
                                product,
                                product_or_product_group,
                                value_to_plot,
                                regression_window =(start_date, end_date))       

        # Add Regression Line
        fig.add_trace(go.Scatter(
            x=df_reg[timestamp_column], 
            y=df_reg[f"{value_to_plot}_trend"], 
            mode='lines', 
            showlegend=False,
            line=dict(color=color_map.get(product, "black") , dash="dot"),
            
        ))
        trend_dict[product] = slope
    
    # Add vertical lines
    first_holiday = True
    if show_holidays:
        for ts in specific_timestamps:
            fig.add_trace(go.Scatter(
                x=[pd.Timestamp(ts), pd.Timestamp(ts)],  # Vertical line
                y=[0, sales_df[value_to_plot].max()],  # Full height of plot
                mode="lines",
                line=dict(color="gold", width=2),
                name="Holiday" if first_holiday else None,  # Show in legend only once
                showlegend=first_holiday,  # Only first trace appears in legend
                legendgroup="holidays"  # Group all traces under "holidays"
            ))
            first_holiday = False  # Set to False after first trace
            

    for i in range(len(positive_super_count) - 1):
        start = positive_super_count.iloc[i][timestamp_column]
        # Find the next timestamp from sales_df or set a default endpoint
        if i + 1 < len(sales_df):
            end = sales_df.loc[
                sales_df[timestamp_column] > start, timestamp_column
            ].min()
        else:
            end = sales_df[timestamp_column].max()

        # fig.add_shape(
        #     type="rect",
        #     x0=start,
        #     x1=end,
        #     y0=0,
        #     y1=1,  # Full vertical range
        #     xref="x",
        #     yref="paper",  # Extends the shaded region across the full vertical space
        #     fillcolor="gray",
        #     opacity=0.3,
        #     layer="below",
        #     line_width=0,
        # )
    fig.update_layout(hovermode = "x unified" )

    return (
        fig,
        sales_df_to_return[
            (sales_df_to_return[timestamp_column] >= pd.to_datetime(start_date))
            & (sales_df_to_return[timestamp_column] <= pd.to_datetime(end_date))
        ],
        start_date,
        end_date,
        trend_dict,
        categories_to_plot
    )
