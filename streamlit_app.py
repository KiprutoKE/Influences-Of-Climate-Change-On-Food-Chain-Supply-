import streamlit as st
import pandas as pd
import plotly.express as px

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Climate & Food Supply Dashboard",
        page_icon="🌍",
        layout="wide"
    )

    # Custom title section
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 30px;">
            <h1 style="
                color: #2e8b57;
                font-size: 2.2em;
                margin: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            ">
                🌱 Influences of Climatic Patterns on Food Chain Supply in Kenya
            </h1>
        </div>
    """, unsafe_allow_html=True)

    # Horizontal line divider
    st.markdown("---")

    # Load historical data
    data = pd.read_csv("data.csv")

    # Rename columns for better readability
    data.rename(columns={
        'Beans_Price_90KG': 'Beans Price 90 KG',
        'Maize_Price_90KG': 'Maize Price 90 KG',
        'Potatoes_Price_50KG': 'Potatoes Price 50 KG',
        'Rice_Price_50KG': 'Rice Price 50 KG',
        'Temperature - (Celsius)': 'Temperature (Celsius)',
        'Rainfall (mm/year)': 'Rainfall (mm/year)'
    }, inplace=True)

    # Calculate percentage changes for price columns
    price_columns = ['Beans Price 90 KG', 'Maize Price 90 KG', 'Potatoes Price 50 KG', 'Rice Price 50 KG']
    for col in price_columns:
        data[f'{col}_Pct_Change'] = data[col].pct_change() * 100

    # ===========================================
    # Sidebar for Analysis Selection
    # ===========================================
    st.sidebar.markdown("---")
    analysis_type = st.sidebar.radio(
        "**Select Analysis Type:**", # Bolding the label for better visual emphasis
        ["Price Trends", "Correlation Analysis", "Pairwise Scatter Plots", "Distribution of Price Changes"]
    )

    # ===========================================
    # Analysis Sections based on Sidebar Selection
    # ===========================================

    if analysis_type == "Price Trends":
        st.title("🌾 Price Trends")
        st.write("Explore the historical price trends of different crops over the years, both in terms of absolute prices and year-over-year percentage changes.") # Intro text for Price Trends
        selected_crops_price_trends = st.sidebar.multiselect(
            "**Select Crops for Price Trends:**", # Bolding the label
            options=price_columns,
            default=price_columns
        )
        if selected_crops_price_trends:
            st.header("Price Trends Over Time")
            col1, col2 = st.columns(2)

            with col1:
                for crop in selected_crops_price_trends:
                    fig = px.line(
                        data,
                        x='Year', y=crop,
                        title=f'Trend of {crop} Prices Over Years', # More descriptive title
                        markers=True,
                        color_discrete_sequence=['#2ca02c'], # Changed color to green for trends
                        line_shape='spline'
                    )
                    fig.update_layout(yaxis_title='Price (KES)') # Added y-axis label
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Yearly Percentage Changes in Prices")
                for crop in selected_crops_price_trends:
                    fig = px.bar(
                        data,
                        x='Year', y=f'{crop}_Pct_Change',
                        title=f'Yearly Percentage Change in {crop} Prices', # More descriptive title
                        color=f'{crop}_Pct_Change',
                        color_continuous_scale='viridis' # Changed color scale
                    )
                    fig.update_layout(yaxis_title='Percentage Change (%)') # Added y-axis label
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one crop to display Price Trends.")


    elif analysis_type == "Correlation Analysis":
        st.title("📊 Correlation Analysis")
        st.header("Correlation Matrix Heatmap")
        st.write("This heatmap visualizes the correlation coefficients between different variables in the dataset. Correlation coefficients range from -1 to 1, where values close to 1 indicate a strong positive correlation, values close to -1 indicate a strong negative correlation, and values near 0 indicate a weak or no correlation.") # Improved description for Correlation Analysis

        corr_matrix = data.corr(numeric_only=True)

        fig_corr = px.imshow(
            corr_matrix,
            labels=dict(x="Variables", y="Variables", color="Correlation Coefficient"), # More descriptive labels
            x=corr_matrix.columns,
            y=corr_matrix.index,
            color_continuous_scale='RdBu',
            title="Correlation Matrix of Agricultural and Climatic Variables" # More descriptive title
        )
        st.plotly_chart(fig_corr, use_container_width=True)


    elif analysis_type == "Pairwise Scatter Plots":
        st.title("📈 Pairwise Scatter Plots")
        st.header("Exploring Relationships Between Variables")
        st.write("Using scatter plots to examine the relationships between pairs of variables. Select variables from the dropdown menus to visualize how they relate to each other. This can help identify potential linear or non-linear relationships.")

        scatter_x_var = st.sidebar.selectbox("**Select X Variable:**", options=data.columns)
        scatter_y_var = st.sidebar.selectbox("**Select Y Variable:**", options=data.columns) 

        if scatter_x_var and scatter_y_var and scatter_x_var != scatter_y_var:
            fig_scatter = px.scatter(
                data,
                x=scatter_x_var,
                y=scatter_y_var,
                title=f"Scatter Plot: {scatter_x_var} vs {scatter_y_var}", # More informative title
                trendline="ols" 
            )
            fig_scatter.update_layout(xaxis_title=scatter_x_var, yaxis_title=scatter_y_var) # Ensure axis titles are set
            st.plotly_chart(fig_scatter, use_container_width=True)
        elif scatter_x_var == scatter_y_var:
            st.warning("Please select different variables for X and Y axes in the scatter plot.")
        else:
            st.info("Select X and Y variables from the sidebar to generate a scatter plot.")


    elif analysis_type == "Distribution of Price Changes":
        st.title("📉 Distribution of Price Changes")
        st.header("Visualize Price Volatility")
        st.write("Explore the distribution of yearly percentage price changes for different crops. Histograms show the frequency of different percentage changes, while box plots provide a comparative summary of the central tendency, dispersion, and outliers for each crop's price changes.") # Intro text for Distribution Analysis

        selected_crops_distribution = st.sidebar.multiselect(
            "**Select Crops for Distribution Analysis:**", # Bolding the label
            options=price_columns,
            default=price_columns
        )

        if selected_crops_distribution:
            col_hist, col_box = st.columns(2)
            with col_hist:
                st.subheader("Histograms of Yearly Price Changes")
                for crop in selected_crops_distribution:
                    fig_hist = px.histogram(
                        data,
                        x=f'{crop}_Pct_Change',
                        title=f'Distribution of Yearly % Change for {crop} Prices', # More descriptive title
                        marginal="rug",
                        color_discrete_sequence=['#ff7f0e'] # Changed histogram color to orange
                    )
                    fig_hist.update_layout(xaxis_title='Yearly Percentage Change (%)', yaxis_title='Frequency') # Added axis labels
                    st.plotly_chart(fig_hist, use_container_width=True)

            with col_box:
                st.subheader("Comparative Box Plots of Yearly Price Changes")
                box_plot_data = [] # Prepare data for box plot - list of series for better naming
                for crop in selected_crops_distribution:
                    box_plot_data.append(data[f'{crop}_Pct_Change'])

                fig_box = px.box(
                    y=box_plot_data, # Use prepared data
                    labels={'value': 'Yearly % Change', 'variable': 'Crop'},
                    title='Comparison of Yearly Price Change Distributions Across Crops', # More descriptive title
                    points="all",
                    color_discrete_sequence=px.colors.qualitative.Set2 # Using a qualitative color palette for box plots
                )
                fig_box.update_layout(yaxis_title='Yearly Percentage Change (%)')
                # Dynamically set trace names for clarity in box plot legend
                for i, crop in enumerate(selected_crops_distribution):
                    fig_box.data[i].name = crop
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("Please select at least one crop for Distribution Analysis.")


    # Show raw data option at the bottom
    st.markdown("---")
    if st.checkbox("Show Raw Data"):
        st.dataframe(data)

if __name__ == "__main__":
    main()