import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# For cross-validation, scaling, and evaluation metrics
from sklearn.model_selection import cross_validate, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Models: Gaussian Process Regression and Random Forest Regressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor

# Prophet for time-series forecasting
from prophet import Prophet

# For residual plots
import matplotlib.pyplot as plt

def main():
    st.set_page_config(page_title="Climate & Food Supply Dashboard", page_icon="🌍", layout="wide")
    
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 30px;">
            <h1 style="color: #2e8b57; font-size: 2.8em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);">
                🌱 Influences of Climatic Patterns on Food Chain Supply in Kenya
            </h1>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Load dataset
    data = pd.read_csv("data.csv")
    
    # Rename columns for readability
    data.rename(columns={
        'Beans_Price_90KG': 'Beans Price 90 KG',
        'Maize_Price_90KG': 'Maize Price 90 KG',
        'Potatoes_Price_50KG': 'Potatoes Price 50 KG',
        'Rice_Price_50KG': 'Rice Price 50 KG',
        'Temperature - (Celsius)': 'Temperature (Celsius)',
        'Rainfall (mm/year)': 'Rainfall (mm/year)'
    }, inplace=True)
    
    # Calculate percentage change for price columns
    price_columns = ['Beans Price 90 KG', 'Maize Price 90 KG', 'Potatoes Price 50 KG', 'Rice Price 50 KG']
    for col in price_columns:
        data[f'{col}_Pct_Change'] = data[col].pct_change() * 100
    
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(data.mean(), inplace=True)
    
    # Sidebar Navigation
    st.sidebar.markdown("---")
    dashboard_section = st.sidebar.radio("**Dashboard Sections:**", ["Exploratory Analysis", "Model Predictions"])
    
    # ------------------------ Exploratory Analysis ------------------------ #
    if dashboard_section == "Exploratory Analysis":
        st.sidebar.subheader("Exploratory Data Analysis")
        analysis_type = st.sidebar.radio("**Select Analysis Type:**", 
                                         ["Price Trends", "Food Production Index Trend", "Correlation Analysis", "Pairwise Scatter Plots", "Distribution of Price Changes"])
        
        if analysis_type == "Price Trends":
            st.title("🌾 Price Trends")
            st.write("Explore historical price trends for selected crops.")
            selected_crops_price_trends = st.sidebar.multiselect("**Select Crops for Price Trends:**", 
                                                                 options=price_columns, default=price_columns)
            if selected_crops_price_trends:
                st.header("Price Trends Over Time")
                col1, col2 = st.columns(2)
                with col1:
                    for crop in selected_crops_price_trends:
                        fig = px.line(data, x='Year', y=crop, 
                                      title=f'Trend of {crop} Prices Over Years', 
                                      markers=True, color_discrete_sequence=['#2ca02c'], 
                                      line_shape='spline')
                        fig.update_layout(yaxis_title='Price (KES)')
                        st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.subheader("Yearly Percentage Changes in Prices")
                    for crop in selected_crops_price_trends:
                        fig = px.bar(data, x='Year', y=f'{crop}_Pct_Change', 
                                     title=f'Yearly Percentage Change in {crop} Prices', 
                                     color=f'{crop}_Pct_Change', color_continuous_scale='viridis')
                        fig.update_layout(yaxis_title='Percentage Change (%)')
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one crop to display Price Trends.")
        
        elif analysis_type == "Food Production Index Trend":
            st.title("🍲 Food Production Index Trend")
            st.header("Food Production Index Over Years")
            fig_fpi = px.line(data, x='Year', y='Food Production Index', 
                              title="Food Production Index Trend", 
                              markers=True, line_shape='spline')
            fig_fpi.update_layout(yaxis_title='Food Production Index')
            st.plotly_chart(fig_fpi, use_container_width=True)
        
        elif analysis_type == "Correlation Analysis":
            st.title("📊 Correlation Analysis")
            st.header("Correlation Matrix Heatmap")
            corr_matrix = data.corr(numeric_only=True)
            fig_corr = px.imshow(corr_matrix, 
                                 labels=dict(x="Variables", y="Variables", color="Correlation Coefficient"), 
                                 x=corr_matrix.columns, y=corr_matrix.index, 
                                 color_continuous_scale='RdBu',
                                 title="Correlation Matrix of Agricultural and Climatic Variables")
            st.plotly_chart(fig_corr, use_container_width=True)
        
        elif analysis_type == "Pairwise Scatter Plots":
            st.title("📈 Pairwise Scatter Plots")
            st.header("Explore Relationships Between Variables")
            scatter_x_var = st.sidebar.selectbox("**Select X Variable:**", options=data.columns)
            scatter_y_var = st.sidebar.selectbox("**Select Y Variable:**", options=data.columns)
            if scatter_x_var and scatter_y_var and scatter_x_var != scatter_y_var:
                fig_scatter = px.scatter(data, x=scatter_x_var, y=scatter_y_var, 
                                         title=f"Scatter Plot: {scatter_x_var} vs {scatter_y_var}",
                                         trendline="ols")
                fig_scatter.update_layout(xaxis_title=scatter_x_var, yaxis_title=scatter_y_var)
                st.plotly_chart(fig_scatter, use_container_width=True)
            elif scatter_x_var == scatter_y_var:
                st.warning("Please select different variables for X and Y.")
            else:
                st.info("Select X and Y variables from the sidebar to generate a scatter plot.")
        
        elif analysis_type == "Distribution of Price Changes":
            st.title("📉 Distribution of Price Changes")
            st.header("Visualize Price Volatility")
            selected_crops_distribution = st.sidebar.multiselect("**Select Crops for Distribution Analysis:**", 
                                                                   options=price_columns, default=price_columns)
            if selected_crops_distribution:
                col_hist, col_box = st.columns(2)
                with col_hist:
                    st.subheader("Histograms of Yearly Price Changes")
                    for crop in selected_crops_distribution:
                        fig_hist = px.histogram(data, x=f'{crop}_Pct_Change', 
                                                title=f'Distribution of Yearly % Change for {crop} Prices',
                                                marginal="rug", color_discrete_sequence=['#ff7f0e'])
                        fig_hist.update_layout(xaxis_title='Yearly Percentage Change (%)', yaxis_title='Frequency')
                        st.plotly_chart(fig_hist, use_container_width=True)
                with col_box:
                    st.subheader("Comparative Box Plots of Yearly Price Changes")
                    box_plot_data = [data[f'{crop}_Pct_Change'] for crop in selected_crops_distribution]
                    fig_box = px.box(y=box_plot_data, 
                                     labels={'value': 'Yearly % Change', 'variable': 'Crop'},
                                     title='Comparison of Yearly Price Change Distributions Across Crops',
                                     points="all", color_discrete_sequence=px.colors.qualitative.Set2)
                    fig_box.update_layout(yaxis_title='Yearly Percentage Change (%)')
                    for i, crop in enumerate(selected_crops_distribution):
                        fig_box.data[i].name = crop
                    st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.warning("Please select at least one crop for Distribution Analysis.")
    
    # ------------------------ Model Predictions ------------------------ #
    elif dashboard_section == "Model Predictions":
        st.title("🔮 Crop Price Prediction")
        st.write("Predict crop prices based on historical data, evaluate model performance, and forecast prices for the next five years.")

        st.sidebar.subheader("Model Configuration")
        target_variable = st.sidebar.selectbox("**Select Target Variable (Crop Price):**", options=price_columns)
        # Exclude target, other price columns, and computed percentage change column
        feature_columns_selector = [col for col in data.columns if col not in price_columns + [target_variable, 'Year', f'{target_variable}_Pct_Change']]
        feature_columns = st.sidebar.multiselect("**Select Feature Columns:**", options=feature_columns_selector, default=feature_columns_selector)
        
        model_type = st.sidebar.selectbox("**Select Model Type:**", 
                                          options=["Gaussian Process Regression", "Random Forest Regressor", "Prophet"])
        test_size = st.sidebar.slider("**Test Data Size (%):**", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        performance_metric = st.sidebar.selectbox("**Select Performance Metric:**", 
                                                    options=["R-squared", "RMSE", "MAE", "MAPE"], index=0)
        
        if not target_variable or not feature_columns:
            st.warning("Please select a Target Variable and at least one Feature Column to train the model.")
        else:
            st.header(f"Predicting {target_variable}")
            
            # ------------------ Gaussian Process Regression ------------------ #
            if model_type == "Gaussian Process Regression":
                X = data[feature_columns]
                y = data[target_variable]
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
                model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
                
                scoring_metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error']
                loo = LeaveOneOut()
                cv_results = cross_validate(model, X_scaled, y, cv=loo, scoring=scoring_metrics, return_estimator=True)
                
                avg_r2   = cv_results['test_r2'].mean()
                avg_rmse = np.sqrt(-cv_results['test_neg_mean_squared_error'].mean())
                avg_mae  = -cv_results['test_neg_mean_absolute_error'].mean()
                avg_mape = -cv_results['test_neg_mean_absolute_percentage_error'].mean()
                
                st.subheader("Model Evaluation - LOOCV (Gaussian Process Regression)")
                st.write(f"**Average R²:** {avg_r2:.2f}")
                st.write(f"**Average RMSE:** {avg_rmse:.2f} (KES)")
                st.write(f"**Average MAE:** {avg_mae:.2f} (KES)")
                st.write(f"**Average MAPE:** {avg_mape:.2f}%")
                
                best_model_idx = np.argmax(cv_results['test_r2'])
                best_model = cv_results['estimator'][best_model_idx]
                predictions_train = best_model.predict(X_scaled)
                residuals = y - predictions_train
                fig_res, ax_res = plt.subplots()
                ax_res.scatter(predictions_train, residuals, color='blue', alpha=0.6)
                ax_res.axhline(0, color='red', linestyle='--')
                ax_res.set_xlabel("Predicted Values")
                ax_res.set_ylabel("Residuals")
                ax_res.set_title("Residual Plot (GPR)")
                st.pyplot(fig_res)
                
                st.subheader("Price Prediction for Next 5 Years (2026-2030) with GPR")
                future_years = range(2026, 2031)
                future_data = pd.DataFrame({'Year': future_years})
                for feature in feature_columns:
                    future_data[feature] = data[feature].mean()
                X_future = future_data[feature_columns]
                X_future_scaled = scaler.transform(X_future)
                future_predictions = best_model.predict(X_future_scaled)
                future_data['Predicted Price'] = future_predictions
                
                fig_future = px.line(data, x='Year', y=target_variable, 
                                     title=f'Historical Prices vs. Predicted Prices for {target_variable} (GPR)',
                                     labels={'Year': 'Year', target_variable: 'Historical Price (KES)'})
                future_trace = px.line(future_data, x='Year', y='Predicted Price', color_discrete_sequence=['red']).data[0]
                future_trace.name = 'Predicted Price (Next 5 Years - GPR)'
                fig_future.add_trace(future_trace)
                fig_future.update_layout(yaxis_title='Price (KES)')
                st.plotly_chart(fig_future, use_container_width=True)
            
            # ------------------ Random Forest Regressor ------------------ #
            elif model_type == "Random Forest Regressor":
                X = data[feature_columns]
                y = data[target_variable]
                
                model = RandomForestRegressor(random_state=42)
                scoring_metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error']
                loo = LeaveOneOut()
                cv_results = cross_validate(model, X, y, cv=loo, scoring=scoring_metrics, return_estimator=True)
                
                avg_r2   = cv_results['test_r2'].mean()
                avg_rmse = np.sqrt(-cv_results['test_neg_mean_squared_error'].mean())
                avg_mae  = -cv_results['test_neg_mean_absolute_error'].mean()
                avg_mape = -cv_results['test_neg_mean_absolute_percentage_error'].mean()
                
                st.subheader("Model Evaluation - LOOCV (Random Forest Regressor)")
                st.write(f"**Average R²:** {avg_r2:.2f}")
                st.write(f"**Average RMSE:** {avg_rmse:.2f} (KES)")
                st.write(f"**Average MAE:** {avg_mae:.2f} (KES)")
                st.write(f"**Average MAPE:** {avg_mape:.2f}%")
                
                best_model_idx = np.argmax(cv_results['test_r2'])
                best_model = cv_results['estimator'][best_model_idx]
                predictions_train = best_model.predict(X)
                residuals = y - predictions_train
                fig_res_rf, ax_res_rf = plt.subplots()
                ax_res_rf.scatter(predictions_train, residuals, color='purple', alpha=0.6)
                ax_res_rf.axhline(0, color='red', linestyle='--')
                ax_res_rf.set_xlabel("Predicted Values")
                ax_res_rf.set_ylabel("Residuals")
                ax_res_rf.set_title("Residual Plot (Random Forest)")
                st.pyplot(fig_res_rf)
                
                st.subheader("Price Prediction for Next 5 Years (2026-2030) with Random Forest")
                future_years = range(2026, 2031)
                future_data = pd.DataFrame({'Year': future_years})
                for feature in feature_columns:
                    future_data[feature] = data[feature].mean()
                X_future = future_data[feature_columns]
                future_predictions = best_model.predict(X_future)
                future_data['Predicted Price'] = future_predictions
                
                fig_future_rf = px.line(data, x='Year', y=target_variable, 
                                        title=f'Historical Prices vs. Predicted Prices for {target_variable} (Random Forest)',
                                        labels={'Year': 'Year', target_variable: 'Historical Price (KES)'})
                future_trace = px.line(future_data, x='Year', y='Predicted Price', color_discrete_sequence=['red']).data[0]
                future_trace.name = 'Predicted Price (Next 5 Years - RF)'
                fig_future_rf.add_trace(future_trace)
                fig_future_rf.update_layout(yaxis_title='Price (KES)')
                st.plotly_chart(fig_future_rf, use_container_width=True)
            
            # ------------------ Prophet ------------------ #
            elif model_type == "Prophet":
                prophet_df = data[['Year', target_variable]].copy()
                prophet_df['ds'] = pd.to_datetime(prophet_df['Year'].astype(str) + '-01-01')
                prophet_df.rename(columns={target_variable: 'y'}, inplace=True)
                
                m = Prophet(yearly_seasonality=True)
                m.fit(prophet_df[['ds', 'y']])
                
                future = m.make_future_dataframe(periods=5, freq='Y')
                forecast = m.predict(future)
                
                st.subheader("Historical vs. Predicted Prices using Prophet")
                fig_prophet = px.line(title=f'Historical and Future Predictions for {target_variable} (Prophet)')
                fig_prophet.add_scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='lines', name='Historical')
                fig_prophet.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast')
                fig_prophet.update_layout(xaxis_title='Year', yaxis_title='Price (KES)')
                st.plotly_chart(fig_prophet, use_container_width=True)
                
                forecast_train = m.predict(prophet_df[['ds']])
                residuals_prophet = prophet_df['y'] - forecast_train['yhat']
                fig_res_prophet, ax_prophet = plt.subplots()
                ax_prophet.scatter(prophet_df['ds'], residuals_prophet, color='green', alpha=0.6)
                ax_prophet.axhline(0, color='red', linestyle='--')
                ax_prophet.set_xlabel("Date")
                ax_prophet.set_ylabel("Residuals")
                ax_prophet.set_title("Residual Plot (Prophet)")
                st.pyplot(fig_res_prophet)
                
    st.markdown("---")
    if st.checkbox("Show Raw Data"):
        st.dataframe(data)

if __name__ == "__main__":
    main()
