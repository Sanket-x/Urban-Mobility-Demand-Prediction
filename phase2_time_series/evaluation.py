"""
================================================================================
STEP 6: EVALUATION METRICS & VISUALIZATION
================================================================================
Purpose:
    - Compare predicted values to actual test-set values using standard metrics.
    - Visualize the results to assess model fit.

Metrics Used:
    1. MAE (Mean Absolute Error)
       - Average of absolute differences between forecast and actual.
       - Highly interpretable (same units as target).

    2. RMSE (Root Mean Squared Error)
       - Penalizes large errors heavily (due to squaring).
       - Useful for identifying if a model frequently produces large spikes in error.

    3. MAPE (Mean Absolute Percentage Error)
       - Evaluates accuracy as a percentage.
       - Warning: Can explode if actual values are near zero.

Interpretation & Justification:
    A good model balances low MAE/RMSE while capturing the shape (seasonality)
    in the visual plot. If a model fits the data physically but has slightly
    worse metrics due to a minor phase shift, it might still be preferred over
    a flat-line model that happens to have lower variance.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf

sns.set_style("whitegrid")
plt.rcParams.update({"figure.figsize": (14, 5)})

def calculate_metrics(y_true: pd.Series, y_pred: pd.Series, model_name: str) -> dict:
    """
    Calculate MAE, RMSE, MAPE, and R2.

    Parameters
    ----------
    y_true : pd.Series
        Actual observed values.
    y_pred : pd.Series
        Forecasted values.
    model_name : str
        Identifier for the model.

    Returns
    -------
    dict
        Dictionary containing metric scores.
    """
                                               
    common_idx = y_pred.index.intersection(y_true.index)
    y_true_align = y_true.loc[common_idx]
    y_pred_align = y_pred.loc[common_idx]

    if len(common_idx) == 0:
        return {"Model": model_name, "MAE": np.nan, "RMSE": np.nan, "MAPE (%)": np.nan, "R2": np.nan}

    mae = mean_absolute_error(y_true_align, y_pred_align)
    rmse = np.sqrt(mean_squared_error(y_true_align, y_pred_align))
    r2 = r2_score(y_true_align, y_pred_align)

    mape = np.mean(np.abs((y_true_align - y_pred_align) / (y_true_align + 1e-6))) * 100

    metrics = {
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape, 4),
        "R2": round(r2, 4)
    }
    return metrics

def display_metrics_table(results_list: list):
    """
    Print a formatted, academic-style comparison table of all tested models.
    """
    df_metrics = pd.DataFrame(results_list)
    print("\n" + "="*50)
    print(" STEP 7: MODEL EVALUATION & COMPARISON SUMMARY")
    print("="*50)
    print(df_metrics.to_string(index=False))
    print("="*50 + "\n")
    return df_metrics

def plot_forecast(train: pd.Series, test: pd.Series, forecast: pd.Series,
                  model_name: str, save_path: str | None = None):
    """
    Plot actual Train + Test against the Forecasted values.
    """
    fig, ax = plt.subplots(figsize=(15, 6))

    train_slice = train.tail(168)

    ax.plot(train_slice.index, train_slice.values, label="Train (last 7 days)", color="#34495E", alpha=0.7)
    ax.plot(test.index, test.values, label="Test (Actual)", color="#2ECC71", linewidth=2)
    ax.plot(forecast.index, forecast.values, label=f"Forecast ({model_name})",
            color="#E74C3C", linestyle="--", linewidth=2.5)

    ax.set_title(f"Demand Forecasting — {model_name} Model vs Actual", fontsize=14)
    ax.set_xlabel("Time (Hourly)", fontsize=12)
    ax.set_ylabel("Demand Count", fontsize=12)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  → Saved forecast plot to {save_path}")
    plt.close(fig)

def plot_residual_analysis(residuals: pd.Series, model_name: str, save_path: str = None):
    """
    Plot Residuals over time and their distribution perfectly vertically aligned.
    
    Academic Goal: Validating that residuals resemble white noise.
    - Residual vs. Time Plot: Checks for equal variance (homoscedasticity) and no leftover trend.
      A good model's residuals should have no visible pattern and hover randomly around zero.
    - Histogram: Validates that the errors are normally distributed (bell-curved) around 0.
    """
    residuals = residuals.dropna()
    mean_res = residuals.mean()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    axes[0].plot(residuals.index, residuals.values, color="#95A5A6", alpha=0.9, linewidth=1.5)
    axes[0].axhline(y=0, color="#E74C3C", linestyle="--", linewidth=2)
    axes[0].set_title(f"{model_name} Residuals vs. Time", fontsize=13)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Error")
    
    sns.histplot(residuals.values, kde=True, color="#3498DB", ax=axes[1])
    axes[1].axvline(x=0, color="#E74C3C", linestyle="--", linewidth=2, label="Zero Mean Reference")
    axes[1].axvline(x=mean_res, color="#2ECC71", linestyle=":", linewidth=2.5, label=f"Actual Mean ({mean_res:.2f})")
    axes[1].set_title(f"{model_name} Residuals Distribution", fontsize=13)
    axes[1].set_xlabel("Error Magnitude")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  → Saved residual analysis plot to {save_path}")
    plt.close(fig)

def plot_residual_acf(residuals: pd.Series, model_name: str, save_path: str = None):
    """
    Plot the ACF (Autocorrelation) of the residuals.
    
    Advanced Validation: 
    Even if residuals look normal, they must not be autocorrelated. 
    - If 95%+ of points fall tightly within the shaded blue confidence band, 
      it indicates NO significant spikes → The model perfectly extracted all timing patterns.
    - If massive spikes extend far outside the blue area, the model missed some patterns
      (and could theoretically be improved).
    """
    residuals = residuals.dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    
    plot_acf(residuals, lags=48, ax=ax, title=f"ACF of Residuals ({model_name})")
    ax.set_xlabel("Lags")
    ax.set_ylabel("Correlation")
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  → Saved residual ACF plot to {save_path}")
    plt.close(fig)
