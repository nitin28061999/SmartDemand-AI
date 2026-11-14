"""Recruiter-facing SmartDemand AI dashboard."""

import pandas as pd
import plotly.express as px
import streamlit as st
from api_client import SmartDemandAPI, SmartDemandAPIError

st.set_page_config(
    page_title="SmartDemand AI | Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background: #0b1120;
        color: #e5e7eb;
    }
    [data-testid="stSidebar"] {
        background: #111827;
    }
    [data-testid="stMetric"] {
        background: #172338;
        border: 1px solid #26364d;
        padding: 18px;
        border-radius: 12px;
    }
    .hero {
        background: linear-gradient(110deg, #14243d, #173b52);
        padding: 28px;
        border-radius: 16px;
        border: 1px solid #29445d;
        margin-bottom: 24px;
    }
    .hero h1 {
        color: #f8fafc;
        margin-bottom: 5px;
    }
    .hero p {
        color: #b8c9dc;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_api() -> SmartDemandAPI:
    """Create one reusable API client."""
    return SmartDemandAPI()


api = get_api()


with st.sidebar:
    st.title("📈 SmartDemand AI")
    st.caption("Retail Demand Forecasting Platform")
    st.divider()

    st.subheader("Production Status")

    try:
        health = api.health_check()

        if health.get("status") == "ok" and health.get("model_loaded"):
            st.success("Production API: Online")
            st.success("Forecasting model: Loaded")
        else:
            st.warning("API is responding, but the model is not ready.")

    except SmartDemandAPIError as error:
        st.error("Production API: Unavailable")
        st.caption(str(error))
        if error.__cause__:
            st.caption(
                f"Diagnostic: {type(error.__cause__).__name__}: "
                f"{error.__cause__}"
            )

    st.divider()
    st.caption("Powered by FastAPI, XGBoost and AWS")
    st.caption("SmartDemand AI | MLOps Portfolio Project")


st.markdown(
    """
    <div class="hero">
        <h1>SmartDemand AI</h1>
        <p>Intelligent retail demand forecasting powered by machine learning.</p>
        <p>Generate future sales predictions for individual stores and products.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


st.subheader("Generate Demand Forecast")
st.write(
    "Choose a store, product, and prediction horizon to generate "
    "forecasts using the deployed production model."
)


with st.form("forecast_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        store = st.selectbox(
            "Store ID",
            options=list(range(1, 11)),
            index=1,
        )

    with col2:
        item = st.selectbox(
            "Item ID",
            options=list(range(1, 51)),
            index=0,
        )

    with col3:
        horizon = st.slider(
            "Forecast horizon (days)",
            min_value=1,
            max_value=90,
            value=7,
        )

    submitted = st.form_submit_button(
        "Generate Forecast",
        type="primary",
        use_container_width=True,
    )


if submitted:
    if not api.api_key:
        st.error(
            "Forecasting credentials are not configured. "
            "The API key must be configured securely on the server."
        )
    else:
        with st.spinner("Generating predictions from the production model..."):
            try:
                result = api.forecast(
                    store=store,
                    item=item,
                    horizon_days=horizon,
                )

                predictions = result.get("predictions", [])

                if not predictions:
                    st.warning("The API returned no predictions.")
                else:
                    df = pd.DataFrame(predictions)

                    required_columns = {"date", "predicted_sales"}

                    if not required_columns.issubset(df.columns):
                        st.error("The API returned an unexpected response format.")
                    else:
                        df["date"] = pd.to_datetime(df["date"])
                        df["predicted_sales"] = pd.to_numeric(
                            df["predicted_sales"],
                            errors="raise",
                        )

                        st.session_state["forecast_result"] = result
                        st.session_state["forecast_df"] = df

            except SmartDemandAPIError as error:
                st.error(str(error))

            except (ValueError, TypeError) as error:
                st.error(
                    f"Unable to process the forecast response: {error}"
                )


if "forecast_df" in st.session_state:
    df = st.session_state["forecast_df"]
    result = st.session_state["forecast_result"]

    st.divider()
    st.subheader("Forecast Results")

    st.caption(
        f"Store {result['store']} · "
        f"Item {result['item']} · "
        f"{result['horizon_days']}-day forecast"
    )

    total_sales = df["predicted_sales"].sum()
    average_sales = df["predicted_sales"].mean()
    minimum_sales = df["predicted_sales"].min()
    maximum_sales = df["predicted_sales"].max()

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Total Forecast", f"{total_sales:,.2f} units")
    k2.metric("Daily Average", f"{average_sales:,.2f} units")
    k3.metric("Minimum Daily Demand", f"{minimum_sales:,.2f} units")
    k4.metric("Peak Daily Demand", f"{maximum_sales:,.2f} units")

    st.subheader("Predicted Sales Trend")

    fig = px.line(
        df,
        x="date",
        y="predicted_sales",
        markers=True,
        template="plotly_dark",
        labels={
            "date": "Forecast Date",
            "predicted_sales": "Predicted Sales (units)",
        },
        title="Daily Demand Forecast",
    )

    fig.update_traces(
        line={"width": 3},
        marker={"size": 8},
        hovertemplate=(
            "Date: %{x|%d %b %Y}<br>"
            "Predicted sales: %{y:.2f} units"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        paper_bgcolor="#0b1120",
        plot_bgcolor="#111827",
        font_color="#e5e7eb",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily Prediction Details")

    display_df = df.copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    display_df["predicted_sales"] = display_df["predicted_sales"].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "date": st.column_config.DateColumn("Forecast Date"),
            "predicted_sales": st.column_config.NumberColumn(
                "Predicted Sales (units)",
                format="%.2f",
            ),
        },
    )

    csv_data = display_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Forecast CSV",
        data=csv_data,
        file_name=f"smartdemand_store_{result['store']}_item_{result['item']}.csv",
        mime="text/csv",
    )

else:
    st.info(
        "Select your store, item, and forecast horizon, "
        "then click Generate Forecast to see predictions."
    )


st.divider()

st.caption(
    "SmartDemand AI | Machine Learning · FastAPI · XGBoost · "
    "Docker · AWS EC2 · Amazon S3 · Prometheus · Grafana · CloudWatch"
)
