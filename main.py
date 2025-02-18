from darts.models import AutoARIMA, ExponentialSmoothing, Prophet
from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values
from fastapi import FastAPI
import psycopg2
import pandas as pd

from datetime import datetime, timedelta

def generate_logical_sales_dataset():
    start_date = datetime(2023, 1, 1)
    
    # Create a pattern of increasing and decreasing sales
    base_pattern = [15, 22, 30, 40, 35, 25, 18, 12, 20, 28, 38, 45]
    
    data = []
    for month in range(12):
        month_start = start_date + timedelta(days=month*30)
        
        # Vary daily sales within a 20% range of the monthly base number
        for day in range(30):
            current_date = month_start + timedelta(days=day)
            quantity = max(1, base_pattern[month] + int((day - 15) * 0.5))
            
            data.append({
                'date': current_date,
                'quantity': quantity
            })
    print(pd.DataFrame(data))
    return pd.DataFrame(data)

app = FastAPI()
# ! Should this be opened and closed for each request?
conn = psycopg2.connect('postgresql://postgres:postgres@localhost:5432/super')

def format_series_data(db_response, connector_cursor):
    column_names = [desc[0] for desc in connector_cursor.description]
    df = pd.DataFrame(db_response, columns=column_names)
    df['date'] = df['date'].astype(str)

    return df

def complete_missing_data(df_raw_data):
    # Fill missing dates with 0 quantity
    all_dates = pd.date_range(start=df_raw_data['date'].min(), end=df_raw_data['date'].max(), freq='D')
    all_dates = pd.DataFrame(all_dates, columns=['date'])
    all_dates['date'] = all_dates['date'].dt.strftime('%Y-%m-%d')

    df_raw_data = pd.merge(all_dates, df_raw_data, on='date', how='left')
    df_raw_data['quantity'] = df_raw_data['quantity'].fillna(0)

    return df_raw_data

def prepare_df_for_model(df_raw_data):
    df_raw_data = complete_missing_data(df_raw_data)
    raw_data_series = TimeSeries.from_dataframe(df_raw_data, time_col='date', value_cols='quantity')
    return raw_data_series

def convert_to_dict_round_values(timeseries):
        df = timeseries.pd_dataframe().round(0)
        df.reset_index(inplace=True)
        df.columns = ['date', 'quantity']
        return df.to_dict('records')


# TODO 
    # - Auto-Hyperparameter tuning
    # - Cross-validation
    # ! Handle not enough data, etc
    # -Include multiple models and compare them to see which one is the best
    # -Include the model that is the best in the response ???

def autoARIMA_model(raw_data):
    data = prepare_df_for_model(raw_data)
    model = AutoARIMA(seasonal=True, m=7, stepwise=True, suppress_warnings=True)
    model.fit(data)
    return model.predict(7)


def exp_smoothing_model(raw_data):
    data = prepare_df_for_model(raw_data)
    model = ExponentialSmoothing(seasonal_periods=7)
    model.fit(data)
    return model.predict(7)

def prophet_model(raw_data):
    data = prepare_df_for_model(raw_data)
    model = Prophet(country_holidays="AR")
    model.fit(data)

    return model.predict(7)


@app.get("/predict_sales/{barcode}")
def predict_sales(barcode: int):
    with conn.cursor() as cur:
        cur.execute(f"""select DATE(sale.created_at) as date, sum(sale_line.quantity) as quantity 
                    from sale_line 
                    inner join sale 
                    on sale.id = sale_line.sale_id 
                    inner join product 
                    on product.id = sale_line.product_id 
                    where product.bar_code = {barcode} and (sale.created_at > (now() - interval '3 years'))
                    group by DATE(sale.created_at);""")
        db_response = cur.fetchall()

        raw_data = format_series_data(db_response, cur)

        aarima_forecast = autoARIMA_model(raw_data)
        exp_smoothing_forecast = exp_smoothing_model(raw_data)
        prophet_forecast = prophet_model(raw_data)
    

    return {
        "auto_arima": convert_to_dict_round_values(aarima_forecast),
        "exp_smoothing": convert_to_dict_round_values(exp_smoothing_forecast),
        "prophet": convert_to_dict_round_values(prophet_forecast)
    }

    
# conn.close()