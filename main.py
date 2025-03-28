from darts.models import AutoARIMA, ExponentialSmoothing, Prophet
from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values
from fastapi import FastAPI
from datetime import datetime
from sklearn.cluster import KMeans, MeanShift
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import pandas as pd


app = FastAPI()
# ! Should this be opened and closed for each request?
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
conn = psycopg2.connect('postgresql://postgres:postgres@localhost:5432/super')

def format_series_data(db_response, connector_cursor):
    column_names = [desc[0] for desc in connector_cursor.description]
    df = pd.DataFrame(db_response, columns=column_names)
    df['date'] = df['date'].astype(str)

    return df

def complete_missing_data(df_raw_data):
    # Fill missing dates with 0 quantity
    print(df_raw_data)
    date_start = str(df_raw_data['date'].min())
    date_end = datetime.now().strftime('%Y-%m-%d')

    all_dates = pd.date_range(start=date_start, end=date_end, freq='D')
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

@app.get("/predict_category_sales/{cat_id}")
def predict_sales(cat_id: int):
    with conn.cursor() as cur:
        cur.execute(f"""select DATE(sale.created_at) as date, sum(sale_line.quantity) as quantity 
                    from sale_line 
                    inner join product_has_category
                    on product_has_category.product_id = sale_line.product_id
                    inner join sale
                    on sale.id = sale_line.sale_id
                    where product_has_category.category_id = {cat_id} and (sale.created_at > (now() - interval '3 years'))
                    group by DATE(sale.created_at);""")
        db_response = cur.fetchall()
        cur.close()
        raw_data = format_series_data(db_response, cur)


        aarima_forecast = autoARIMA_model(raw_data)
        exp_smoothing_forecast = exp_smoothing_model(raw_data)
        prophet_forecast = prophet_model(raw_data)
    

    return {
        "auto_arima": convert_to_dict_round_values(aarima_forecast),
        "exp_smoothing": convert_to_dict_round_values(exp_smoothing_forecast),
        "prophet": convert_to_dict_round_values(prophet_forecast)
    }

@app.get("/client_retention")
def client_retention():
#   graph que muestre por cliente x=num compras; y=monto con cuadrantes en color... 
#       (permite ver si los clientes gralmente vuelven a hacer compras (RETENCION DEL CLIENTE))
#           -deberia agarrar el la media matematica o la mediana???

    with conn.cursor() as cur:
        cur.execute(f"""SELECT count(*), client_id, sum(sl.quantity*sl.unit_price) as spent, max(sl.quantity*sl.unit_price) as max_spent
                    , min(sl.quantity*sl.unit_price) as min_spent, max(s.created_at) as last_sale
                    from sale s
                    inner join sale_line sl
                        on s.id= sl.sale_id
                    group by client_id;""")
        db_response = cur.fetchall()
        cur.close()
    column_names = [desc[0] for desc in cur.description]
    db_response = [dict(zip(column_names, row)) for row in db_response]
    return db_response

@app.get("/sales_behaviour")
def sales_behaviour():
#   analizando CADA COMPRA (COMPORTAMIENTO DE LA DEMANDA)

    with conn.cursor() as cur:
        cur.execute(f"""SELECT count(sl.id) as item_amount, s.id as sale_id, sum(sl.quantity*sl.unit_price) as sale_total
                    from sale s
                    inner join sale_line sl
                    on s.id= sl.sale_id
                    group by s.id;""")
        db_response = cur.fetchall()
        cur.close()
    column_names = [desc[0] for desc in cur.description]
    db_response = [dict(zip(column_names, row)) for row in db_response]

    return db_response

@app.get("/client_clusters")
def client_clusters():
# should I be doing the clustering with the client_id?

    with conn.cursor() as cur:
        cur.execute(f"""select client_id, category_id, sum(quantity) as quantity 
                    from sale_line
                    inner join product_has_category
                    on sale_line.product_id = product_has_category.product_id
                    inner join sale
                    on sale_line.sale_id = sale.id
                    where sale.created_at > (now() - interval '1 years')
                    group by client_id, category_id;""")
        ticket_x_category_quantity = cur.fetchall()
        df_ticket_x_category_quantity = pd.DataFrame(ticket_x_category_quantity, columns=['client_id', 'category_id', 'quantity'])
        cur.close()
    
    with conn.cursor() as cur:
        cur.execute(f"""select id, name from category order by id asc;""")
        all_categories = cur.fetchall()
        df_all_categories = pd.DataFrame(all_categories, columns=['category_id', 'category_name'])
        cur.close()
    
    cols= df_all_categories['category_id'].tolist()
    col_names = df_all_categories['category_name'].tolist()
    df_merged = pd.DataFrame(columns=cols, index=df_ticket_x_category_quantity['client_id'].unique())

    for i in range(len(ticket_x_category_quantity)):
        row= df_ticket_x_category_quantity.iloc[i]
        df_merged.loc[row['client_id'], row['category_id']] = row['quantity']
    df_merged = df_merged.fillna(0)

    scaler = StandardScaler()
    df_merged_scaled = scaler.fit_transform(df_merged)

    inertia = []
    for k in range(1, 12):
        kmeans = KMeans(n_clusters=k, random_state=1337)
        kmeans.fit(df_merged_scaled)
        inertia.append(kmeans.inertia_)
    
    inertia_diff = [inertia[i-1] - inertia[i] for i in range(1, len(inertia))]
    optimal_k = inertia_diff.index(max(inertia_diff)) + 3 

    kmeans = KMeans(n_clusters=optimal_k, random_state=1337)
    clusters = kmeans.fit_predict(df_merged)

    df_merged.columns = col_names
    df_merged['cluster'] = clusters

    cluster_averages = df_merged.groupby('cluster').mean().round(0)

    return {
        "clusters_data":cluster_averages.to_dict('index')
    }

@app.get("/sales_clusters")
def sales_clusters():
    with conn.cursor() as cur:
        cur.execute(f"""select sale_id, category_id, sum(quantity) as quantity 
                    from sale_line
                    inner join product_has_category
                    on sale_line.product_id = product_has_category.product_id
                    inner join sale
                    on sale_line.sale_id = sale.id
                    where sale.created_at > (now() - interval '1 years')
                    group by sale_id, category_id;""")
        ticket_x_category_quantity = cur.fetchall()
        df_ticket_x_category_quantity = pd.DataFrame(ticket_x_category_quantity, columns=['sale_id', 'category_id', 'quantity'])
        cur.close()
    
    with conn.cursor() as cur:
        cur.execute(f"""select id, name from category;""")
        all_categories = cur.fetchall()
        df_all_categories = pd.DataFrame(all_categories, columns=['category_id', 'category_name'])
        cur.close()
    
    cols= df_all_categories['category_id'].tolist()
    col_names = df_all_categories['category_name'].tolist()
    df_merged = pd.DataFrame(columns=cols, index=df_ticket_x_category_quantity['sale_id'].unique())

    for i in range(len(ticket_x_category_quantity)):
        row= df_ticket_x_category_quantity.iloc[i]
        df_merged.loc[row['sale_id'], row['category_id']] = row['quantity']
    df_merged = df_merged.fillna(0)

    scaler = StandardScaler()
    df_merged_scaled = scaler.fit_transform(df_merged)

    inertia = []
    for k in range(1, 12):
        kmeans = KMeans(n_clusters=k, random_state=1337)
        kmeans.fit(df_merged_scaled)
        inertia.append(kmeans.inertia_)
    print(inertia)
    inertia_diff = [inertia[i-1] - inertia[i] for i in range(1, len(inertia))]
    optimal_k = inertia_diff.index(max(inertia_diff)) + 3 

    kmeans = KMeans(n_clusters=optimal_k, random_state=1337)
    clusters = kmeans.fit_predict(df_merged)

    df_merged.columns = col_names
    df_merged['cluster'] = clusters

    cluster_averages = df_merged.groupby('cluster').mean().round(0)

    return {
        "clusters_data":cluster_averages.to_dict('index')
    }