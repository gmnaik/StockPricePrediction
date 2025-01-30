from dash import Dash,html,dcc,State, ctx
from datetime import date,datetime,timedelta

import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from dash.dependencies import Input,Output

from model import forecast_price

today = datetime.today()
sixty_days_before = today - timedelta(days=60)
yesterday = today - timedelta(days=1)
sixty_days_before = yesterday - timedelta(days=60)
# Format it as a string
today_str = today.strftime("%Y-%m-%d")
sixty_days_before_str = sixty_days_before.strftime("%Y-%m-%d")
yesterday_str = yesterday.strftime("%Y-%m-%d")

# User defined function to generate stock price graph

def get_stock_price_fig(df):
   print("Inside get_stock_price_fig: \n",df)
   
   df.columns = df.columns.droplevel('Ticker')
   
   fig = px.line(df,x= 'Date',y= ['Open','Close'], title="Closing and Opening Price vs Date")
   return fig

def get_indicators_fig(df):
   print("Inside get_indicators_fig: \n",df)
   df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
   fig = px.scatter(df,x="Date",y="EWA_20",title="Exponential Moving Average vs Date")
   fig.update_traces(mode='lines')  # We use 'lines' to plot a continuous line
   return fig

nav = html.Div(
 [
 html.H2("Welcome to the Stock Dash App!", className="start-heading"),
 
 # stock code input
 html.Div([
 html.P("Input stock code:",className="stockcode-heading"),
 html.Div(
    [dcc.Input(id='stock-id',placeholder='',type='text',value=''),
    html.Button('Submit',id='submit-button',n_clicks=0)],className="stockid-holder")
 ],className="stockcode-input"),
 
 # Date range picker input
 html.Div([
    html.Div(
        html.Div(dcc.DatePickerRange(id='my-date-picker-range',min_date_allowed=date(2020, 1, 1),max_date_allowed=today_str,end_date=yesterday_str))
 )
 ],className="start-date-parent-div"),
 
 # Stock price button
 # Indicators button
 # Number of days of forecast input
 # Forecast button
 html.Div([
   html.Div(
      [
         html.Button('Stock Price', id='stock-price-button'),
         html.Button('Indicators', id='indicator-button')
      ],className= "stock-indicator-button-div"
   ),
   html.Div(
      [
         dcc.Input(id='no-of-days',placeholder='Number of days',type='number',value=''),
         html.Button('Forecast', id='forecast-button')
      ],className= "noofdays-forecast-div" 
   )
 ],className="stockindicator-forecast-parentdiv"),
 ],
 className="nav")
 
content = html.Div(
 [
 html.Div(
 [  # Logo
 # Company Name
 html.H1(id='company-name')
 ],
 className="header"),
 html.Div( #Description
 id="description", className="decription_ticker"),
 html.Div([
 # Stock price plot
 ], id="stockpriceplot-content"),
html.Div([
 # Indicator plot
 ], id="indicatorplot-content"),
 html.Div([
 # Forecast plot
 ], id="forecastplot-content")
 ],
 className="content")
 
app = Dash(__name__)
server = app.server
 
app.layout = html.Div([nav,content],className="container")

# Define the callback
@app.callback(
    # 1. Outputs related to stock name and description
    Output(component_id='company-name', component_property='children'),
    Output(component_id='description', component_property='children'),
    
    # 2. Outputs related to stock price graph
    Output(component_id='stockpriceplot-content', component_property='children'),
    
    # 3. Outputs related to indicators price graph
    Output(component_id='indicatorplot-content', component_property='children'),
    
    # 4. Outputs related to forecast price graph
    #Output(component_id='forecastplot-content', component_property='children'),
    
    # 1. Inputs related to stock name and description
    Input('submit-button', 'n_clicks'),
    State(component_id='stock-id', component_property='value'),
    
    # 2. Inputs related to stock price graph
    Input('stock-price-button', 'n_clicks'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    
    # 3. Inputs related to stock price graph
    Input('indicator-button', 'n_clicks'),
    
    # 4. Inputs related to forecast price graph
    Input('forecast-button', 'n_clicks'),
    State(component_id='no-of-days', component_property='value')
)



def update_output_div(submit_button, input_value, stockprice_button, start_date, end_date, indicator_button,forecast_button,days_input):
   shortname = ""
   long_BusinessSummary = ""
   graph = html.Div()
   indicatorgraph = html.Div()
   
   if "submit-button" == ctx.triggered_id:
      ticker = yf.Ticker(str(input_value))
      inf = ticker.info
      
      shortname = inf['shortName']
      long_BusinessSummary = inf['longBusinessSummary']
      
   elif "stock-price-button" == ctx.triggered_id:
      ticker = yf.Ticker(str(input_value))
      inf = ticker.info
      
      shortname = inf['shortName']
      long_BusinessSummary = inf['longBusinessSummary']
      
      stockprice_df = yf.download(str(input_value), str(start_date), str(end_date) )
      stockprice_df.reset_index(inplace=True)
      graph = dcc.Graph(figure=get_stock_price_fig(stockprice_df))
   
   elif "indicator-button" == ctx.triggered_id:
      ticker = yf.Ticker(str(input_value))
      inf = ticker.info
      
      shortname = inf['shortName']
      long_BusinessSummary = inf['longBusinessSummary']
      
      stockprice_df = yf.download(str(input_value), str(start_date), str(end_date) )
      stockprice_df.reset_index(inplace=True)
      graph = dcc.Graph(figure=get_stock_price_fig(stockprice_df))
      
      indicatorgraph = dcc.Graph(figure=get_indicators_fig(stockprice_df))
   
   elif "forecast-button" == ctx.triggered_id:
      ticker = yf.Ticker(str(input_value))
      inf = ticker.info
      
      shortname = inf['shortName']
      long_BusinessSummary = inf['longBusinessSummary']
      
      onefifty_days_before_stockprice = yesterday - timedelta(days=10000)
      
      onefifty_days_before_stockprice_str = onefifty_days_before_stockprice.strftime("%Y-%m-%d")
      #yesterday_str = yesterday.strftime("%Y-%m-%d")
      
      stockprice_forecast = yf.download(str(input_value), str(onefifty_days_before_stockprice_str), str(yesterday_str) )
      vix_data = yf.download('^VIX', str(onefifty_days_before_stockprice_str), str(yesterday_str) )
      stockprice_forecast.reset_index(inplace=True)
      stockpricegraph = forecast_price(stockprice_forecast,vix_data)
      print("stockpricegraph",stockpricegraph)
   else:
      pass
      
   return shortname,long_BusinessSummary,graph,indicatorgraph
    
if __name__ == '__main__':
 app.run_server(debug=True)