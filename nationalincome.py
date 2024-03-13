import streamlit as st
import plotly.express as px
import pandas as pd
import os
import matplotlib.pyplot as pltt
import warnings
import xgboost as xgb
import pandas as pd
import numpy as np
import seaborn as sn
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')
color_pal=sn.color_palette()
st.set_page_config(page_title="AGRIBUSINESS STORE ANALYTICS", page_icon=":bar_chart:",layout = "wide")
st.title(":bar_chart: BILLING SYSTEM AND ANALYSIS OF AGRICULTURAL PRODUCE")
st.markdown('<style>div.block-container{padding-top:1rem}</style>',unsafe_allow_html=True)

f1 = st.file_uploader(":file_folder: upload a file",type =(["csv","xlsx"]))
if f1 is not None:
    filename = f1.name
    st.write(filename)
    df = pd.read_excel(filename)
else:
    os.chdir(r"C:/Users/NEWTON SHIVACHI/OneDrive/Desktop/project")
    df = pd.read_excel("agribizdata1.xlsx")
col1,col2 = st.columns((2))
df["Date"] = pd.to_datetime(df["Date"])

start_date = pd.to_datetime(df["Date"]).min()
end_date = pd.to_datetime(df["Date"]).max()
with col1:
    date1 = pd.to_datetime(st.date_input("Startdate",start_date))
with col2:
    date2 = pd.to_datetime(st.date_input("Enddate",end_date))

df = df[(df["Date"]>= date1)&(df["Date"]<=date2)].copy()
st.sidebar.header("Choose your filter: ")
store = st.sidebar.multiselect("Select your store",df["Store"].unique())
if not store:
    df1 = df.copy
else:
    df1 =df[df["Store"].isin(store)]
product = st.sidebar.multiselect("Select your a product", df1["Product"].unique())
if not product:
    df2 = df1.copy()
else:
    df2 = df1[df1["Product"].isin(product)]

if not store and not product:
    filtered_df = df
elif not product:
    filtered_df = df[df["Store"].isin(store)]
elif not store:
    filtered_df = df[df["Product"].isin(product)]
elif store and product:
    filtered_df = df2[df["Product"].isin(product)&df["Store"].isin(store)]
store_df = filtered_df.groupby(by = ["Store"], as_index = False)["Sales"].sum()
with col1:
    st.subheader("Storewise Sales")
    fig = px.bar(store_df, x = "Store", y= "Sales", text = ['${:,.2f}'.format(x) for x in store_df["Sales"]],
                template= "seaborn")
    st.plotly_chart(fig,use_container_width = True, height = 200)
with col2:
    st.subheader("Payment means")
    fig = px.pie(filtered_df, values = "Sales", names = "Payment",hole = 0.5)
    fig.update_traces(text = filtered_df["Payment"], textposition = "outside")
    st.plotly_chart(fig, use_container_width=True)
cl1, cl2 = st.columns((2))
with cl1:
    with st.expander("View your store sales"):
        st.write(store_df.style.background_gradient(cmap="Blues"))
        csv = store_df.to_csv(index = False).encode('utf-8')
        st.download_button("Download Data", data= csv,file_name = "store.csv",mime = "text/csv",
                           help = 'click here to download the data as csv file')
with cl2:
    with st.expander("View_your_payment_means"):
        payment = filtered_df.groupby(by = "Payment", as_index = False)["Sales"].sum()
        st.write(payment.style.background_gradient(cmap="Oranges"))
        csv = payment.to_csv(index = False).encode('utf-8')
        st.download_button("Download Data", data= csv,file_name = "payment.csv",mime = "text/csv",
                           help = 'click here to download the data as csv file')

filtered_df["month_year"] = filtered_df["Date"].dt.to_period("M")
st.subheader("Timeseries Analysis")

linechart = pd.DataFrame(filtered_df.groupby(filtered_df["month_year"].dt.strftime("%Y:%b"))["Sales"].sum()).reset_index()
fig2 = px.line(linechart, x = "month_year", y = "Sales", labels = {"Sales":"Amount"},height = 500, width= 1000,template ="gridon")
st.plotly_chart(fig2, use_container_width = True)

with st.expander("View Time Series Data"):
    st.write(linechart.T.style.background_gradient(cmap="Blues"))
    csv = linechart.to_csv(index = False).encode("utf-8")
    st.download_button("Download Data", data = csv, file_name = "TimeSeries.csv",mime = "text/csv")

st.subheader("TreeMap")
fig3 = px.treemap(filtered_df,path=["Store","Product"], values = "Sales",hover_data = ["Sales"],
                  color = "Product")
fig3.update_layout(width = 800, height = 650)
st.plotly_chart(fig3,use_container_width = True)

import plotly.figure_factory as ff
st.subheader(":point_right: Monthwise product Sales")
with st.expander("Summary_Table"):
    df_sample = df[0:2][["Store","Product"]]
    fig = ff.create_table(df_sample,colorscale = "cividis")
    st.plotly_chart(fig, use_container_width = True)
    st.markdown("Monthwise Product Table")
    filtered_df ["month"] = filtered_df ["Date"].dt.month_name()
    product_year = pd.pivot_table(data = filtered_df, values = "Sales",index = ["Product"],columns ="month")
    st.write(product_year.style.background_gradient(cmap = "Blues"))
st1, st2= st.columns((2))
with st1:
    with st.expander("View store sales Mean"):
        sales = filtered_df.groupby(by ="Store", as_index = False)["Sales"].mean()
        st.write(sales)

with st2:
    with st.expander("View store sales Variance"):
        sales = filtered_df.groupby(by ="Store", as_index = False)["Sales"].var()
        st.write(sales)
st3, st4 = st.columns((2))
with st3:
    with st.expander("View Store Maximum Sales"):
        sales =  filtered_df.groupby(by ="Store", as_index = False)["Sales"].max()
        st.write(sales)    
with st4:
    with st.expander("View Store Minimum Sales"):
        sales =  filtered_df.groupby(by ="Store", as_index = False)["Sales"].min()
        st.write(sales) 
st5, st6 = st.columns((2))
with st5:
    with st.expander("Boxplot for Stores"):
        fg = px.box(filtered_df, x = "Store", y = "Sales")
        st.plotly_chart(fg, use_container_width= True)
 
st1, st2= st.columns((2))
with st1:
    with st.expander("View product mean sales"):
        sales = filtered_df.groupby(by ="Product", as_index = False)["Sales"].mean()
        st.write(sales)
with st2:
    with st.expander("Boxplot for products"):
        fg = px.box(filtered_df, x = "Product", y = "Sales")
        st.plotly_chart(fg, use_container_width= True)
        
st1, st2= st.columns((2))
with st1:
    with st.expander("View product sales variance"):
        sales = filtered_df.groupby(by ="Product", as_index = False)["Sales"].var()
        st.write(sales)
with st2:
    with st.expander("View product maximum sales"):
        sales = filtered_df.groupby(by ="Product", as_index = False)["Sales"].max()
        st.write(sales)
st1, st2= st.columns((2))
with st1:
    with st.expander("View product minimum sales"):
        sales = filtered_df.groupby(by ="Product", as_index = False)["Sales"].min()
        st.write(sales)
with st2:
    with st.expander("View product total sales"):
        sales = filtered_df.groupby(by ="Product", as_index = False)["Sales"].sum()
        st.write(sales)
st1, st2= st.columns((2))
with st1:
    with st.expander("View store sales Mean"):
        sales = filtered_df.groupby(by ="Product", as_index = False)["Quantity"].mean()
        st.write(sales)

with st2:
    with st.expander("View store sales Variance"):
        sales = filtered_df.groupby(by ="Product", as_index = False)["Quantity"].sum()
        st.write(sales)
filtered=filtered_df[["Date","Sales"]] 
filtered["Date"]=pd.to_datetime(filtered["Date"],format="%d-%m-%Y")
filtered["Date"]=pd.to_datetime(filtered["Date"],errors="coerse")
filtered["Date"]=pd.to_datetime(filtered["Date"],format="%d-$m-%Y %H:%M:%S")
filtered=filtered.set_index("Date")
filtered.index= pd.to_datetime(filtered.index)

train= filtered.loc[filtered.index <"3/3/2017"]
test= filtered.loc[filtered.index>="3/3/2017"]
with st.expander("View sales model plot"):
    fig, ax=pltt.subplots(figsize=(15,5))
    train= filtered.loc[filtered.index <"3/3/2017"]
    test= filtered.loc[filtered.index>="3/3/2017"]
    train.plot(ax=ax)
    test.plot(ax=ax)
    st.write(fig)
with st.expander("View sales model plot"):   
    dat=filtered.loc[(filtered.index >"01/01/2017") & (filtered.index<="1/12/2018")]
    st.write(dat)
    fig,ax=pltt.subplots(figsize=(15,5))
    dat.plot(ax=ax)
    st.write(fig)
    da=filtered.index.dayofweek
    st.write(da)
    def create_fetures(filtered):
        filtered=filtered.copy()
        filtered["dayofweek"]=filtered.index.dayofweek
        filtered["month"]=filtered.index.month
        filtered["quater"]=filtered.index.quarter
        filtered["dayofyear"]=filtered.index.dayofyear
        return filtered
    filtered=create_fetures(filtered)
    fi1=px.box(filtered,x="dayofweek",y="Sales")
    st.write(fi1)
    fi11=px.box(filtered,x="month",y="Sales")
    st.write(fi11)
    fi111=px.box(filtered,x="quater",y="Sales")
    st.write(fi111)
    fi111=px.box(filtered,x="dayofyear",y="Sales")
    st.write(fi111)
    train=create_fetures(train)
    test=create_fetures(test)
    FEATURES=["dayofweek","month","quater","dayofyear"]
    TARGET=["Sales"]
    x_train=train[FEATURES]
    y_train=train[TARGET]
    x_test=test[FEATURES]
    y_test=test[TARGET]
    reg=XGBRegressor(n_estimator=0,early_stopping_rounds=1000,
                     learning_rate=0.9)
    st.write(reg)
    reg1=reg.fit(x_train,y_train,
                 eval_set=[(x_train,y_train),(x_test,y_test)],
                 verbose=100)
    st.write(reg1)
    reg2=pd.DataFrame(data=reg.feature_importances_,
                      index=reg.feature_names_in_,
                      columns=["importance"])
    st.write(reg2)
    reg3=reg.predict(x_test)
    st.write(reg3)
    test["prediction"]=reg3
    reg4=filtered.merge(test[["prediction"]],how="left",left_index=True,right_index=True)
    st.write(reg4)
    fig,ax=pltt.subplots(figsize=(15,5))
    take=reg4["prediction"]
    take.plot(ax=ax)
    train["Sales"].plot(ax=ax)
    st.write(fig)
    fig,ax=pltt.subplots(figsize=(15,5))
    take=reg4["prediction"]
    take.plot(ax=ax)
    test["Sales"].plot(ax=ax)
    st.write(fig)
    fig,ax=pltt.subplots(figsize=(15,5))
    take=reg4["prediction"]
    take.plot(ax=ax)
    st.write(fig)
    error=np.sqrt(mean_squared_error(test["Sales"],test["prediction"]))
    st.write(error)
