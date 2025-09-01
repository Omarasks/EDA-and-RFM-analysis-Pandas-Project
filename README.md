# Case Study: EDA-and-RFM-Customer-Segmentation-analysis-Pandas-Project

## Project Overview 

**Project Title**: Customer Segmentation with RFM Analysis
**Dataset**: Retail-Transactions-Dataset.csv
**Source**: kaggle
**Tools*: Pandas 

My boss gave me this a case study project that demonstrates the implementation of Exploratory Data Analysis (EDA) to find sales trend patterns, calculate key business metrics. 
using the RFM analysis to segment customers based on purchasing behaviour to help identify the most valuable set of customers.

## Objectives 
1. **Data Preparation**: Import dataset and clean up fields with uneccessary extra string characters
2. **EDA**: Core business metrics, Sales performance and Customer behaviour
3. **Customer Segmentation**: RFM bucket for 3 class of customers 'Champion', 'New customers' and 'Churn'

## Project Structure 

### 1. Data Preparation
**Import data**: Imported data from the kaggle using pandas 

``` python
import os
import kagglehub
import pandas as pd

file_path = kagglehub.dataset_download("prasad22/retail-transactions-dataset")
data = file_path
data = os.path.join(data, "Retail_Transactions_Dataset.csv")

df = pd.read_csv(data)
display(df.head())
```
<img width="1408" height="444" alt="rfm-img" src="https://github.com/user-attachments/assets/b74a763f-e7ca-4564-98d1-2ab182123c76" />

**Data cleaning**: I wrote a function `ext_products` using python list comprehension and loop function to extract string items from the list of items and remove unwanted string characters "[]" and "''" from the `Products` column, This will enable us to perform analysis on each product sales
``` python
def ext_products(products):
  if isinstance(products, str):
    return [product.strip() for product in products.strip("[]").replace("'", "").split(", ") ]
  return [] 
```
create a product_item_list column and append all items in the product in the list to the new column
``` python
df['product_item_list'] = df['Product'].apply(ext_products)
df.head()
```
<img width="1429" height="384" alt="rfm-img-2" src="https://github.com/user-attachments/assets/2d401818-7390-449f-988a-1ecbe87e09bd" />

using the explode python function, I separated the list into a single cell enforcing the atomicity of a cell and also to perform more granular analysis on the Product field, to enforce this I created a new column `product_item_list`

``` python
product_list_explode = df.explode('product_item_list')
product_list_explode.head()
```
<img width="1417" height="505" alt="rfm-img-3" src="https://github.com/user-attachments/assets/915482a0-85d3-4fc3-8cb1-3e38a93e38c6" /> 

### 2. EDA for cor business metrics

**Task 1. What is the total revenue?**
``` python
new_total_revenue = df['Total_Cost'].sum()
print(new_total_revenue)
```
**Task 2. What is the total revenue per product, and which products are our top sellers?**
``` python
each_product_performance = product_list_explode.groupby('product_item_list').agg(
  sum_total_revenue = ('Total_Cost', 'sum'),
  sum_total_items_sold = ('Total_Items', 'sum')
).reset_index()
display(each_product_performance.head(10))

-- Top 10 products by sales performance
top_10_revenue = each_product_performance.sort_values('sum_total_revenue', ascending=False)
display(top_10_revenue.head(10))
```
<img width="520" height="353" alt="sales-perf" src="https://github.com/user-attachments/assets/cd2d6ff9-217a-46f7-98b2-6006b1d42573" />

**Task 3. How does revenue break down by city?** 
``` python
bpf_city = df.groupby('City').agg(
    cities_revenue = ('Total_Cost', 'sum')
).sort_values('cities_revenue', ascending=False)
display(bpf_city.head(10))
```

How does revenue break down by store type?
``` python
bpf_store_type = df.groupby('Store_Type').agg(
    total_rev_store_ty = ('Total_Cost', 'sum')
).reset_index()
display(bpf_store_type.head(10))
```
**Task 4. Show revenue trend my month (MRR)**
``` python
df['Order_Date'] = df['Date'].dt.date
display(df.head())
-- create a new table order month 
df['Order_Month'] = df['Date'].dt.strftime('%Y-%m')
display(df.head())

-- To find monthly sales grouped by month
monthly_sales = df.groupby('Order_Month')['Total_Cost'].sum().reset_index()
display(monthly_sales.head())
```
<img width="1411" height="485" alt="order month" src="https://github.com/user-attachments/assets/61af2e25-f00e-4743-b54d-51c8de98b80f" />

``` pyhton
monthly_sales = monthly_sales.sort_values('Order_Month', ascending=True)
display(monthly_sales.head())

-- plot monthly sales trend
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['Order_Month'], monthly_sales['Total_Cost'])
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.title('Monthly Sales Trend')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()
```
<img width="1189" height="590" alt="monthly sales trend" src="https://github.com/user-attachments/assets/afb2e80f-95d6-4a77-b02e-d2839984e89d" />


**Task 5. Calculate the Average monthly revenue** 
``` python
average_monthly_sales = monthly_sales['Total_Cost'].mean()
print(f"The average monthly sales is: {average_monthly_sales:.2f}")
```
**Task 6. Calculate the average number of items per transaction**
``` python
avg_number_items_per_transaction = df['Total_Items'].mean()
print(f"The average number of items per transaction is: {avg_number_items_per_transaction:.2f}")
```
**Task 7. Find out the most popular payment methods 
``` python
popular_payment_methods = df.groupby('Payment_Method')['Total_Cost'].sum().sort_values(ascending=False)
display(popular_payment_methods.head())
```
**Task 8. Explain sales trends by customer category
``` python
sales_by_customer_cat = df.groupby('Customer_Category')['Total_Cost'].sum()
display(sales_by_customer_cat.head())
```

<img width="293" height="234" alt="by-categ" src="https://github.com/user-attachments/assets/dbab73be-9e23-44d0-93af-77ccc6b3001c" /> 

**Task 9. summarise yearly sales by customer category 
``` python
df['Order_Year'] = df['Date'].dt.year
display(df.head())

yearly_sales_by_customer_category = df.groupby(['Order_Year', 'Customer_Category'])['Total_Cost'].sum().reset_index()
display(yearly_sales_by_customer_category.head())

yearly_sales_by_customer_category = yearly_sales_by_customer_category.reset_index()
display(yearly_sales_by_customer_category.head())

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
data = df
data_pivot = data.pivot(index='Customer_Category', columns='Season', values='Total_Cost')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.title('Yearly Sales Trend by Customer Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()
```
**Task 10. summarise sales trends by seasons**
``` python
sales_trend_by_Season = df.groupby(['Season', 'Customer_Category'])['Total_Cost'].sum().reset_index()
display(sales_trend_by_Season.head(10))

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Order_Year', y='Total_Cost', hue='Season')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.title('Yearly Sales Trend by Season')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()
```
**Task 11 summarise sales trend by customer category and seasons**
``` python
sales_by_season_customer_category = df.groupby(['Season', 'Customer_Category'])['Total_Cost'].sum().reset_index()
display(sales_by_season_customer_category.head())

display(sales_by_season_customer_category.sort_values('Total_Cost', ascending=False).head())

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.barplot(data=sales_by_season_customer_category, x='Season', y='Total_Cost', hue='Customer_Category')
plt.xlabel('Season')
plt.ylabel('Total Sales')
plt.title('Total Sales by Season and Customer Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()
```
### 3. Customer Segmentation using RFM bucket analysis 
Understanding how recently did each customer make a purchase? number of days since their last transaction and monetary value of their purchase over a period.

**Recency (R)** 
``` python
recency_df = df[['Date','Transaction_ID', 'Customer_Name', 'Product', 'Total_Items', 'Total_Cost']]
display(recency_df.head())

recency_df['Date'] = pd.to_datetime(recency_df['Date'])

latest_date = recency_df['Date'].max()
recency_df['Recency'] = latest_date - recency_df['Date']
display(recency_df.head())

recency_df = recency_df.sort_values(by= 'Last buy', ascending=True)
recency_df.head()

```

<img width="1072" height="408" alt="img-rece" src="https://github.com/user-attachments/assets/09f88b02-cdcd-4dad-a4d1-f510f4c6eea4" /> 

**Frequency (F)** 
``` python
customer_frequency = df.groupby('Customer_Name')['Transaction_ID'].count()
display(customer_frequency.head(20))

customer_frequency_df = customer_frequency.reset_index(name='Frequency')
merged_df = pd.merge(Recency_df, customer_frequency_df, on='Customer_Name', how='left')
display(merged_df.head()) 
```
<img width="1124" height="215" alt="freq" src="https://github.com/user-attachments/assets/f494ebe1-b31c-4443-bd8f-24bceef6090c" />

**Monetary (M)**
``` python 
monetary_df = df.groupby('Customer_Name')['Total_Cost'].sum().reset_index(name='Monetary')
rf_data = pd.merge(merged_df, monetary_df, on='Customer_Name', how='left')
display(rf_data.head())
```
<img width="1226" height="208" alt="monetary" src="https://github.com/user-attachments/assets/0fbcab18-4e1a-49fd-9aad-a908d4c4040d" /> 

**Score ranking for Recency, Frequency and Monetary**
The goal is to convert the raw RFM values into a score (e.g., from 1 to 5). This is best done using the pd.qcut() function, which bins data into equal-sized quantiles.
``` python
-- lower values are better. So, we'll assign the highest score to the lowest recency values.
rf_data['R_score'] = pd.qcut(rf_data['Last buy'], 5, labels=[5,4,3,2,1]).astype(int)

Frequency and Monetary are direct: higher values are better. We'll assign the highest score to the highest values
--frequency score 
rf_data['F_score'] = pd.qcut(rf_data['Frequency'], 5, labels=False, duplicates='drop') + 1
rf_data['F_score'] = rf_data['F_score'].astype(int)

-- monetary score
rf_data['M_score'] = pd.qcut(rf_data['Monetary'], 5, labels=[1,2,3,4,5]).astype(int)
display(rf_data.head())

```

<img width="1397" height="210" alt="rfm-score" src="https://github.com/user-attachments/assets/8af4c4fa-5656-4a93-bf80-ec3f4b303959" />

**Creating the RFM buckets** 
Once I had all the RFM scores I combined them to create an RFM bucket 

``` python
rf_data['RFM_score'] =  rf_data['R_score'].astype(str) + rf_data['F_score'].astype(str) + rf_data['M_score'].astype(str) 
```
<img width="1409" height="313" alt="rfm bucket" src="https://github.com/user-attachments/assets/011c7a5e-1796-4867-a3df-ec79b69d2d1a" /> 

**Classifying customers into bucket** 
I created a function to define customer segments based on these scores to map customers in to buckets e.g ("Champions", "New Customer", "Dormant", "Churn")
``` python
def rfm_segment(rfm_score):
  if rfm_score in ['555', '554', '544', '545', '454', '455', '445']:
    return 'Champions'
  elif rfm_score in ['512', '513', '514', '515', '412', '413', '414', '415']:
    return 'New Customers'
  if rfm_score in ['111', '112', '121', '122', '131', '132']:
    return 'Dormant'
  return 'Churn'

rf_data['RFM_segment'] = rf_data['RFM_score'].apply(rfm_segment)

rf_data
```
<img width="1395" height="403" alt="churn" src="https://github.com/user-attachments/assets/15e3be32-7e9f-41e9-b68b-a95baef1516b" /> 

**View customers based their buckets** 
``` python
Champions = rf_data[rf_data['RFM_segment'] == 'Champions']
churn = rf_data[rf_data['RFM_segment'] == 'Churn']
new_customers = rf_data[rf_data['RFM_segment'] == 'New Customers']
dormant = rf_data[rf_data['RFM_segment'] == 'Dormant']
```



**Insights**
`sales has consistently declined on the 12 month every year in the last 4 years` 
`average monthly sales is $989721.14`
`The average number of items per transaction is 5.5 ~6 items per transaction`
`The top 3 most valuable customer category are the "Homemaker", "Middle Aged" and "Professional"`
`The highest sales were observed in Summer for Senior Citizens with total sales of $10,271.25, followed by Fall for Professionals ($9,985.50), and Spring for Teenagers ($9,820.15)`
`The lowest sales were observed in Winter for Professionals with total sales of $9,379.80, followed by Winter for Students ($9,390.45), and Spring for Middle-Aged individuals ($9,420.20).` 
`Sales tend to be higher in Summer and Fall compared to Winter and Spring.`
`Over 11124 customers have churned over the period`

Author: Farouq Omar Aremu 

`Thank you `





