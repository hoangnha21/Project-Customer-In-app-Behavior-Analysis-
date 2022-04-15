# Python_Project_Data_Analytics
Customer_to_Subscription_through_app_behavior
import numpy as np
import pandas as pd

appData=pd.read_csv("FineTech_appData.csv")
appData.head(10)

#null values
appData.isnull().sum()
