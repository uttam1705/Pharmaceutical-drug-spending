#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\Uttam Prajapati\Desktop\Data Science\SQL project 2\Machine learning project\pharmaceutical drug spending by country.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df1=df.drop(columns="FLAG_CODES")
df1


# In[8]:


unique_values = df1['LOCATION'].unique()
unique_values


# In[9]:


plt.figure(figsize=(25,25))
sns.countplot(df["LOCATION"])


# In[10]:


df1.describe(include="all")


# ## astype() function is used to convert the data type of a Pandas DataFrame or Series to a specified data type. 
# 
# 

# ### set_index() function is a method available in the Pandas library that is used to set one or more columns of a DataFrame as the index.

# In[11]:


df_clean = df1.astype({'LOCATION': 'category'}).set_index(['LOCATION', 'TIME'])
df_clean


# In[12]:


df_clean.info()


# ## To add additional column

# In[13]:


df_clean['PC_HCSpending_GPD'] = df_clean['PC_GDP']*100/df_clean['PC_HEALTHXP']


# ### To add difference between the previous year and the current year
# ### it then calculates the difference between the current row and the previous row within each group, and stores the result in a new column named "delta_{column}", where {column} is the name of the original column.
# ### This creates a new column for each original column that contains the difference between adjacent rows within each group, which can be useful for analyzing trends and changes within each location. Note that the loop only calculates differences within each location, as specified by the groupby operation. This is important because without the groupby, the differences could also include the time intervals between adjacent locations, which is not what we want in this case.

# In[14]:


for column in ["PC_HEALTHXP", "PC_GDP", "USD_CAP", "TOTAL_SPEND", "PC_HCSpending_GPD"]:
    df_clean["delta_{}".format(column)] = df_clean.groupby('LOCATION').diff()[column]


# In[15]:


for column in ["PC_HEALTHXP", "PC_GDP", "USD_CAP", "TOTAL_SPEND", "PC_HCSpending_GPD"]:
    df_clean['delta_{}'.format(column)] = df_clean.groupby('LOCATION').diff()[column]


# In[16]:


df_clean.head()


# ### VisualAnalysis is a tool for interactive data exploration and visualization that is built on top of pandas and Plotly. It allows you to quickly visualize and explore your data using a web-based interface with interactive plots and filters.
# 
# ### In this case, the code is creating a visual analysis of df_clean with a categorical column of "LOCATION", allowing you to explore the data by location. The reset_index() function is called on the DataFrame before passing it to VisualAnalysis to ensure that the index is a simple sequential integer index, which is required for the tool to work properly.
# 
# ### Note that VisualAnalysis is an interactive tool that opens a new tab in your web browser. You can use the controls in the interface to explore the data and visualize it in various ways.
# 
# 
# 
# 

# In[17]:


get_ipython().system('pip install pandas-visual-analysis --quiet')
from pandas_visual_analysis import VisualAnalysis
VisualAnalysis(df_clean.reset_index(), categorical_columns=["LOCATION"])


# In[18]:


import plotly.express as px
import plotly.graph_objects as go
fig = px.line(df_clean, x=df_clean.index.get_level_values('TIME'), y="PC_HEALTHXP", color=df_clean.index.get_level_values('LOCATION'))
fig.add_trace(go.Scatter(x=df_clean.groupby('TIME').mean().index, y=df_clean.groupby('TIME').mean()["PC_HEALTHXP"], mode='lines', line={'width': 6, 'color': 'black'}, name='Mean'))
fig.show()
# We see fluctuations in the PC_HEALTHXP across the years but on first sight there isn't a direct pattern.
# We had already seen during the pandas_profiling that there was no strong correlation between TIME and PC_HEALTHXP, but it is always best to also look at the graph in case there were periodic patterns, ...


# ### The code you provided is using the plotly library to create a line plot of the PC_HEALTHXP variable over time (TIME) for different countries (LOCATION). It also adds a line showing the mean value of PC_HEALTHXP over time.
# 
# ### Based on the graph, you observe fluctuations in PC_HEALTHXP across the years for different countries, but there doesn't seem to be a clear pattern. This is consistent with the earlier finding that there was no strong correlation between TIME and PC_HEALTHXP.
# 
# ### It is always a good idea to visually inspect the data and check for any patterns or trends that may not be evident from numerical summaries. Periodic patterns or trends may be indicative of seasonality, which could impact the analysis and interpretation of the data.
# 
# ### Overall, the code you provided seems to be a useful visualization for exploring the relationship between PC_HEALTHXP and TIME for different countries.

# In[28]:


import plotly.express as px
fig = px.scatter(df_clean, y="PC_GDP", x="USD_CAP", color=df_clean.index.get_level_values('LOCATION'))
fig.add_trace(go.Scatter(x=pd.cut(df_clean['USD_CAP'],15, retbins=True)[1], 
                         y=df_clean.groupby(pd.cut(df_clean['USD_CAP'],15, 
                                                   retbins=True)[0]).mean()["PC_GDP"], mode='lines', 
                         line={'width': 6, 'color': 'black'}, name='Mean'))
fig.show()
# We did see a stronger correlation during the pandas_profiling between PC_HEALTHXP and USD_CAP. We see this returning in below scatter plot


# ### The code you provided is using the plotly library to create a scatter plot of the relationship between PC_GDP and USD_CAP for different countries (LOCATION). It also adds a line showing the mean PC_GDP for each of 15 equal-sized bins of USD_CAP.
# 
# ### This visualization allows for an exploration of the relationship between PC_GDP and USD_CAP, which may indicate whether there is a positive correlation between the two variables. Additionally, the mean line for each bin provides an overall trend of the relationship between the two variables.
# 
# ### However, it's important to note that the choice of bin size can have an impact on the interpretation of the results. Using 15 equal-sized bins may not capture all the variability in the data, and using a different number of bins or a different binning approach may provide different insights.
# 
# ### Overall, the code you provided is a useful visualization for exploring the relationship between PC_GDP and USD_CAP, but it is important to carefully interpret the results and consider the impact of different binning strategies.

# In[29]:


import plotly.express as px
fig = px.scatter(df_clean, y="PC_HEALTHXP", x="USD_CAP", color=df_clean.index.get_level_values('LOCATION'))
fig.add_trace(go.Scatter(x=pd.cut(df_clean['USD_CAP'],15, retbins=True)[1], 
                         y=df_clean.groupby(pd.cut(df_clean['USD_CAP'],15, retbins=True)[0]).mean()["PC_HEALTHXP"], 
                         mode='lines', line={'width': 6, 'color': 'black'}, name='Mean'))
fig.show()
# However, we do not see the same correlation in PC_HEALTHXP which seems to suggest that if nations 
# spent more per capita on drugs, it is because they spent more on healthcare in general. 
# Thus there is no/little correlation between USD_CAP and PC_HEALTHXP


# In[32]:


import plotly.express as px
fig = px.scatter(df_clean, y="PC_HCSpending_GPD", x="USD_CAP", color=df_clean.index.get_level_values('LOCATION'))
fig.add_trace(go.Scatter(x=pd.cut(df_clean['USD_CAP'],15, retbins=True)[1], 
                         y=df_clean.groupby(pd.cut(df_clean['USD_CAP'],15, 
                         retbins=True)[0]).mean()["PC_HCSpending_GPD"], 
                         mode='lines', line={'width': 6, 'color': 'black'}, name='Mean'))
fig.show()
# In below graph you see the HC spending as part of GDP does increase. 
# So an increase in spending per capita is not related to an relative equal increase in GPD, 
# as else we'd not see a correlation (see analysis between PC_HEALTHXP and USD_CAP)


# ### In the code snippet you provided, a scatter plot is created using the px.scatter() function from the Plotly Express library. The plot visualizes the relationship between the variables "PC_HCSpending_GPD" and "USD_CAP", with points colored by the country represented in the data.
# 
# ### A new trace is then added to the plot using the add_trace() method from the go module of the Plotly library. This trace represents the mean value of "PC_HCSpending_GPD" for each of 15 equally sized bins of the "USD_CAP" variable. The mean values are calculated using the groupby() and mean() methods of the Pandas library.
# 
# ### By visualizing this new trace, it is clear that there is a positive relationship between "USD_CAP" and "PC_HCSpending_GPD". Specifically, as "USD_CAP" increases, the mean value of "PC_HCSpending_GPD" tends to increase as well.
# 
# ### This observation suggests that an increase in healthcare spending per capita is not necessarily related to an equal increase in GDP. This contrasts with the analysis between "PC_HEALTHXP" and "USD_CAP" that you performed earlier, which found no strong correlation between these variables.

# ###1.px.scatter(df_clean, y="PC_HCSpending_GPD", x="USD_CAP", color=df_clean.index.get_level_values('LOCATION')): This function creates a scatter plot using Plotly Express. It takes the cleaned DataFrame df_clean as input and specifies that the "PC_HCSpending_GPD" variable should be plotted on the y-axis and the "USD_CAP" variable on the x-axis. The color argument is used to color-code the points based on the country represented in the data.
# 
# ###2.go.Scatter(x=pd.cut(df_clean['USD_CAP'],15, retbins=True)[1], y=df_clean.groupby(pd.cut(df_clean['USD_CAP'],15, retbins=True)[0]).mean()["PC_HCSpending_GPD"], mode='lines', line={'width': 6, 'color': 'black'}, name='Mean'): This function adds a new trace to the existing plot using the go module of Plotly. Specifically, it creates a line plot showing the mean value of "PC_HCSpending_GPD" for each of 15 equally sized bins of the "USD_CAP" variable. The x argument specifies the bin boundaries and the y argument calculates the mean value for each bin using the groupby() and mean() methods of the Pandas library. The mode argument specifies that the trace should be shown as a line plot, and the line argument sets the width and color of the line. Finally, the name argument sets the label for the trace, which will appear in the plot legend.
# 
# ###3.fig.show(): This function displays the plot in the output.

# 
# #add a scatter plot trace
# #fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name='Data'))
# 
# #add a line plot trace
# #fig.add_trace(go.Scatter(x=x_data, y=fit_data, mode='lines', name='Fit'))

# In[ ]:




