#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize']=8,4
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv("economic_freedom_index2019_data.csv")


# In[3]:


df


# In[4]:


for i in df:
    if type(df[i][0]) == str:
        print(i)


# In[7]:


print(df.info())


# In[9]:


df.isnull().sum()

df.isnull().any()
# In[12]:


sns.heatmap(df.isnull(), yticklabels=False, annot=True)


# In[13]:


df_dropped = df.dropna()


# In[14]:


df_dropped


# In[17]:


df_dropped1 = df_dropped.interpolate()


# In[18]:


df_dropped


# In[19]:


df['GDP per Capita (PPP)'] = df['GDP per Capita (PPP)'].str.replace('$', '')
df['GDP per Capita (PPP)'] = df['GDP per Capita (PPP)'].str.replace(',', '')
df['GDP per Capita (PPP)'] = df['GDP per Capita (PPP)'].str.split(' ').str[0]
df['GDP per Capita (PPP)'] = df['GDP per Capita (PPP)'].astype(float)


# In[20]:


type(df['GDP per Capita (PPP)'][0])


# In[21]:


df['Unemployment (%)'] = df['Unemployment (%)'].str.replace(',', '')
df['Unemployment (%)'] = df['Unemployment (%)'].str.split(' ').str[0]
df['Unemployment (%)'] = df['Unemployment (%)'].astype(float)


# In[22]:


type(df['Unemployment (%)'][0])


# In[23]:


df["FDI Inflow (Millions)"]= df["FDI Inflow (Millions)"].str.replace(',', '')
df["FDI Inflow (Millions)"]= df["FDI Inflow (Millions)"].str.split(' ').str[0]
df["FDI Inflow (Millions)"]= df["FDI Inflow (Millions)"].astype(float)


# In[24]:


type(df["FDI Inflow (Millions)"][0])


# In[25]:


df["GDP (Billions, PPP)"] = df["GDP (Billions, PPP)"].str.replace('$', '')
df["GDP (Billions, PPP)"]= df["GDP (Billions, PPP)"].str.replace(',', '')
df["GDP (Billions, PPP)"]= df["GDP (Billions, PPP)"].str.split(' ').str[0]
df["GDP (Billions, PPP)"]= df["GDP (Billions, PPP)"].astype(float)


# In[62]:


underdeveloped=df[df["GDP per Capita (PPP)"]<12000]


# In[63]:


developing= df[(df["GDP per Capita (PPP)"]>=12000) & (df["GDP per Capita (PPP)"]<=25000)]


# In[64]:


developed=df[df["GDP per Capita (PPP)"]>25000]


# In[65]:


len(developed["Country Name"])


# In[66]:


developed["GDP (Billions, PPP)"].sum()


# In[67]:


developing["GDP (Billions, PPP)"].sum()


# In[68]:


underdeveloped["GDP (Billions, PPP)"].sum()


# In[69]:


my_dict={"country":["developed","developing","underdeveloped"],
"GDP":[64940.5,43645.9,18485.699999999997]}


# In[79]:


list1=[64940.5,43645.9,18485.699999999997]

list2=["developed","developing","underdeveloped"]


# In[80]:


plt.pie(list1,labels=list2,autopct='%2.1f%%' )
plt.show()


# In[81]:


len(developed["Country Name"])


# In[82]:


len(developing["Country Name"])


# In[83]:


len(underdeveloped["Country Name"])


# In[86]:


list3=[56,41,87]

list4=["developed","developing","underdeveloped"]


# In[87]:


plt.pie(list3,labels=list4,autopct='%2.1f%%' )
plt.show()


# In[28]:


df.describe()


# In[29]:


sns.distplot(df["GDP Growth Rate (%)"], )


# In[30]:


sns.distplot(df["World Rank"] )


# In[31]:


df["Population (Millions)"]= df["Population (Millions)"].str.replace(',', '')
df["Population (Millions)"]= df["Population (Millions)"].str.split(' ').str[0]
df["Population (Millions)"]= df["Population (Millions)"].astype(float)


# In[32]:


sns.distplot(df["Population (Millions)"],bins=1)


# In[33]:


sns.distplot(df["Region Rank"],bins=1)


# In[34]:


sns.distplot(df["2019 Score"],bins=8)


# In[35]:


sns.distplot(df["Property Rights"],bins=3)


# In[36]:


sns.distplot(df["Judical Effectiveness"],bins=9)


# In[37]:


sns.distplot(df["Government Integrity"],bins=4)


# In[38]:


sns.distplot(df["5 Year GDP Growth Rate (%)"],bins=5)


# In[39]:


sns.distplot(df["Unemployment (%)"],bins=7)


# In[40]:


sns.distplot(df["FDI Inflow (Millions)"],bins=7)


# In[41]:


sns.distplot(df["Public Debt (% of GDP)"],bins=7)


# In[45]:


sns.distplot(df["Inflation (%)"],bins=7)


# In[49]:


sns.distplot(df["Tariff Rate (%)"],bins=7)


# In[50]:


sns.distplot(df["Corporate Tax Rate (%)"],bins=7)


# In[52]:


sns.distplot(df["Gov't Spending"],bins=7)


# In[53]:


sns.distplot(df["Tax Burden"],bins=7)


# In[61]:


sns.distplot(df["Public Debt (% of GDP)"],bins=7)


# In[96]:


sns.lmplot(data=df, x='GDP per Capita (PPP)', y='Corporate Tax Rate (%)')


# In[98]:


sns.lmplot(data=df, x='GDP per Capita (PPP)', y='Tariff Rate (%)', hue='Region')


# In[103]:


sns.lmplot(data=df, x='GDP per Capita (PPP)', y="Gov't Spending")


# In[105]:


sns.lmplot(data=df, x='GDP per Capita (PPP)', y="Gov't Spending", hue='Region')


# In[104]:


sns.lmplot(data=df, x='GDP per Capita (PPP)', y="Corporate Tax Rate (%)")


# In[106]:


sns.lmplot(data=df, x='GDP per Capita (PPP)', y="Corporate Tax Rate (%)", hue='Region')


# In[122]:


#Analysis for developed countries
developed=df[df["GDP per Capita (PPP)"]>25000]


# In[123]:


developed


# In[124]:


d = developed.corr()


# In[125]:


d


# In[132]:


top_correlations = d['GDP per Capita (PPP)'].sort_values().head(6)


# In[133]:


top_correlations


# In[136]:


print(developed.columns)


# In[141]:


d1 = developed[(developed['GDP Growth Rate (%)'] > 2) & (developed['GDP Growth Rate (%)']< 3)]


# In[147]:


sns.lmplot(data=d1, x='GDP per Capita (PPP)', y='GDP Growth Rate (%)', hue='Region')


# In[150]:


sorted_gdp_per_capita = d['GDP per Capita (PPP)'].sort_values().tail()


# In[151]:


sorted_gdp_per_capita


# In[152]:


sns.lmplot(data=developed, x='GDP per Capita (PPP)', y="Judical Effectiveness")


# In[153]:


sns.lmplot(data=developed, x='GDP per Capita (PPP)', y="Judical Effectiveness", hue='Region')


# In[154]:


sns.lmplot(data=developed, x='GDP per Capita (PPP)', y="Government Integrity")


# In[155]:


sns.lmplot(data=developed, x='GDP per Capita (PPP)', y="Government Integrity", hue='Region')


# In[156]:


sns.lmplot(data=developed, x='GDP per Capita (PPP)', y="Population (Millions)")


# In[157]:


sns.lmplot(data=developed, x='GDP per Capita (PPP)', y="Population (Millions)", hue='Region')


# In[161]:


#Analysis for developing countries
developing_countries = df[(df["GDP per Capita (PPP)"] > 12000) & (df["GDP per Capita (PPP)"] < 25000)]


# In[162]:


d= developing.corr()


# In[164]:


d["GDP per Capita (PPP)"].sort_values().head(6)


# In[168]:


d2 = developing[(developing['GDP Growth Rate (%)'] > 4) & (developing['GDP Growth Rate (%)'] < 6)]


# In[170]:


d2


# In[173]:


sns.lmplot(data=d2, x='GDP per Capita (PPP)', y='GDP Growth Rate (%)', hue='Region')


# In[176]:


e = developing[developing["Region"] == "Europe"]
correlation_matrix = e[['GDP per Capita (PPP)', 'GDP Growth Rate (%)']].corr()
print(correlation_matrix)


# In[177]:


d= developing.corr()


# In[185]:


d['GDP per Capita (PPP)'].sort_values().tail(6)


# In[186]:


sns.lmplot(data=developing, x='GDP per Capita (PPP)', y="Government Integrity")


# In[187]:


sns.lmplot(data=developing, x='GDP per Capita (PPP)', y='Government Integrity', hue='Region')


# In[188]:


e=developing[developing ["Region"]=="Sub-Saharan Africa"]
e
e[['GDP per Capita (PPP)', "Government Integrity"]]. corr()


# In[189]:


sns.lmplot(data=developing, x='GDP per Capita (PPP)', y="Property Rights")


# In[190]:


sns.lmplot(data=developing, x='GDP per Capita (PPP)', y='Property Rights', hue='Region')


# In[191]:


e=developing[developing ["Region"]=="Sub-Saharan Africa"]
e
e[['GDP per Capita (PPP)', "Property Rights"]]. corr()


# In[193]:


sns.lmplot(data=developing, x='GDP per Capita (PPP)', y="Fiscal Health")


# In[194]:


sns.lmplot(data=developing, x='GDP per Capita (PPP)', y='Fiscal Health', hue='Region')


# In[227]:


e=developing[developing ["Region"]=="Europe"]
e
e[['GDP per Capita (PPP)', "Fiscal Health"]]. corr()


# In[198]:


#analysis for underdeveloped countries
underdeveloped = df[df["GDP per Capita (PPP)"] <= 12000]


# In[199]:


d = underdeveloped.corr()
d["GDP per Capita (PPP)"].sort_values().head(6)


# In[201]:


d= underdeveloped[underdeveloped['GDP Growth Rate (%)'] >= 15]


# In[203]:


sns.lmplot(data=d, x='GDP per Capita (PPP)', y='GDP Growth Rate (%)', hue='Region')


# In[204]:


e= underdeveloped.corr()
e['GDP per Capita (PPP)'].sort_values().tail(6)


# In[205]:


sns.lmplot(data=developing, x='GDP per Capita (PPP)', y="Trade Freedom")


# In[206]:


sns.lmplot(data=developing, x='GDP per Capita (PPP)', y='Trade Freedom', hue='Region')


# In[217]:


e=underdeveloped[underdeveloped ["Region"]=="Middle East and North Africa"]
e
e[['GDP per Capita (PPP)', "Trade Freedom"]]. corr()


# In[209]:


sns.lmplot(data=developing, x='GDP per Capita (PPP)', y="Property Rights")


# In[210]:


sns.lmplot(data=developing, x='GDP per Capita (PPP)', y='Property Rights', hue='Region')


# In[216]:


e=underdeveloped[underdeveloped ["Region"]=="Asia-Pacific"]
e
e[['GDP per Capita (PPP)', "Property Rights"]]. corr()


# In[213]:


sns.lmplot(data=developing, x='GDP per Capita (PPP)', y= "Tax Burden % of GDP")


# In[214]:


sns.lmplot(data=developing, x='GDP per Capita (PPP)', y='Tax Burden % of GDP', hue='Region')


# In[215]:


e=underdeveloped[underdeveloped ["Region"]=="Middle East and North Africa"]
e
e[['GDP per Capita (PPP)', "Tax Burden % of GDP"]]. corr()


# In[224]:


heat = df.drop(['CountryID', 'Country Name', 'WEBNAME', 'Region', 'World Rank',
                         'Region Rank', '2019 Score'], axis=1).corr()


# In[225]:


sns.heatmap(correlation_matrix, vmax=1, vmin=-1, linewidths=1, linecolor='black', cmap='coolwarm', annot=True, fmt=".2f")


# In[226]:


sns. heatmap (heat, vmax=1, vmin=-1, linewidths=1, linecolor='Black')


# In[ ]:




