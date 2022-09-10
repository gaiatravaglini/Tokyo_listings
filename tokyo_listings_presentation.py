import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
import numpy as np
np.set_printoptions(suppress=True)

import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import matplotlib as lb

import folium
from folium import plugins
import streamlit as st

tokyo_listings_df =pd.read_csv('/Users/gaiatravaglini/Desktop/Tokyo_listings/listings.csv', na_values='')

from PIL import Image
image = Image.open('Tokyo.jpg')

st.image(image, caption='')

st.title(' Airbnb Tokyo Activities Analysis')

st.write('The project aims at discovering some particular insights within data collected in 2022 from AirBnb in Tokyo area.')
st.write('The project is structured as follows:')
st.write('- **Data Cleaning:** filter the dataset by removing unuseful attributes and replace null values.')
st.write('- **Data Exploration**')
st.write('- **Clustering:** divide data points into groups.')
st.write('- **Classification:** classify data points into different classes, according to a certain target.')

tokyo_listings_df.drop(['neighbourhood_group','last_review','reviews_per_month','calculated_host_listings_count','number_of_reviews_ltm','license'], axis=1,inplace=True)
tokyo_listings_df.fillna({'host_name':0}, inplace=True)
tokyo_listings_corr=tokyo_listings_df.corr()

if st.checkbox('Show Raw Data:'):
    st.write('The raw data:')
    st.write(tokyo_listings_df)
    
if st.checkbox('Show Data Description:'):
    st.write('Size:',tokyo_listings_df.shape)
    st.write('**Categorical Data**:', '4 attributes',  '_(name, host_name, neighbourhood, room_type)_')
    st.write('**Quantitative Data**:', '6 attributes',  '_(latitude, longitude, price, minimum_nights, number_of_reviews, availability_365)_')
    st.write('Correlation Matrix')
    fig, ax= plt.subplots(figsize=(10,6))
    sns.heatmap(tokyo_listings_corr, annot=True)
    st.write(fig)



st.header('Data Exploration')

st.subheader('Hosts with the highest number of listings activities')
popular_host=tokyo_listings_df['host_id'].value_counts().head(14)
popular_host=popular_host.reset_index()
popular_host=popular_host.rename(columns={'index':'host_id','host_id':'listings_count'})

order_listings=popular_host.groupby('host_id')['listings_count'].sum().sort_values(ascending=False).index.values

fig, ax=plt.subplots(figsize=(20,12))
fig1=sns.barplot(data=popular_host, x='host_id',y='listings_count',palette='Pastel1_d',order=order_listings)
fig1.set_title('Number of Activities per Host')
fig1.set_ylabel('Number of listings')
fig1.set_xlabel('Host Id')
st.write(fig)

with st.expander("See explanation"):
     st.write(
         "The chart  shows the neighbourhoods with the highest number of listings in Tokyo. \n \n \n **1st:** Host Hiroshi (ID:258668827), 98 listings.")

st.subheader('Neighbourhoods in Tokyo with the highest number of listings')

neighbour_pop =tokyo_listings_df['neighbourhood'].value_counts()[tokyo_listings_df['neighbourhood'].value_counts()>200]
#filter neighbourhoods with at least 200 activities

neighbour_pop.count()

fig, ax= plt.subplots(figsize=(20,12))
plt.pie(neighbour_pop, labels=neighbour_pop, autopct = '%1.1f%%', startangle=100)       #autopic show percentage; 1 one decimal
plt.legend(neighbour_pop.index,loc='center left',bbox_to_anchor=(1, 0.5))
plt.title('Neighbourhood Popularity')
st.write(fig)

with st.expander("See explanation"):
     st.write(
         'The chart shows how frequent the different types of accomodation are and their respective listing counts. \n \n \n In this graph, to allow for a better visualization, only the neighbourhoods with the highest number of listings are considered (> 200 listings)')

top_neigh=tokyo_listings_df.loc[tokyo_listings_df['neighbourhood'].isin(['Chuo Ku', 'Arakawa Ku', 'Edogawa Ku', 'Katsushika Ku',
       'Minato Ku', 'Toshima Ku', 'Sumida Ku', 'Shibuya Ku',
       'Shinjuku Ku', 'Nakano Ku', 'Taito Ku', 'Setagaya Ku', 'Kita Ku',
       'Ota Ku'])]

fig, ax= plt.subplots(figsize=(20,12))
fig3 =px.scatter_mapbox(data_frame=top_neigh,
                      lat="latitude",
                      lon="longitude",
                      color='neighbourhood',
                    hover_data=["name"],
                     hover_name="neighbourhood",
                     height=400,
                      width=600,zoom=9.5)
                     

fig3.update_layout(mapbox_style="carto-positron")
fig3.update_layout(margin={"r":0,"t":1,"l":0,"b":0})
st.write('Map of Tokyo Neighbourhood Distribution')
st.plotly_chart(fig3,se_container_width=False, sharing="streamlit")

with st.expander("See explanation"):
     st.write(
         'The graph allows us to have a better understanding of how Airbnb activities are distributed in Tokyo, grouped by their respective neighbourhood.')

st.subheader('Accomodation Frequency in Tokyo')

room_type_pop =tokyo_listings_df['room_type'].value_counts()    

fig, ax= plt.subplots(figsize=(20,12))
plt.pie(room_type_pop, labels=room_type_pop,autopct = '%1.1f%%', startangle=90)       
plt.legend(tokyo_listings_df['room_type'].value_counts().index,title='Room Category')
plt.title('Room Popularity')
st.write(fig)
with st.expander("See explanation"):
     st.write(
         'The chart shows how frequent the different types of accomodation are and their respective listing counts')

st.subheader('Frequency of each accomodation type in top neighbourhoods')

top_neigh=top_neigh.sort_values(by='neighbourhood')

fig, ax= plt.subplots(figsize=(20,12))
fig5= sns.catplot(x='room_type', col='neighbourhood',col_wrap=4, data=top_neigh, kind='count', height=5)
fig5.set_axis_labels("",'Room Count')
fig5.despine(left=True)
fig5.set_xticklabels(rotation=90)
st.pyplot(fig5)
with st.expander("See explanation"):
     st.write('The graph shows the frequency of each type of accomodation for top neighbourhoods. \n \n \n For all neighbourhoods, the winner is _entire home_ , which means that more hosts prefer to rent their entire apartments rather than rent them in single rooms')

st.subheader('Average price of an accomodation in top neighbourhoods')

list1=top_neigh['neighbourhood'].value_counts()


list2=top_neigh['price'].groupby(top_neigh['neighbourhood']).mean()


pop_neigh= pd.concat([list1,list2],axis=1)
pop_neigh=pop_neigh.reset_index()
pop_neigh= pop_neigh.rename(columns={'neighbourhood':'count','index':'neighbourhood'})
pop_neigh=pop_neigh.sort_values(by='price', ascending=False)


order_price=pop_neigh.groupby('neighbourhood')['price'].sum().sort_values(ascending=False).index.values

fig, ax= plt.subplots(figsize=(20,12))
fig6=sns.barplot(data=pop_neigh, x='neighbourhood',y='price',palette='Pastel1_d',order=order_price)
fig6.set_title('Average Price per Neighbourhood')
fig6.set_ylabel('Average Price')
fig6.set_xlabel('Neighbourhood')
st.write(fig)
with st.expander("See explanation"):
     st.write('The graph shows the average price of the listings for each top neighbourhoods. \n \n \n Chuo has the highest average price and it could depend on its central position')

st.subheader('Average price for each accomodation type in top neighbourhoods')

avgprice_room=top_neigh.groupby(['neighbourhood','room_type'])['price'].mean()
avgprice_room=avgprice_room.reset_index()

#hotel room for Shibuya has no value, we can drop it

i=avgprice_room[((avgprice_room.neighbourhood == 'Shibuya Ku') &(avgprice_room.room_type == 'Hotel room'))].index  #get the index in the dataframe

avgprice_room=avgprice_room.drop(i)

fig, ax= plt.subplots(figsize=(20,12))
fig7= sns.catplot(x='room_type',y='price', col='neighbourhood',col_wrap=4,kind='bar', data=avgprice_room , height=5)
fig7.set_axis_labels("",'Average Price')
fig7.despine(left=True)
fig7.set_xticklabels(rotation=90)
st.pyplot(fig7)
with st.expander("See explanation"):
     st.write(' The graph has a more detailed analysis: the average price of each accomodation category per neighbourhood. \n \n \n Entire Home and Private Room are the most expensive in almost every neighbourhood.')

st.subheader('Distribution of price for each top neighbourhood')
d =top_neigh['price'].groupby(top_neigh['neighbourhood']).describe()

fig, ax= plt.subplots(figsize=(20,12))
fig8=sns.boxplot(data=top_neigh, x='neighbourhood',y='price')
fig8.get_yaxis().set_major_formatter(lb.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.ylim(0,100000)
plt.title('Price Distribution (¥)')
st.write(fig)
with st.expander("See explanation"):
     st.write('The graph shows how prices in each top neighbourhood are distributed. \n \n \n Each box shows the mean, the 1st quartile(0.25%) and the 3rd quartile (0.75%) and its extreme points')
     if st.checkbox('Show Statistics'):
        st.write(d)
top_neigh['price'].groupby(top_neigh['neighbourhood']).describe()   #check statistics


fig9=px.scatter_mapbox(data_frame=top_neigh,
                      lat="latitude",
                      lon="longitude",
                      color="price",
                    hover_data=["name"],
                     hover_name="neighbourhood",
                     height=400,
                      width=600,
                     size="price", zoom=10)


fig9.update_layout(mapbox_style="carto-positron")
fig9.update_layout(margin={"r":0,"t":0,"l":0,"b":0})  
st.write('Map of Tokyo Price Distribution')
st.plotly_chart(fig9, se_container_width=False, sharing="streamlit")
with st.expander("See explanation"):
     st.write('The graph works as an interactive tool to understand how prices are distributed on a map of Tokyo. \n \n \n The activities with an higher price are marked by a bigger and lighter circle and they are mainly mapped on the south-center area')
     

st.header('Clustering')

from sklearn.cluster import KMeans
if st.checkbox('Show price statistics'):
    st.write('The dataset statistics:',tokyo_listings_df.price.describe())
#filter prices between 1st and 3rd quartile



st.write('To allow for a better visualization, It is useful to filter the price between the first quartile(0.25) and thirdquartile(0.75)')
df=tokyo_listings_df.loc[(tokyo_listings_df['price'] >=5093) & (tokyo_listings_df['price'] <=13207)]

if st.checkbox('Show new data'):
    st.write('The new filtered data are:',df)

st.subheader('1st Clustering: Minimum Nights and Price')

fig, ax= plt.subplots(figsize=(20,12))
sns.scatterplot(df['price'],df['minimum_nights'])
plt.xlabel('Price')
plt.ylabel('Minimum Nights')
plt.yticks(list(range(0,100,5)))
plt.xticks(list(range(5093,13208,700)))
plt.ticklabel_format(style='plain')
plt.title('Price Behaviour')
st.write(fig)

if st.sidebar.checkbox('Show Elbow Method Code:'):
    st.write('The elbow method:')
    st.code('square_distances = [] \n \n \n x = df[[ price, minimum_nights]] \n \n \n for i in range(1, 11): \n \n \n km = KMeans(n_clusters=i, random_state=42) \n \n \n km.fit(x) \n \n \n square_distances.append(km.inertia_)')

square_distances = []
x = df[['price','minimum_nights']]
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(x)
    square_distances.append(km.inertia_)

fig, ax= plt.subplots(figsize=(20,12))
plt.plot(range(1,11), square_distances, 'bx-')
plt.xlabel('K')
plt.ylabel('inertia')
plt.title('Elbow Method')
plt.xticks(list(range(1,11)))
st.write(fig)
with st.expander("See explanation"):
     st.write('The **Elbow Method** shows the number of clusters that should be adopted when adopting Kmeans. \n \n \n The Inertia variable on the y-axis plots the Within-Cluster-Sum of Squared Errors (WSS) against different values of k, and choose the k for which WSS becomes first starts to diminish.')

if st.sidebar.checkbox('Show Kmeans Code:'):
    st.write('Kmeans Initialization:')
    st.code('km = KMeans(n_clusters=4, random_state=42) \n \n \n y_pred = km.fit_predict(x)')

km = KMeans(n_clusters=4, random_state=42)
y_pred = km.fit_predict(x)

fig, ax= plt.subplots(figsize=(20,12))
for i in range(4):
    plt.scatter(x.loc[y_pred==i,'price'], x.loc[y_pred==i, 'minimum_nights'])
plt.xlabel('Price')
plt.ylabel('Minimum Nights')
plt.title('Clustering with K=4')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=200, marker='X', c='black', edgecolors='black', label='centroids')
st.write(fig)
with st.expander("See explanation"):
     st.write('There 4 clusters, based on 4 increasing ranges of price. \n \n \n It is interesting to see that as the price increases, also some points which allows for a longer minimum stay appear')

st.subheader('2nd Clustering: Number of Reviews and Price')

fig, ax= plt.subplots(figsize=(20,12))
sns.scatterplot(df['number_of_reviews'],df['price'])
plt.xlabel('Total number of Reviews')
plt.ylabel('Price')
plt.xticks(list(range(0,700,100)))
plt.yticks(list(range(5093,13208,500)))
plt.ticklabel_format(style='plain')
plt.title('Price Behaviour')
st.write(fig)

square_distances = []
x = df[['number_of_reviews','price']]
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(x)
    square_distances.append(km.inertia_)

fig, ax= plt.subplots(figsize=(20,12))
plt.plot(range(1,11), square_distances, 'bx-')
plt.xlabel('K')
plt.ylabel('inertia')
plt.title('Elbow Method')
plt.xticks(list(range(1,11)))
st.write(fig)

km = KMeans(n_clusters=4, random_state=42)
y_pred = km.fit_predict(x)

fig, ax= plt.subplots(figsize=(20,12))
for i in range(4):
    plt.scatter(x.loc[y_pred==i,'number_of_reviews'], x.loc[y_pred==i, 'price'])
plt.xticks(list(range(0,700,100)))
plt.yticks(list(range(5093,13208,500)))
plt.xlabel('Total Reviews')
plt.ylabel('Price')
plt.title('Clustering with K=4')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=200, marker='X', c='black', edgecolors='black', label='centroids')
st.write(fig)

st.subheader('3rd Clustering: Longitude and Latitude')
fig, ax= plt.subplots(figsize=(20,12))
plt.scatter(tokyo_listings_df['longitude'], tokyo_listings_df['latitude'])
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Airbnb Acitivities Distribution')
st.write(fig)

square_distances = []
x = tokyo_listings_df[['longitude','latitude']]
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(x)
    square_distances.append(km.inertia_)

fig, ax= plt.subplots(figsize=(20,12))
plt.plot(range(1,11), square_distances, 'bx-')
plt.xlabel('K')
plt.ylabel('inertia')
plt.title('Elbow Method')
plt.xticks(list(range(1,11)))
st.write(fig)

km = KMeans(n_clusters=5, random_state=42)
y_pred = km.fit_predict(x)


fig, ax= plt.subplots(figsize=(20,12))
for i in range(5):
    plt.scatter(x.loc[y_pred==i, 'longitude'], x.loc[y_pred==i, 'latitude'])
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Clustering with K=5')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=200, marker='X', c='black', edgecolors='black', label='centroids')
st.write(fig)

if st.checkbox('Show Map:'):
    st.map(tokyo_listings_df)


st.header('Pricipal Component Analysis')
from sklearn.decomposition import PCA
st.write('PCA is a method that reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information of the initial dataset.')

y = tokyo_listings_df['room_type']

x = tokyo_listings_df.drop(['id','host_id','name','host_name','neighbourhood','room_type'], axis=1)

pca=PCA()
pca.fit(x)


fig, ax= plt.subplots(figsize=(20,12))
plt.plot(range(1,7), pca.explained_variance_ratio_.cumsum(), marker='o',linestyle='--')
plt.title('Explained Variance by components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
st.write(fig)
with st.expander("See explanation"):
     st.write('Principal components are new variables constructed as linear combinations of the initial variables. \n \n \n The principal components are uncorrelated and most of the information within the initial variables is compressed into the first components. \n \n \n PCA put maximum possible information in the first component, then maximum remaining information in the second and so on.')




pca = PCA(n_components=2)
x_pca = pca.fit(x).transform(x)
vr=format(pca.explained_variance_ratio_)
st.write('The shape of the dataset is:',x_pca.shape)


print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
st.write('Explained variation per principal component:',vr)


fig, ax= plt.subplots(figsize=(20,12))
labels = y.unique()
for i in range(len(labels)):
    plt.scatter(x_pca[y==labels[i], 0], x_pca[y==labels[i], 1], label=labels[i])
plt.title('Principal Component Analysis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
st.write(fig)

st.subheader('PCA Testing')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(random_state=42)
accuracies = []
kf = KFold(n_splits=10, shuffle=True, random_state=42)

if st.sidebar.checkbox('Show Kfold Code:'):
    st.write('Kfold')
    st.code('for train_index, test_index in kf.split(x_pca, y): \n \n \n x_train, y_train = x_pca[train_index], y.iloc[train_index] \n \n \n x_test, y_test = x_pca[train_index], y.iloc[train_index] \n \n \n model.fit(x_train, y_train) \n \n \n y_pred = model.predict(x_test) \n \n \n accuracy = accuracy_score(y_pred, y_test) \n \n \n accuracies.append(accuracy) \n \n \n print(Accuracy = , accuracy)')


for train_index, test_index in kf.split(x_pca, y):
    x_train, y_train = x_pca[train_index], y.iloc[train_index]
    x_test, y_test = x_pca[train_index], y.iloc[train_index]

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_pred, y_test)
    print('Accuracy = ', accuracy)
    st.write('Accuracy:',accuracy)


st.header('Classification')

from sklearn.model_selection import train_test_split

st.subheader('1st Classification Analysis')

if st.checkbox('Show 1st Classification Code'):
    st.code('y = tokyo_listings_df[room_type] \n  x = tokyo_listings_df.drop([id, host_id,name, host_name, neighbourhood, room_type], axis=1) \n \n x_train, x_test, y_train, y_test =train_test_split(x, y, test_size=0.3, random_state=42) \n \n  model = RandomForestClassifier() \n   model.fit(x_train, y_train) \n \n \n y_pred = model.predict(x_test) \n accuracy= sum(y_pred == y_test) / len(y_pred)')


y = tokyo_listings_df['room_type']

x = tokyo_listings_df.drop(['id','host_id','name','host_name','neighbourhood','room_type'], axis=1)

x_train, x_test, y_train, y_test =train_test_split(x, y, test_size=0.3, random_state=42)  


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier() 
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy= sum(y_pred == y_test) / len(y_pred)
st.write('Accuracy:',accuracy)

st.subheader('Kfold Validation')
kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []
i = 0
for train_index, test_index in kf.split(x):
    i += 1
    model = RandomForestClassifier(random_state=42)
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)
    accuracies.append(accuracy)
    print(i, ') accuracy = ', accuracy)
st.write('Mean accuracy: ', np.array(accuracies).mean())

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

st.subheader("Confusion Matrix")
fig5 = plt.figure()
conf_matrix = confusion_matrix(y_pred , y_test)
sns.heatmap(conf_matrix , annot=True , xticklabels=['Entire home/apt' , 'Private room','Hotel room','Shared room'] , yticklabels=['Entire home/apt' , 'Private room','Hotel room','Shared room'], fmt='g')
plt.ylabel("True")
plt.xlabel("Predicted")
st.pyplot(fig5)
with st.expander("See explanation"):
     st.write('True Positive:You predicted positive and it’s true. \n \n True Negative: You predicted negative and it’s true. \n \n False Positive (Type 1 Error): You predicted positive and it’s false. \n \n False Negative (Type 2 Error): You predicted negative and it’s false.')
     st.write('For **entire apartment/home:** \n \n **TP:** C0,0 \n \n **FN:** C0,1 + C0,2 + c0,3 \n \n **FP:** C1,0 + C2,0 + C3,0 \n \n  **TN:** remaining cells ')


st.subheader('2nd Classification Analysis')

if st.checkbox('Show 2nd Classification Code'):
    st.code('y = tokyo_listings_df[neighbourhood] \n  x = tokyo_listings_df.drop([id, host_id,name, host_name, neighbourhood, room_type], axis=1) \n \n x_train, x_test, y_train, y_test =train_test_split(x, y, test_size=0.3, random_state=42) \n \n  model = RandomForestClassifier() \n   model.fit(x_train, y_train) \n \n \n y_pred = model.predict(x_test) \n accuracy= sum(y_pred == y_test) / len(y_pred)')

y = tokyo_listings_df['neighbourhood']

x = tokyo_listings_df.drop(['id','host_id','name','host_name','neighbourhood','room_type'], axis=1)

x_train, x_test, y_train, y_test =train_test_split(x, y, test_size=0.3, random_state=42)


model = RandomForestClassifier() 
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy= sum(y_pred == y_test) / len(y_pred)
st.write('Accuracy:',accuracy)

st.subheader('Kfold Validation')
kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []
i = 0
for train_index, test_index in kf.split(x):
    i += 1
    model = RandomForestClassifier(random_state=42)
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)
    accuracies.append(accuracy)
    print(i, ') accuracy = ', accuracy)
st.write('Mean accuracy: ', np.array(accuracies).mean())


from sklearn.metrics import classification_report

st.subheader('Classification Report')
cl_report=print(classification_report(y_test, y_pred,zero_division=1))
if st.checkbox('Show report'):
  st.text('Model Report:\n ' + classification_report(y_test, y_pred))
  with st.expander("See explanation"):
     st.write('**Precision:** Percentage of correct positive predictions relative to total positive predictions. \n \n **Recall:** Percentage of correct positive predictions relative to total actual positives. \n \n **F1 Score:** weighted harmonic mean of precision and recall. The closer to 1, the better the model. \n \n F1 Score: 2 * (Precision * Recall) / (Precision + Recall). \n \n **Support:** how many points belonged to each class in the test dataset.')