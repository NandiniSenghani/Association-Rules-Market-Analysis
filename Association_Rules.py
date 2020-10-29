# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 19:29:35 2020

@author: Nandini senghani
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


#####################################################Groceries data set#################################

groceries = []
with open("groceries.csv") as f:groceries = f.read()
groceries=groceries.split("\n") 
groceries_list=[]
for i in groceries:
    groceries_list.append(i.split(","))
all_groceries_list=[]
all_groceries_list=[i for item in groceries_list for i in item]


item_frequencies= Counter(all_groceries_list)
item_frequencies=sorted(item_frequencies.items(),key=(lambda x:x[1]))#lambda signifies an anonymous function. 
#In this case, this function takes the single argument x and returns x[1] (i.e. the item at index 1 in x).

frequencies=list(reversed([i[1] for i in item_frequencies] ))
items=list(reversed([i[0] for i in item_frequencies] ))

#plotting
plt.bar(x=list(range(0,11)),height=frequencies[0:11],color="blue");
plt.xticks(list(range(0,11)),),items[0:11],plt.xlabel("items"),plt.ylabel("frequencies")


groceries_data=pd.DataFrame(pd.Series(groceries_list))
groceries_data
#removing any empty transactions
groceries_data= groceries_data.iloc[:9835,:]
groceries_data.columns=["transactions"]
Y=groceries_data["transactions"].str.join(sep="*").str.get_dummies(sep="*")
frequent_items=apriori(Y,0.005,True,3)
frequent_items.shape
frequent_items.sort_values("support",ascending=False,inplace=True)
plt.bar(x = list(range(1,11)),height = frequent_items.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),frequent_items.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')
groceries_rules=association_rules(frequent_items,metric="lift",min_threshold=4)
groceries_rules
groceries_rules.sort_values(by="lift",ascending=False,inplace=True)

##################################################################Book(1) Dataset#########################################
book=pd.read_csv("book.csv")
book.head()
book_new=book.melt(var_name="books",value_name="values")
books=pd.crosstab(index=book_new['values'], columns=book_new['books'])
books.iloc[1,:].plot(kind="bar")
books.iloc[0,:].plot(kind="bar")
frequent_items=apriori(book, min_support=0.005, max_len=3,use_colnames = True)
frequent_items.shape
frequent_items.sort_values("support",ascending=False,inplace=True)

plt.bar(x = list(range(0,11)),height = frequent_items.support[0:11],color='rgmyk');plt.xticks(list(range(0,11)),frequent_items.itemsets[0:11]);plt.xlabel('item-sets');plt.ylabel('support')

rules=association_rules(frequent_items,metric="lift",min_threshold=4)
rules_c=association_rules(frequent_items,metric="confidence",min_threshold=0.7)
rules.sort_values('lift',ascending = False,inplace=True)

#To eliminate Redudancy in Rules 

def to_list(i):
    return (sorted(list(i)))
#Sorting, listing and appending

ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)
ma_X = ma_X.apply(sorted)
rules_sets = list(ma_X)
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
# getting rules without any redudancy 

rules_no_redudancy  = rules.iloc[index_rules,:]
# Sorting them with respect to list and getting top 10 rules 

rules_no_redudancy.sort_values('lift',ascending=False).head(10)

plt.bar(x = list(range(0,11)),height = rules_no_redudancy.lift[0:11],color='rgmyk');plt.xticks(list(range(0,11)),rules_no_redudancy.antecedents[0:11])

plt.scatter(rules_no_redudancy['support'],rules_no_redudancy['lift'], alpha=0.5);plt.xlabel('support');plt.ylabel('lift');plt.title('Support vs Lift')
plt.plot(rules_no_redudancy['lift'], rules_no_redudancy['confidence'],'go')



##################################################################my_movies dataset######################################
movies = []
# loading file in transaction format
with open("my_movies.csv") as f:
    movies = f.read()

#splitting the data "\n"
movies = movies.split("\n")
movies_list = []
for i in movies:
    movies_list.append(i.split(","))    
all_movies_list = []
for i in movies_list:
    all_movies_list = all_movies_list+i

movie_freq = Counter(all_movies_list)

movie_freq = sorted(movie_freq.items(),key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in movie_freq]))
items = list(reversed([i[0] for i in movie_freq]))

plt.bar(height = frequencies[0:11],x = list(range(0,11)),color='rgbkymc');plt.xticks(list(range(0,11),),items[0:11]);plt.xlabel("items");plt.ylabel("Count")

movies_series  = pd.DataFrame(pd.Series(movies_list))

movies_series = movies_series.iloc[:11,:]

movies_series.columns=['movies1']

X = movies_series['movies1'].str.join(sep='*').str.get_dummies(sep='*')

frequent_itemsets = apriori(X, min_support=0.005, max_len=3,use_colnames = True)

frequent_itemsets.sort_values('support',ascending = False,inplace=True)

plt.bar(x = list(range(0,11)),height = frequent_itemsets.support[0:11],color='rgmyk');plt.xticks(list(range(0,11)),frequent_itemsets.itemsets[0:11]);plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.sort_values('lift',ascending = False,inplace=True)

#To eliminate Redudancy in Rules 

def to_list(i):
    return (sorted(list(i)))
#Sorting, listing and appending

ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)
ma_X = ma_X.apply(sorted)
rules_sets = list(ma_X)
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
# getting rules without any redudancy 

rules_no_redudancy  = rules.iloc[index_rules,:]
# Sorting them with respect to list and getting top 10 rules 

rules_no_redudancy.sort_values('lift',ascending=False).head(10)

plt.bar(x = list(range(0,11)),height = rules_no_redudancy.lift[0:11],color='rgmyk');plt.xticks(list(range(0,11)),rules_no_redudancy.antecedents[0:11])

plt.scatter(rules_no_redudancy['support'],rules_no_redudancy['lift'], alpha=0.5);plt.xlabel('support');plt.ylabel('lift');plt.title('Support vs Lift')
plt.plot(rules_no_redudancy['lift'], rules_no_redudancy['confidence'],'go')
