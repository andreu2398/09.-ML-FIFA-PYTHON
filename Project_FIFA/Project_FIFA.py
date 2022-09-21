# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 19:03:30 2021

@author: Andreu
"""

"""
THE OBJECTIVE OF THIS PROJECT IS TO ANALYSE AND REPRESENT THE DATA ABOUT THE PLAYERS.


1. Scatter plots of the best 50 field players with different colors depending on the year of the FIFA (fifa15 -> red , fifa18 -> yellow , fifa21 -> blue),
with skill variables (pace , shooting , passing , dribbling , defending , physic)

2. Represent with plot/s how the age of the 100 best players has been changing through the years.

3. What are the most used team_jersey_number's among the 100 best players and the 100 worst players. And finally among all the players.

4. What are the most common team_position's among the 100 best players and the 100 worst players. And finally among all the players.

5. Represent the variables (age , height_cm , weight_kg) for the best 50 field players.

6. Regression about how the variables (29) (attacking_crossing , attacking_finishing , attacking_heading_accuracy , attacking_short_passing , attacking_volleys ,
					skill_dribbling , skill_curve , skill_fk_accuracy , skill_long_passing , skill_ball_control ,
					movement_acceleration , movement_sprint_speed , movement_agility , movement_reactions , movement_balance ,
					power_shot_power , power_jumping , power_stamina , power_strength , power_long_shots ,
					mentality_agression , mentality_interceptions , mentality_positioning , mentality_vision , mentality_penalties , mentality_composure ,
					defending_marking *(it looks like some values are missing) , defending_standing_tackle , defending_sliding_tackle)
	affect the global score. Use all players as observations.

"""
#%% LIBRARIES AND DATA

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
import urllib.request
import webbrowser
import datetime
import re
import seaborn as sns
import plotly
import plotly.graph_objs as go
import plotly.express as px
import random
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df15 = pd.read_csv("data\players_15.csv")
df16 = pd.read_csv("data\players_16.csv")
df17 = pd.read_csv("data\players_17.csv")
df18 = pd.read_csv("data\players_18.csv")
df19 = pd.read_csv("data\players_19.csv")
df20 = pd.read_csv("data\players_20.csv")
df21 = pd.read_csv("data\players_21.csv")

#%% 1

"""
1. Scatter plots (violin cause it represents it better) of the best 50 field players with different colors
depending on the year of the FIFA (fifa15 -> red , fifa18 -> yellow , fifa21 -> blue),
with skill variables (pace , shooting , passing , dribbling , defending , physic)
"""
aa15 = df15[["long_name" , "player_positions" , "overall" , "pace" , "shooting" , "passing" , "dribbling" , "defending" , "physic"]]
aa15 = aa15.loc[~aa15["player_positions"].str.contains("GK")]
aa15 = aa15.sort_values("overall" , ascending = False)
aa15 = aa15.head(40).reset_index(drop = True)
aa15["year"] = 2015

x = list(aa15[["pace" , "shooting" , "passing" , "dribbling" , "defending" , "physic"]].columns.values)

aa15 = pd.melt(frame = aa15 , id_vars = ["long_name" , "player_positions" , "overall" , "year"] , value_vars = x , var_name = "skill" , value_name = "value")


aa18 = df18[["long_name" , "player_positions" , "overall" , "pace" , "shooting" , "passing" , "dribbling" , "defending" , "physic"]]
aa18 = aa18.loc[~aa18["player_positions"].str.contains("GK")]
aa18 = aa18.sort_values("overall" , ascending = False)
aa18 = aa18.head(40).reset_index(drop = True)
aa18["year"] = 2018

aa18 = pd.melt(frame = aa18 , id_vars = ["long_name" , "player_positions" , "overall" , "year"] , value_vars = x , var_name = "skill" , value_name = "value")


aa21 = df21[["long_name" , "player_positions" , "overall" , "pace" , "shooting" , "passing" , "dribbling" , "defending" , "physic"]]
aa21 = aa21.loc[~aa21["player_positions"].str.contains("GK")]
aa21 = aa21.sort_values("overall" , ascending = False)
aa21 = aa21.head(40).reset_index(drop = True)
aa21["year"] = 2021

aa21 = pd.melt(frame = aa21 , id_vars = ["long_name" , "player_positions" , "overall" , "year"] , value_vars = x , var_name = "skill" , value_name = "value")


aa = aa15.append(aa18).append(aa21).reset_index(drop = True)

sns.set(font_scale = 3)
sns.set_theme(style="whitegrid" , color_codes=True)

g = sns.catplot(x = "skill" , y = "value" , kind = "violin" , data = aa ,
            hue = "year" , aspect = 1.5)

g.set(xlabel ="Skill Parameter", ylabel = "" , title ='Skill level distribution of the 40 best players of each year')

g.fig.set_figwidth(16)
g.fig.set_figheight(9)

plt.savefig('outputs\question1.jpg', format='jpeg', dpi=300)

plt.close()
#%% 2

"""
2. Represent with plot/s how the age of the 100 best players has been changing through the years.
"""

aa15 = df15[["long_name" , "overall" , "age"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa15["year"] = 2015
bb15 = df15[["long_name" , "overall" , "age"]].sort_values("overall" , ascending = False).sample(100).reset_index(drop = True)
bb15["year"] = 2015

aa16 = df16[["long_name" , "overall" , "age"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa16["year"] = 2016
bb16 = df16[["long_name" , "overall" , "age"]].sort_values("overall" , ascending = False).sample(100).reset_index(drop = True)
bb16["year"] = 2016

aa17 = df17[["long_name" , "overall" , "age"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa17["year"] = 2017
bb17 = df17[["long_name" , "overall" , "age"]].sort_values("overall" , ascending = False).sample(100).reset_index(drop = True)
bb17["year"] = 2017

aa18 = df18[["long_name" , "overall" , "age"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa18["year"] = 2018
bb18 = df18[["long_name" , "overall" , "age"]].sort_values("overall" , ascending = False).sample(100).reset_index(drop = True)
bb18["year"] = 2018

aa19 = df19[["long_name" , "overall" , "age"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa19["year"] = 2019
bb19 = df19[["long_name" , "overall" , "age"]].sort_values("overall" , ascending = False).sample(100).reset_index(drop = True)
bb19["year"] = 2019

aa20 = df20[["long_name" , "overall" , "age"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa20["year"] = 2020
bb20 = df20[["long_name" , "overall" , "age"]].sort_values("overall" , ascending = False).sample(100).reset_index(drop = True)
bb20["year"] = 2020

aa21 = df21[["long_name" , "overall" , "age"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa21["year"] = 2021
bb21 = df21[["long_name" , "overall" , "age"]].sort_values("overall" , ascending = False).sample(100).reset_index(drop = True)
bb21["year"] = 2021

aa = aa15.append(aa16).append(aa17).append(aa18).append(aa19).append(aa20).append(aa21).reset_index(drop = True)
aa["top"] = "yes"

bb = bb15.append(bb16).append(bb17).append(bb18).append(bb19).append(bb20).append(bb21).reset_index(drop = True)
bb["top"] = "no"

cc = aa.append(bb).reset_index(drop = True)

#%% 2 (first plot)

sns.set(font_scale = 2.2)
sns.set_theme(style="whitegrid" , color_codes=True)

g = sns.displot(data = aa , x = "age" , hue = "year", kind = "kde" , aspect = 1.5)

g.set(xlabel ="Age", ylabel = "Density" , title ='Age distribution of the 100 best players of each year')

g.fig.set_figwidth(16)
g.fig.set_figheight(9)

plt.savefig('outputs\question2.1.jpg', format='jpeg', dpi=300)

plt.close()
#%% 2 (second plot)

sns.set(font_scale = 1)
sns.set_theme(style="white" , color_codes=True)

g = sns.displot(data = cc , palette = "pastel" , x = "age" , hue = "top" , kind = "kde" , col = "year" , col_wrap = 4)


g.set_titles("Are the best players also older?")
g.set_xlabels("Age")
g.set_ylabels("Desnsity")
g.set_xticklabels(labels = [12,16,20,24,28,32,36,40,44,48])

#g.set(xlabel ="Age", ylabel = "Density" , title ='Are the best players also older?')

g.fig.set_figwidth(16)
g.fig.set_figheight(9)

plt.savefig('outputs\question2.2.jpg' , format = 'jpeg' , dpi = 300)

plt.close()
#%% 3 (top players)

"""
3. What are the most used team_jersey_number's among the 100 best players and the 100 worst players in the period between 2015 and 2021. And finally among all the players.
Bar plot of the top 6 most used team_jersey_number among the top 100 players and all the players.
Bar plot of the top 6 least used team_jersey_number from 1-99 among all the players
"""

aa15 = df15[["long_name" , "overall" , "team_jersey_number"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa15["year"] = 2015

aa16 = df16[["long_name" , "overall" , "team_jersey_number"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa16["year"] = 2016

aa17 = df17[["long_name" , "overall" , "team_jersey_number"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa17["year"] = 2017

aa18 = df18[["long_name" , "overall" , "team_jersey_number"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa18["year"] = 2018

aa19 = df19[["long_name" , "overall" , "team_jersey_number"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa19["year"] = 2019

aa20 = df20[["long_name" , "overall" , "team_jersey_number"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa20["year"] = 2020

aa21 = df21[["long_name" , "overall" , "team_jersey_number"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa21["year"] = 2021

aa = aa15.append(aa16).append(aa17).append(aa18).append(aa19).append(aa20).append(aa21).reset_index(drop = True)

plt.scatter(x = aa["team_jersey_number"] , y = aa["year"] , alpha = 0.15 , c = aa["year"])

plt.title("Which are the Jersey Numbers picked by the best 100 players?")
plt.xlabel("Jersey Number")
plt.xticks([0,10,20,30,40,60,80,100])

plt.savefig("outputs/question3.1.jpg" , dpi = 300)

plt.close()
#%% 3 (random players)

aa15 = df15[["long_name" , "overall" , "team_jersey_number"]].sort_values("overall" , ascending = False).sample(100).reset_index(drop = True)
aa15["year"] = 2015

aa16 = df16[["long_name" , "overall" , "team_jersey_number"]].sort_values("overall" , ascending = False).sample(100).reset_index(drop = True)
aa16["year"] = 2016

aa17 = df17[["long_name" , "overall" , "team_jersey_number"]].sort_values("overall" , ascending = False).sample(100).reset_index(drop = True)
aa17["year"] = 2017

aa18 = df18[["long_name" , "overall" , "team_jersey_number"]].sort_values("overall" , ascending = False).sample(100).reset_index(drop = True)
aa18["year"] = 2018

aa19 = df19[["long_name" , "overall" , "team_jersey_number"]].sort_values("overall" , ascending = False).sample(100).reset_index(drop = True)
aa19["year"] = 2019

aa20 = df20[["long_name" , "overall" , "team_jersey_number"]].sort_values("overall" , ascending = False).sample(100).reset_index(drop = True)
aa20["year"] = 2020

aa21 = df21[["long_name" , "overall" , "team_jersey_number"]].sort_values("overall" , ascending = False).sample(100).reset_index(drop = True)
aa21["year"] = 2021

aa = aa15.append(aa16).append(aa17).append(aa18).append(aa19).append(aa20).append(aa21).reset_index(drop = True)

plt.scatter(x = aa["team_jersey_number"] , y = aa["year"] , alpha = 0.15 , c = aa["year"])

plt.title("Which are the Jersey Numbers picked by the players?")
plt.xlabel("Jersey Number")
plt.xticks([0,10,20,30,40,60,80,100])

plt.savefig("outputs/question3.2.jpg" , dpi = 300)

plt.close()
#%% 3 (html)

aa15 = df15[["long_name" , "overall" , "team_jersey_number"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa15["year"] = 2015

aa16 = df16[["long_name" , "overall" , "team_jersey_number"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa16["year"] = 2016

aa17 = df17[["long_name" , "overall" , "team_jersey_number"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa17["year"] = 2017

aa18 = df18[["long_name" , "overall" , "team_jersey_number"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa18["year"] = 2018

aa19 = df19[["long_name" , "overall" , "team_jersey_number"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa19["year"] = 2019

aa20 = df20[["long_name" , "overall" , "team_jersey_number"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa20["year"] = 2020

aa21 = df21[["long_name" , "overall" , "team_jersey_number"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa21["year"] = 2021

aa = aa15.append(aa16).append(aa17).append(aa18).append(aa19).append(aa20).append(aa21).reset_index(drop = True)
aa["count"] = 1

aa1 = aa.groupby(["team_jersey_number"] , as_index = False).sum(["count"])[["team_jersey_number" , "count"]].sort_values("count" , ascending = False).head(6).reset_index(drop = True)
aa1 = aa1["team_jersey_number"].to_list()

aa = aa.loc[aa["team_jersey_number"].isin(aa1)]
aa = aa.groupby(["team_jersey_number" , "year"] , as_index = False).sum(["count"])[["team_jersey_number" , "year" , "count"]]
aa["team_jersey_number"] = aa.team_jersey_number.astype(int)
aa['team_jersey_number'] = aa.team_jersey_number.astype(str)
aa['year'] = aa.year.astype(str)

g = px.bar(aa , x = "team_jersey_number" , y = "count" , color = "year" , barmode = "group" ,
          color_discrete_sequence = px.colors.sequential.RdBu , opacity=0.9 ,
          hover_name = "year" , title = "Top 6 most used Team Jersey among the 100 best players" ,
          labels = dict(team_jersey_number = "Jersey Number", count = "Number of<br>players" , year = "Year"))

g.update_traces(hovertemplate='<br>Num of players: %{y}<br>Jersey Number: %{x}')

#g.update_xaxes(title_text = "Jersey Number")
#g.update_yaxes(title_text = "Number of<br>players")
#g.update_layout(hoverlabel = dict(bgcolor = "white" , font_size = 16 , font_family = "Rockwell"))

g.write_html("outputs/question3.3.html")

#%% 4 (centered)

"""
4. What are the most common team_position's among the 100 best players and the 100 worst players. And finally among all the players. (CENTERED WAY)
"""
"""
aa15 = df15[["long_name" , "overall" , "player_positions"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa15["year"] = 2015

aa16 = df16[["long_name" , "overall" , "player_positions"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa16["year"] = 2016

aa17 = df17[["long_name" , "overall" , "player_positions"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa17["year"] = 2017

aa18 = df18[["long_name" , "overall" , "player_positions"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa18["year"] = 2018

aa19 = df19[["long_name" , "overall" , "player_positions"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa19["year"] = 2019

aa20 = df20[["long_name" , "overall" , "player_positions"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa20["year"] = 2020

aa21 = df21[["long_name" , "overall" , "player_positions"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa21["year"] = 2021

aa = aa15.append(aa16).append(aa17).append(aa18).append(aa19).append(aa20).append(aa21).reset_index(drop = True)

aa1 = pd.DataFrame(columns = ["long_name" , "overall" , "player_positions" , "year" , "player_position"])

pos = ["CAM" , "CB" , "CDM" , "CF" , "CM" , "GK" , "LB" , "LM" , "LW" , "LWB" , "RB" , "RM" ,
       "RW" , "RWB" , "ST"]
year = [2015 , 2016 , 2017 , 2018 , 2019 , 2020 , 2021]

for y in year:
    for x in pos:
        zz = aa.loc[aa["year"] == y].loc[aa["player_positions"].str.contains("^"+x , flags = re.I , regex = True)]
        zz["player_position"] = x
        aa1 = aa1.append(zz)

aa2 = pd.DataFrame(columns = ["long_name" , "overall" , "player_position" , "year" , "x_pos_code"])
aa3 = pd.DataFrame(columns = ["long_name" , "overall" , "player_position" , "year" , "y_pos_code"])

listx = {"GK":2.0  ,"LCB":6 , "LB":6 , "RCB":6 , "RB":6 , "CB":6 , "LWB":9 , "CDM":9 , "RWB":9 , "LM":12.5 ,
         "CM":12.5 , "RM":12.5 , "CAM":17 , "LW":19 , "RW":19 , "CF":20 , "LF":21 , "LS":21 , "RF":21 , "RS":21 , "ST":22}

listy = {"GK":7.5  ,"LCB":11.5 , "LB":13 , "RCB":3.5 , "RB":2 , "CB":7.5 , "LWB":14 , "CDM":7.5 , "RWB":1 , "LM":14.5 ,
         "CM":7.5 , "RM":0.5 , "CAM":7.5 , "LW":13 , "RW":2 , "CF":7.5 , "LF":11.5 , "LS":11.5 , "RF":3.5 , "RS":3.5 , "ST":7.5}

for i in listx:
    zz = aa1.loc[aa1["player_position"] == i]
    zz["x_pos_code"] = listx[i]
    zz = zz[["long_name" , "overall" , "player_position" , "year" , "x_pos_code"]]
    aa2 = aa2.append(zz)

for i in listy:
    zz = aa1.loc[aa1["player_position"] == i]
    zz["y_pos_code"] = listy[i]
    zz = zz[["long_name" , "overall" , "player_position" , "year" , "y_pos_code"]]
    aa3 = aa3.append(zz)

result = pd.merge(aa2, aa3, on = ["long_name" , "overall" , "player_position" , "year"])
"""

#%% 4 (top players)

"""
4. What are the most common team_position's among the 100 best players and the 100 worst players. And finally among all the players. (NOT CENTERED WAY)
"""

aa15 = df15[["long_name" , "overall" , "player_positions"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa15["year"] = 2015

aa16 = df16[["long_name" , "overall" , "player_positions"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa16["year"] = 2016

aa17 = df17[["long_name" , "overall" , "player_positions"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa17["year"] = 2017

aa18 = df18[["long_name" , "overall" , "player_positions"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa18["year"] = 2018

aa19 = df19[["long_name" , "overall" , "player_positions"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa19["year"] = 2019

aa20 = df20[["long_name" , "overall" , "player_positions"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa20["year"] = 2020

aa21 = df21[["long_name" , "overall" , "player_positions"]].sort_values("overall" , ascending = False).head(100).reset_index(drop = True)
aa21["year"] = 2021

aa = aa15.append(aa16).append(aa17).append(aa18).append(aa19).append(aa20).append(aa21).reset_index(drop = True)

aa1 = pd.DataFrame(columns = ["long_name" , "overall" , "player_positions" , "year" , "player_position"])

pos = ["CAM" , "CB" , "CDM" , "CF" , "CM" , "GK" , "LB" , "LM" , "LW" , "LWB" , "RB" , "RM" ,
       "RW" , "RWB" , "ST"]
year = [2015 , 2016 , 2017 , 2018 , 2019 , 2020 , 2021]

for y in year:
    for x in pos:
        zz = aa.loc[aa["year"] == y].loc[aa["player_positions"].str.contains("^"+x , flags = re.I , regex = True)]
        zz["player_position"] = x
        aa1 = aa1.append(zz)

aa2 = pd.DataFrame(columns = ["long_name" , "overall" , "player_position" , "year" , "x_pos_code"])
aa3 = pd.DataFrame(columns = ["long_name" , "overall" , "player_position" , "year" , "y_pos_code"])

listx = {"GK":2.0  ,"LCB":6 , "LB":6 , "RCB":6 , "RB":6 , "CB":6 , "LWB":9 , "CDM":9.5 , "RWB":9 , "LM":12.5 ,
         "CM":12.5 , "RM":12.5 , "CAM":17 , "LW":19 , "RW":19 , "CF":20 , "LF":21 , "LS":21 , "RF":21 , "RS":21 , "ST":22}

listy = {"GK":7.5  ,"LCB":11.5 , "LB":13 , "RCB":3.5 , "RB":2 , "CB":7.5 , "LWB":14 , "CDM":7.5 , "RWB":1 , "LM":14.5 ,
         "CM":7.5 , "RM":0.5 , "CAM":7.5 , "LW":13 , "RW":2 , "CF":7.5 , "LF":11.5 , "LS":11.5 , "RF":3.5 , "RS":3.5 , "ST":7.5}

for i in listx:
    zz = aa1.loc[aa1["player_position"] == i]
    zz["x_pos_code"] = listx[i]
    zz = zz[["long_name" , "overall" , "player_position" , "year" , "x_pos_code"]]
    aa2 = aa2.append(zz)
for i in listy:
    if i == "CB":
        zz = aa1.loc[aa1["player_position"] == i]
        fakeint = int(len(zz)/2)
        fakefloat = len(zz)/2
        
        if fakeint == fakefloat:
            zz1 = zz.head(int(len(zz)/2))
            zz1["y_pos_code"] = 5.5
            zz2 = zz.tail(int(len(zz)/2))
            zz2["y_pos_code"] = 9.5
            zz = zz1.append(zz2)
            zz = zz[["long_name" , "overall" , "player_position" , "year" , "y_pos_code"]]
            aa3 = aa3.append(zz)
        else:
            zz1 = zz.head(int(len(zz)/2)+1)
            zz1["y_pos_code"] = 5.5
            zz2 = zz.tail(int(len(zz)/2))
            zz2["y_pos_code"] = 9.5
            zz = zz1.append(zz2)
            zz = zz[["long_name" , "overall" , "player_position" , "year" , "y_pos_code"]]
            aa3 = aa3.append(zz)
        
    elif i == "CM":
        zz = aa1.loc[aa1["player_position"] == i]
        fakeint = int(len(zz)/2)
        fakefloat = len(zz)/2
        
        if fakeint == fakefloat:
            zz1 = zz.head(int(len(zz)/2))
            zz1["y_pos_code"] = 5
            zz2 = zz.tail(int(len(zz)/2))
            zz2["y_pos_code"] = 10
            zz = zz1.append(zz2)
            zz = zz[["long_name" , "overall" , "player_position" , "year" , "y_pos_code"]]
            aa3 = aa3.append(zz)
        else:
            zz1 = zz.head(int(len(zz)/2)+1)
            zz1["y_pos_code"] = 5
            zz2 = zz.tail(int(len(zz)/2))
            zz2["y_pos_code"] = 10
            zz = zz1.append(zz2)
            zz = zz[["long_name" , "overall" , "player_position" , "year" , "y_pos_code"]]
            aa3 = aa3.append(zz)
    else:
        zz = aa1.loc[aa1["player_position"] == i]
        zz["y_pos_code"] = listy[i]
        zz = zz[["long_name" , "overall" , "player_position" , "year" , "y_pos_code"]]
        aa3 = aa3.append(zz)

result = pd.merge(aa2, aa3, on = ["long_name" , "overall" , "player_position" , "year"])

img = plt.imread("resources/football_field.jpg")
fig, ax = plt.subplots()
ax.imshow(img, extent=[-2.05, 27, -2, 17])
my = sns.kdeplot(x = result.x_pos_code , y = result.y_pos_code , cmap = "Reds" , shade = True , bw_adjust = 0.6 , alpha = 0.55)
my.set(xticklabels=[])
my.set(xlabel=None)
my.set(yticklabels=[])
my.set(ylabel=None)


plt.savefig('outputs\question4.1.jpg' , format = 'jpeg' , dpi = 300 , bbox_inches = 'tight')
plt.close()
#%% 4 (random players)

"""
AMONG ALL THE PLAYERS
"""

aa15 = df15[["long_name" , "overall" , "player_positions"]].sort_values("overall" , ascending = False).sample(100).reset_index(drop = True)
aa15["year"] = 2015

aa16 = df16[["long_name" , "overall" , "player_positions"]].sort_values("overall" , ascending = False).sample(100).reset_index(drop = True)
aa16["year"] = 2016

aa17 = df17[["long_name" , "overall" , "player_positions"]].sort_values("overall" , ascending = False).sample(100).reset_index(drop = True)
aa17["year"] = 2017

aa18 = df18[["long_name" , "overall" , "player_positions"]].sort_values("overall" , ascending = False).sample(100).reset_index(drop = True)
aa18["year"] = 2018

aa19 = df19[["long_name" , "overall" , "player_positions"]].sort_values("overall" , ascending = False).sample(100).reset_index(drop = True)
aa19["year"] = 2019

aa20 = df20[["long_name" , "overall" , "player_positions"]].sort_values("overall" , ascending = False).sample(100).reset_index(drop = True)
aa20["year"] = 2020

aa21 = df21[["long_name" , "overall" , "player_positions"]].sort_values("overall" , ascending = False).sample(100).reset_index(drop = True)
aa21["year"] = 2021

aa = aa15.append(aa16).append(aa17).append(aa18).append(aa19).append(aa20).append(aa21).reset_index(drop = True)

aa1 = pd.DataFrame(columns = ["long_name" , "overall" , "player_positions" , "year" , "player_position"])

pos = ["CAM" , "CB" , "CDM" , "CF" , "CM" , "GK" , "LB" , "LM" , "LW" , "LWB" , "RB" , "RM" ,
       "RW" , "RWB" , "ST"]
year = [2015 , 2016 , 2017 , 2018 , 2019 , 2020 , 2021]

for y in year:
    for x in pos:
        zz = aa.loc[aa["year"] == y].loc[aa["player_positions"].str.contains("^"+x , flags = re.I , regex = True)]
        zz["player_position"] = x
        aa1 = aa1.append(zz)

aa2 = pd.DataFrame(columns = ["long_name" , "overall" , "player_position" , "year" , "x_pos_code"])
aa3 = pd.DataFrame(columns = ["long_name" , "overall" , "player_position" , "year" , "y_pos_code"])

listx = {"GK":2.0  ,"LCB":6 , "LB":6 , "RCB":6 , "RB":6 , "CB":6 , "LWB":9 , "CDM":9.5 , "RWB":9 , "LM":12.5 ,
         "CM":12.5 , "RM":12.5 , "CAM":17 , "LW":19 , "RW":19 , "CF":20 , "LF":21 , "LS":21 , "RF":21 , "RS":21 , "ST":22}

listy = {"GK":7.5  ,"LCB":11.5 , "LB":13 , "RCB":3.5 , "RB":2 , "CB":7.5 , "LWB":14 , "CDM":7.5 , "RWB":1 , "LM":14.5 ,
         "CM":7.5 , "RM":0.5 , "CAM":7.5 , "LW":13 , "RW":2 , "CF":7.5 , "LF":11.5 , "LS":11.5 , "RF":3.5 , "RS":3.5 , "ST":7.5}

for i in listx:
    zz = aa1.loc[aa1["player_position"] == i]
    zz["x_pos_code"] = listx[i]
    zz = zz[["long_name" , "overall" , "player_position" , "year" , "x_pos_code"]]
    aa2 = aa2.append(zz)
for i in listy:
    if i == "CB":
        zz = aa1.loc[aa1["player_position"] == i]
        fakeint = int(len(zz)/2)
        fakefloat = len(zz)/2
        
        if fakeint == fakefloat:
            zz1 = zz.head(int(len(zz)/2))
            zz1["y_pos_code"] = 5.5
            zz2 = zz.tail(int(len(zz)/2))
            zz2["y_pos_code"] = 9.5
            zz = zz1.append(zz2)
            zz = zz[["long_name" , "overall" , "player_position" , "year" , "y_pos_code"]]
            aa3 = aa3.append(zz)
        else:
            zz1 = zz.head(int(len(zz)/2)+1)
            zz1["y_pos_code"] = 5.5
            zz2 = zz.tail(int(len(zz)/2))
            zz2["y_pos_code"] = 9.5
            zz = zz1.append(zz2)
            zz = zz[["long_name" , "overall" , "player_position" , "year" , "y_pos_code"]]
            aa3 = aa3.append(zz)
        
    elif i == "CM":
        zz = aa1.loc[aa1["player_position"] == i]
        fakeint = int(len(zz)/2)
        fakefloat = len(zz)/2
        
        if fakeint == fakefloat:
            zz1 = zz.head(int(len(zz)/2))
            zz1["y_pos_code"] = 5
            zz2 = zz.tail(int(len(zz)/2))
            zz2["y_pos_code"] = 10
            zz = zz1.append(zz2)
            zz = zz[["long_name" , "overall" , "player_position" , "year" , "y_pos_code"]]
            aa3 = aa3.append(zz)
        else:
            zz1 = zz.head(int(len(zz)/2)+1)
            zz1["y_pos_code"] = 5
            zz2 = zz.tail(int(len(zz)/2))
            zz2["y_pos_code"] = 10
            zz = zz1.append(zz2)
            zz = zz[["long_name" , "overall" , "player_position" , "year" , "y_pos_code"]]
            aa3 = aa3.append(zz)
    else:
        zz = aa1.loc[aa1["player_position"] == i]
        zz["y_pos_code"] = listy[i]
        zz = zz[["long_name" , "overall" , "player_position" , "year" , "y_pos_code"]]
        aa3 = aa3.append(zz)

result = pd.merge(aa2, aa3, on = ["long_name" , "overall" , "player_position" , "year"])

img = plt.imread("resources/football_field.jpg")
fig, ax = plt.subplots()
ax.imshow(img, extent=[-2.05, 27, -2, 17])
my = sns.kdeplot(x = result.x_pos_code , y = result.y_pos_code , cmap = "Reds" , shade = True , bw_adjust = 0.6 , alpha = 0.55)
my.set(xticklabels=[])
my.set(xlabel=None)
my.set(yticklabels=[])
my.set(ylabel=None)


plt.savefig('outputs\question4.2.jpg' , format = 'jpeg' , dpi = 300 , bbox_inches = 'tight')
plt.close()

#%% 5

"""
5. Represent the variables (age , height_cm , weight_kg) for the best 50 field players in a table.
"""

aa15 = df15[["long_name" , "overall" , "age", "height_cm" , "weight_kg"]].sort_values("overall" , ascending = False).head(50).reset_index(drop = True)
aa15["year"] = 2015

aa16 = df16[["long_name" , "overall" , "age", "height_cm" , "weight_kg"]].sort_values("overall" , ascending = False).head(50).reset_index(drop = True)
aa16["year"] = 2016

aa17 = df17[["long_name" , "overall" , "age", "height_cm" , "weight_kg"]].sort_values("overall" , ascending = False).head(50).reset_index(drop = True)
aa17["year"] = 2017

aa18 = df18[["long_name" , "overall" , "age", "height_cm" , "weight_kg"]].sort_values("overall" , ascending = False).head(50).reset_index(drop = True)
aa18["year"] = 2018

aa19 = df19[["long_name" , "overall" , "age", "height_cm" , "weight_kg"]].sort_values("overall" , ascending = False).head(50).reset_index(drop = True)
aa19["year"] = 2019

aa20 = df20[["long_name" , "overall" , "age", "height_cm" , "weight_kg"]].sort_values("overall" , ascending = False).head(50).reset_index(drop = True)
aa20["year"] = 2020

aa21 = df21[["long_name" , "overall" , "age", "height_cm" , "weight_kg"]].sort_values("overall" , ascending = False).head(50).reset_index(drop = True)
aa21["year"] = 2021

aa = aa15.append(aa16).append(aa17).append(aa18).append(aa19).append(aa20).append(aa21).reset_index(drop = True)
aa = aa[["long_name" , "year" , "overall" , "age", "height_cm" , "weight_kg"]]

rowEvenColor = 'lightgrey'
rowOddColor = 'white'

g = go.Figure(data = [go.Table(
    header = dict(values = ["<b>NAME</b>" , "<b>Year</b>" , "<b>Overall</b>" , "<b>Age</b>" , "<b>Height</b>" , "<b>Weight<b>"],
                line_color = 'darkslategray',
                fill_color = "grey",
                font = dict(color = 'white' , size = 12),
                align = ["left" , "center"]),
    cells = dict(values = [aa.long_name , aa.year , aa.overall , aa.age , aa.height_cm , aa.weight_kg],
               fill_color = [[rowOddColor,rowEvenColor]*(int(len(aa)/2))],
               font = dict(color = 'darkslategray', size = 11),
               align = ["left" , "center"]))
])

g.write_html("outputs/question5.html")

"""
With Dash I will learn how to do this in an app
"""

#%% 6

"""
6. Regression about how the variables (29) (attacking_crossing , attacking_finishing , attacking_heading_accuracy , attacking_short_passing , attacking_volleys ,
					skill_dribbling , skill_curve , skill_fk_accuracy , skill_long_passing , skill_ball_control ,
					movement_acceleration , movement_sprint_speed , movement_agility , movement_reactions , movement_balance ,
					power_shot_power , power_jumping , power_stamina , power_strength , power_long_shots ,
					mentality_agression , mentality_interceptions , mentality_positioning , mentality_vision , mentality_penalties , mentality_composure ,
					defending_marking *(it looks like some values are missing) , defending_standing_tackle , defending_sliding_tackle)
	affect the global score. Use all players except for Goalkeepers as observations.
"""

aa15 = df15.loc[~df15["player_positions"].str.contains("GK")]
aa15 = aa15[["overall" , "attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" , "mentality_composure" ,
             "defending_marking", "defending_standing_tackle" , "defending_sliding_tackle"]].reset_index(drop = True)
aa15["year"] = 2015

aa16 = df16.loc[~df16["player_positions"].str.contains("GK")]
aa16 = aa16[["overall" , "attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" , "mentality_composure" ,
             "defending_marking", "defending_standing_tackle" , "defending_sliding_tackle"]].reset_index(drop = True)
aa16["year"] = 2016

aa17 = df17.loc[~df17["player_positions"].str.contains("GK")]
aa17 = aa17[["overall" , "attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" , "mentality_composure" ,
             "defending_marking", "defending_standing_tackle" , "defending_sliding_tackle"]].reset_index(drop = True)
aa17["year"] = 2017

aa18 = df18.loc[~df18["player_positions"].str.contains("GK")]
aa18 = aa18[["overall" , "attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" , "mentality_composure" ,
             "defending_marking", "defending_standing_tackle" , "defending_sliding_tackle"]].reset_index(drop = True)
aa18["year"] = 2018

aa19 = df19.loc[~df19["player_positions"].str.contains("GK")]
aa19 = aa19[["overall" , "attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" , "mentality_composure" ,
             "defending_marking", "defending_standing_tackle" , "defending_sliding_tackle"]].reset_index(drop = True)
aa19["year"] = 2019

aa20 = df20.loc[~df20["player_positions"].str.contains("GK")]
aa20 = aa20[["overall" , "attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" , "mentality_composure" ,
             "defending_marking", "defending_standing_tackle" , "defending_sliding_tackle"]].reset_index(drop = True)
aa20["year"] = 2020

aa21 = df21.loc[~df21["player_positions"].str.contains("GK")]
aa21 = aa21[["overall" , "attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" , "mentality_composure" ,
             "defending_marking", "defending_standing_tackle" , "defending_sliding_tackle"]].reset_index(drop = True)
aa21["year"] = 2021

aa = aa15.append(aa16).append(aa17).append(aa18).append(aa19).append(aa20).append(aa21).reset_index(drop = True)
aa = aa[["overall" , "attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" , "mentality_composure" ,
             "defending_marking", "defending_standing_tackle" , "defending_sliding_tackle"]]

#%% 6 Dropping NA for all years (it is the same as 17,18 and 19 together)
"""
But this is not well done because for the years 15, 16 there are not values for mentality_composture &
                                also fot the years 20, 21 there are not values for defending_marking.
So it is necessary to make different regressions for each year.
"""
aa = aa.dropna()

X = aa.drop("overall" , axis=1)

Y = aa.overall

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

Y_pred_train = model.predict(X_train)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_train, Y_pred_train))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_train, Y_pred_train))

Y_pred_test = model.predict(X_test)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred_test))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred_test))
"""
yintercept = '%.2f' % model.intercept_
LogP = '%.2f LogP' % model.coef_[0]
MW = '%.4f MW' % model.coef_[1]
RB = '%.4f RB' % model.coef_[2]
AP = '%.2f AP' % model.coef_[3]

print('LogS = ' + 
      ' ' + 
      yintercept + 
      ' ' + 
      LogP + 
      ' ' + 
      MW + 
      ' + ' + 
      RB + 
      ' ' + 
      AP)
"""

plt.figure(figsize=(5,11))

# 2 row, 1 column, plot 1
plt.subplot(2, 1, 1)
plt.scatter(x=Y_train, y=Y_pred_train, c="#7CAE00", alpha=0.05)

# Add trendline
# https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs
z = np.polyfit(Y_train, Y_pred_train, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted overall')


# 2 row, 1 column, plot 2
plt.subplot(2, 1, 2)
plt.scatter(x=Y_test, y=Y_pred_test, c="#619CFF", alpha=0.05)

z = np.polyfit(Y_test, Y_pred_test, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted overall')
plt.xlabel('Real overall')

plt.savefig('outputs/question6/question6.jpg', format='jpeg', dpi=300)
plt.close()

#%% 6 Dropping NA for aa15 (without mentality_composture)

aa15 = aa15[["overall" , "attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" ,
             "defending_marking", "defending_standing_tackle" , "defending_sliding_tackle"]].dropna()

X = aa15.drop("overall" , axis=1)

Y = aa15.overall

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

Y_pred_train = model.predict(X_train)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_train, Y_pred_train))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_train, Y_pred_train))

Y_pred_test = model.predict(X_test)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred_test))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred_test))
"""
yintercept = '%.2f' % model.intercept_
LogP = '%.2f LogP' % model.coef_[0]
MW = '%.4f MW' % model.coef_[1]
RB = '%.4f RB' % model.coef_[2]
AP = '%.2f AP' % model.coef_[3]

print('LogS = ' + 
      ' ' + 
      yintercept + 
      ' ' + 
      LogP + 
      ' ' + 
      MW + 
      ' + ' + 
      RB + 
      ' ' + 
      AP)
"""

plt.figure(figsize=(5,11))

# 2 row, 1 column, plot 1
plt.subplot(2, 1, 1)
plt.scatter(x=Y_train, y=Y_pred_train, c="#7CAE00", alpha=0.05)

# Add trendline
# https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs
z = np.polyfit(Y_train, Y_pred_train, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')


# 2 row, 1 column, plot 2
plt.subplot(2, 1, 2)
plt.scatter(x=Y_test, y=Y_pred_test, c="#619CFF", alpha=0.05)

z = np.polyfit(Y_test, Y_pred_test, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')

plt.savefig("outputs/question6/question6.15.jpg" , format = 'jpeg' , dpi=300)
plt.close()

#%% 6 Dropping NA for aa16 (without mentality_composture)

aa16 = aa16[["overall" , "attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" ,
             "defending_marking", "defending_standing_tackle" , "defending_sliding_tackle"]].dropna()

X = aa16.drop("overall" , axis=1)

Y = aa16.overall

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

Y_pred_train = model.predict(X_train)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_train, Y_pred_train))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_train, Y_pred_train))

Y_pred_test = model.predict(X_test)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred_test))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred_test))
"""
yintercept = '%.2f' % model.intercept_
LogP = '%.2f LogP' % model.coef_[0]
MW = '%.4f MW' % model.coef_[1]
RB = '%.4f RB' % model.coef_[2]
AP = '%.2f AP' % model.coef_[3]

print('LogS = ' + 
      ' ' + 
      yintercept + 
      ' ' + 
      LogP + 
      ' ' + 
      MW + 
      ' + ' + 
      RB + 
      ' ' + 
      AP)
"""

plt.figure(figsize=(5,11))

# 2 row, 1 column, plot 1
plt.subplot(2, 1, 1)
plt.scatter(x=Y_train, y=Y_pred_train, c="#7CAE00", alpha=0.05)

# Add trendline
# https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs
z = np.polyfit(Y_train, Y_pred_train, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')


# 2 row, 1 column, plot 2
plt.subplot(2, 1, 2)
plt.scatter(x=Y_test, y=Y_pred_test, c="#619CFF", alpha=0.05)

z = np.polyfit(Y_test, Y_pred_test, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')

plt.savefig("outputs/question6/question6.16.jpg" , format = 'jpeg' , dpi=300)
plt.close()

#%% 6 Dropping NA for aa17 (with all columns)

aa17 = aa17[["overall" , "attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" , "mentality_composure" ,
             "defending_marking", "defending_standing_tackle" , "defending_sliding_tackle"]].dropna()

X = aa17.drop("overall" , axis=1)

Y = aa17.overall

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

Y_pred_train = model.predict(X_train)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_train, Y_pred_train))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_train, Y_pred_train))

Y_pred_test = model.predict(X_test)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred_test))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred_test))
"""
yintercept = '%.2f' % model.intercept_
LogP = '%.2f LogP' % model.coef_[0]
MW = '%.4f MW' % model.coef_[1]
RB = '%.4f RB' % model.coef_[2]
AP = '%.2f AP' % model.coef_[3]

print('LogS = ' + 
      ' ' + 
      yintercept + 
      ' ' + 
      LogP + 
      ' ' + 
      MW + 
      ' + ' + 
      RB + 
      ' ' + 
      AP)
"""

plt.figure(figsize=(5,11))

# 2 row, 1 column, plot 1
plt.subplot(2, 1, 1)
plt.scatter(x=Y_train, y=Y_pred_train, c="#7CAE00", alpha=0.05)

# Add trendline
# https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs
z = np.polyfit(Y_train, Y_pred_train, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')


# 2 row, 1 column, plot 2
plt.subplot(2, 1, 2)
plt.scatter(x=Y_test, y=Y_pred_test, c="#619CFF", alpha=0.05)

z = np.polyfit(Y_test, Y_pred_test, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')

plt.savefig("outputs/question6/question6.17.jpg" , format = 'jpeg' , dpi=300)
plt.close()

#%% 6 Dropping NA for aa18 (with all columns)

aa18 = aa18[["overall" , "attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" , "mentality_composure" ,
             "defending_marking", "defending_standing_tackle" , "defending_sliding_tackle"]].dropna()

X = aa18.drop("overall" , axis=1)

Y = aa18.overall

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

Y_pred_train = model.predict(X_train)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_train, Y_pred_train))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_train, Y_pred_train))

Y_pred_test = model.predict(X_test)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred_test))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred_test))
"""
yintercept = '%.2f' % model.intercept_
LogP = '%.2f LogP' % model.coef_[0]
MW = '%.4f MW' % model.coef_[1]
RB = '%.4f RB' % model.coef_[2]
AP = '%.2f AP' % model.coef_[3]

print('LogS = ' + 
      ' ' + 
      yintercept + 
      ' ' + 
      LogP + 
      ' ' + 
      MW + 
      ' + ' + 
      RB + 
      ' ' + 
      AP)
"""

plt.figure(figsize=(5,11))

# 2 row, 1 column, plot 1
plt.subplot(2, 1, 1)
plt.scatter(x=Y_train, y=Y_pred_train, c="#7CAE00", alpha=0.05)

# Add trendline
# https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs
z = np.polyfit(Y_train, Y_pred_train, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')


# 2 row, 1 column, plot 2
plt.subplot(2, 1, 2)
plt.scatter(x=Y_test, y=Y_pred_test, c="#619CFF", alpha=0.05)

z = np.polyfit(Y_test, Y_pred_test, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')

plt.savefig("outputs/question6/question6.18.jpg" , format = 'jpeg' , dpi=300)
plt.close()


#%% 6 Dropping NA for aa19 (with all columns)

aa19 = aa19[["overall" , "attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" , "mentality_composure" ,
             "defending_marking", "defending_standing_tackle" , "defending_sliding_tackle"]].dropna()

X = aa19.drop("overall" , axis=1)

Y = aa19.overall

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

Y_pred_train = model.predict(X_train)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_train, Y_pred_train))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_train, Y_pred_train))

Y_pred_test = model.predict(X_test)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred_test))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred_test))
"""
yintercept = '%.2f' % model.intercept_
LogP = '%.2f LogP' % model.coef_[0]
MW = '%.4f MW' % model.coef_[1]
RB = '%.4f RB' % model.coef_[2]
AP = '%.2f AP' % model.coef_[3]

print('LogS = ' + 
      ' ' + 
      yintercept + 
      ' ' + 
      LogP + 
      ' ' + 
      MW + 
      ' + ' + 
      RB + 
      ' ' + 
      AP)
"""

plt.figure(figsize=(5,11))

# 2 row, 1 column, plot 1
plt.subplot(2, 1, 1)
plt.scatter(x=Y_train, y=Y_pred_train, c="#7CAE00", alpha=0.05)

# Add trendline
# https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs
z = np.polyfit(Y_train, Y_pred_train, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')


# 2 row, 1 column, plot 2
plt.subplot(2, 1, 2)
plt.scatter(x=Y_test, y=Y_pred_test, c="#619CFF", alpha=0.05)

z = np.polyfit(Y_test, Y_pred_test, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')

plt.savefig("outputs/question6/question6.19.jpg" , format = 'jpeg' , dpi=300)
plt.close()

#%% 6 Dropping NA for aa20 (without defending_marking)

aa20 = aa20[["overall" , "attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" , "mentality_composure" ,
             "defending_standing_tackle" , "defending_sliding_tackle"]].dropna()

X = aa20.drop("overall" , axis=1)

Y = aa20.overall

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

Y_pred_train = model.predict(X_train)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_train, Y_pred_train))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_train, Y_pred_train))

Y_pred_test = model.predict(X_test)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred_test))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred_test))
"""
yintercept = '%.2f' % model.intercept_
LogP = '%.2f LogP' % model.coef_[0]
MW = '%.4f MW' % model.coef_[1]
RB = '%.4f RB' % model.coef_[2]
AP = '%.2f AP' % model.coef_[3]

print('LogS = ' + 
      ' ' + 
      yintercept + 
      ' ' + 
      LogP + 
      ' ' + 
      MW + 
      ' + ' + 
      RB + 
      ' ' + 
      AP)
"""

plt.figure(figsize=(5,11))

# 2 row, 1 column, plot 1
plt.subplot(2, 1, 1)
plt.scatter(x=Y_train, y=Y_pred_train, c="#7CAE00", alpha=0.05)

# Add trendline
# https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs
z = np.polyfit(Y_train, Y_pred_train, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')


# 2 row, 1 column, plot 2
plt.subplot(2, 1, 2)
plt.scatter(x=Y_test, y=Y_pred_test, c="#619CFF", alpha=0.05)

z = np.polyfit(Y_test, Y_pred_test, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')

plt.savefig("outputs/question6/question6.20.jpg" , format = 'jpeg' , dpi=300)
plt.close()

#%% 6 Dropping NA for aa21 (without defending_marking)

aa21 = aa21[["overall" , "attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" , "mentality_composure" ,
             "defending_standing_tackle" , "defending_sliding_tackle"]].dropna()

X = aa21.drop("overall" , axis=1)

Y = aa21.overall

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

Y_pred_train = model.predict(X_train)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_train, Y_pred_train))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_train, Y_pred_train))

Y_pred_test = model.predict(X_test)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred_test))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred_test))
"""
yintercept = '%.2f' % model.intercept_
LogP = '%.2f LogP' % model.coef_[0]
MW = '%.4f MW' % model.coef_[1]
RB = '%.4f RB' % model.coef_[2]
AP = '%.2f AP' % model.coef_[3]

print('LogS = ' + 
      ' ' + 
      yintercept + 
      ' ' + 
      LogP + 
      ' ' + 
      MW + 
      ' + ' + 
      RB + 
      ' ' + 
      AP)
"""

plt.figure(figsize=(5,11))

# 2 row, 1 column, plot 1
plt.subplot(2, 1, 1)
plt.scatter(x=Y_train, y=Y_pred_train, c="#7CAE00", alpha=0.05)

# Add trendline
# https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs
z = np.polyfit(Y_train, Y_pred_train, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')


# 2 row, 1 column, plot 2
plt.subplot(2, 1, 2)
plt.scatter(x=Y_test, y=Y_pred_test, c="#619CFF", alpha=0.05)

z = np.polyfit(Y_test, Y_pred_test, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')

plt.savefig("outputs/question6/question6.21.jpg" , format = 'jpeg' , dpi=300)
plt.close()

#%% 6 one model for each position
"""
Lets see if we get a better regression model if we divide the players depending on the position in which they play:
    one different model for every each position year 2021
"""

aa21 = df21[["overall" , "player_positions" , "attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" , "mentality_composure" ,
             "defending_standing_tackle" , "defending_sliding_tackle"]].reset_index(drop = True)

pos = ["CAM" , "CB" , "CDM" , "CF" , "CM" , "GK" , "LB" , "LM" , "LW" , "LWB" , "RB" , "RM" ,
       "RW" , "RWB" , "ST"]

aa1 = pd.DataFrame(columns = ["overall" , "player_position" , "attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" , "mentality_composure" ,
             "defending_standing_tackle" , "defending_sliding_tackle"])

for x in pos:
        zz = aa21.loc[aa21["player_positions"].str.contains("^"+x , flags = re.I , regex = True)]
        zz["player_position"] = x
        aa1 = aa1.append(zz)

aa = aa1[["overall" , "player_position" , "attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" , "mentality_composure" ,
             "defending_standing_tackle" , "defending_sliding_tackle"]].reset_index(drop = True)

aa = aa.loc[aa["player_position"] == "CAM"]

aa = aa.dropna()

X = aa.drop(["overall" , "player_position"] , axis=1)

Y = aa.overall

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

Y_pred_train = model.predict(X_train)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_train, Y_pred_train))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_train, Y_pred_train))

Y_pred_test = model.predict(X_test)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred_test))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred_test))
"""
yintercept = '%.2f' % model.intercept_
LogP = '%.2f LogP' % model.coef_[0]
MW = '%.4f MW' % model.coef_[1]
RB = '%.4f RB' % model.coef_[2]
AP = '%.2f AP' % model.coef_[3]

print('LogS = ' + 
      ' ' + 
      yintercept + 
      ' ' + 
      LogP + 
      ' ' + 
      MW + 
      ' + ' + 
      RB + 
      ' ' + 
      AP)
"""

plt.figure(figsize=(5,11))

# 2 row, 1 column, plot 1
plt.subplot(2, 1, 1)

# Add trendline
# https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs
z = np.polyfit(Y_train.astype(str).astype(float), Y_pred_train, 1)
p = np.poly1d(z)

plt.scatter(x=Y_train, y=Y_pred_train, c="#7CAE00", alpha=0.05)

plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')


# 2 row, 1 column, plot 2
plt.subplot(2, 1, 2)

z = np.polyfit(Y_test.astype(str).astype(float), Y_pred_test, 1)
p = np.poly1d(z)

plt.scatter(x=Y_test, y=Y_pred_test, c="#619CFF", alpha=0.1)

plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')

plt.savefig("outputs/question6/question6.CAM.jpg" , format = 'jpeg' , dpi=300)
plt.close()

#%% TRY TO PREDICT IF A PLAYER IS GK AND DO THE SAME FOR CB, CM AND ST (LOGISTIC REGRESSION)

# Maybe it is good to activate warnings back again. They are unactivated because it is easier for visualization
import warnings
warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


df = df15[["attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" ,
             "defending_standing_tackle" , "defending_sliding_tackle" ,
             "goalkeeping_diving" , "goalkeeping_handling" , "goalkeeping_kicking" , "goalkeeping_positioning" , "goalkeeping_reflexes" ,
             "player_positions"]].dropna()

pos = ["CB"]
#pos = ["GK" , "CB" , "CM" , "ST"]

for i in pos:

    bb = df.loc[df["player_positions"].str.contains(i)]
    bb["player_position"] = 1
    bb = bb.drop(["player_positions"] , axis = 1)
    cc = df.loc[~df["player_positions"].str.contains(i)]
    cc["player_position"] = 0
    cc = cc.drop(["player_positions"] , axis = 1)
    
    aa = bb.append(cc)
    
    data = aa.drop(["player_position"] , axis = 1)
    target = aa.player_position
    
    X = data
    Y = target
    
    # Logistic regression implementation
    
    # We divide the "train" data in training and testing in order to test the algorithms
    
    X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2)
    
    # We scale the data
    
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    
    # Define the algorithm that we are going to use
    
    algorithm = LogisticRegression()
    
    # We train the model
    
    algorithm.fit(X_train , Y_train)
    
    # We realize the prediction
    
    Y_pred = algorithm.predict(X_test)
    
    # We verify the confusion matrix
    
    matrix = confusion_matrix(Y_test , Y_pred)
    
    print("Confusion matrix")
    print(matrix)
    
    # We calculate the precision of the model
    
    precision = precision_score(Y_test , Y_pred)
    
    # We calculate the accuracy of the model
    
    accuracy = accuracy_score(Y_test , Y_pred)
    
    # We calculate the recall of the model
    
    recall = recall_score(Y_test , Y_pred)
    
    # We calculate F1 score
    
    scoref1 = f1_score(Y_test , Y_pred)
    
    # We calculate the ROC - AUC score of the model
    
    roc_auc = roc_auc_score(Y_test , Y_pred)
    print("Model's ROC - AUC score:")
    print(roc_auc)
    
    print("Model's precision for" , i , "is:" , precision)
    print("Model's accuracy for" , i , "is:" , accuracy)
    print("Model's recall for" , i , "is:" , recall)
    print("Model's F1 score for" , i , "is:" , scoref1)
    print("Model's ROC - AUC score for" , i , "is:" , roc_auc)

# Second part

df = df15.append(df16).append(df17).append(df18).append(df19).append(df20).append(df21)

df = df[["attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" ,
             "defending_standing_tackle" , "defending_sliding_tackle" ,
             "goalkeeping_diving" , "goalkeeping_handling" , "goalkeeping_kicking" , "goalkeeping_positioning" , "goalkeeping_reflexes" ,
             "player_positions"]].dropna()

pos = ["ST"]
#pos = ["GK" , "CB" , "CM" , "ST"]

for i in pos:

    bb = df.loc[df["player_positions"].str.contains(i)]
    bb["player_position"] = 1
    bb = bb.drop(["player_positions"] , axis = 1)
    cc = df.loc[~df["player_positions"].str.contains(i)]
    cc["player_position"] = 0
    cc = cc.drop(["player_positions"] , axis = 1)
    
    aa = bb.append(cc)
    
    data = aa.drop(["player_position"] , axis = 1)
    target = aa.player_position
    
    X = data
    Y = target
    
    # Logistic regression implementation
    
    # We divide the "train" data in training and testing in order to test the algorithms
    
    X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2)
    
    # We scale the data
    
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    
    # Define the algorithm that we are going to use
    
    algorithm = LogisticRegression()
    
    # We train the model
    
    algorithm.fit(X_train , Y_train)
    
    # We realize the prediction
    
    Y_pred = algorithm.predict(X_test)
    
    # We verify the confusion matrix
    
    matrix = confusion_matrix(Y_test , Y_pred)
    
    print("Confusion matrix")
    print(matrix)
    
    # We calculate the precision of the model
    
    precision = precision_score(Y_test , Y_pred)
    
    # We calculate the accuracy of the model
    
    accuracy = accuracy_score(Y_test , Y_pred)
    
    # We calculate the recall of the model
    
    recall = recall_score(Y_test , Y_pred)
    
    # We calculate F1 score
    
    scoref1 = f1_score(Y_test , Y_pred)
    
    # We calculate the ROC - AUC score of the model
    
    roc_auc = roc_auc_score(Y_test , Y_pred)
    print("Model's ROC - AUC score:")
    print(roc_auc)
    
    print("Model's precision for total" , i , "is:" , precision)
    print("Model's accuracy for total" , i , "is:" , accuracy)
    print("Model's recall for total" , i , "is:" , recall)
    print("Model's F1 score for total" , i , "is:" , scoref1)
    print("Model's ROC - AUC score for total" , i , "is:" , roc_auc)


#%% TRY TO PREDICT IF A PLAYER IS GK AND DO THE SAME FOR CB, CM AND ST (K NEAREST NEIGHTBOURS)

# Maybe it is good to activate warnings back again. They are unactivated because it is easier for visualization
import warnings
warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


df = df15[["attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" ,
             "defending_standing_tackle" , "defending_sliding_tackle" ,
             "goalkeeping_diving" , "goalkeeping_handling" , "goalkeeping_kicking" , "goalkeeping_positioning" , "goalkeeping_reflexes" ,
             "player_positions"]].dropna()

pos = ["ST"]
#pos = ["GK" , "CB" , "CM" , "ST"]

for i in pos:

    bb = df.loc[df["player_positions"].str.contains(i)]
    bb["player_position"] = 1
    bb = bb.drop(["player_positions"] , axis = 1)
    cc = df.loc[~df["player_positions"].str.contains(i)]
    cc["player_position"] = 0
    cc = cc.drop(["player_positions"] , axis = 1)
    
    aa = bb.append(cc)
    
    data = aa.drop(["player_position"] , axis = 1)
    target = aa.player_position
    
    X = data
    Y = target
    
    # Logistic regression implementation
    
    # We divide the "train" data in training and testing in order to test the algorithms
    
    X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2)
    
    # We scale the data
    
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    
    # Define the algorithm that we are going to use
    
    algorithm = KNeighborsClassifier(n_neighbors = 5 , metric = "minkowski" , p = 2)
    
    # We train the model
    
    algorithm.fit(X_train , Y_train)
    
    # We realize the prediction
    
    Y_pred = algorithm.predict(X_test)
    
    # We verify the confusion matrix
    
    matrix = confusion_matrix(Y_test , Y_pred)
    
    print("Confusion matrix")
    print(matrix)
    
    # We calculate the precision of the model
    
    precision = precision_score(Y_test , Y_pred)
    
    # We calculate the accuracy of the model
    
    accuracy = accuracy_score(Y_test , Y_pred)
    
    # We calculate the recall of the model
    
    recall = recall_score(Y_test , Y_pred)
    
    # We calculate F1 score
    
    scoref1 = f1_score(Y_test , Y_pred)
    
    # We calculate the ROC - AUC score of the model
    
    roc_auc = roc_auc_score(Y_test , Y_pred)
    print("Model's ROC - AUC score:")
    print(roc_auc)
    
    print("Model's precision for" , i , "is:" , precision)
    print("Model's accuracy for" , i , "is:" , accuracy)
    print("Model's recall for" , i , "is:" , recall)
    print("Model's F1 score for" , i , "is:" , scoref1)
    print("Model's ROC - AUC score for" , i , "is:" , roc_auc)

# Second part

df = df15.append(df16).append(df17).append(df18).append(df19).append(df20).append(df21)

df = df[["attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" ,
             "defending_standing_tackle" , "defending_sliding_tackle" ,
             "goalkeeping_diving" , "goalkeeping_handling" , "goalkeeping_kicking" , "goalkeeping_positioning" , "goalkeeping_reflexes" ,
             "player_positions"]].dropna()

pos = ["ST"]
#pos = ["GK" , "CB" , "CM" , "ST"]

for i in pos:

    bb = df.loc[df["player_positions"].str.contains(i)]
    bb["player_position"] = 1
    bb = bb.drop(["player_positions"] , axis = 1)
    cc = df.loc[~df["player_positions"].str.contains(i)]
    cc["player_position"] = 0
    cc = cc.drop(["player_positions"] , axis = 1)
    
    aa = bb.append(cc)
    
    data = aa.drop(["player_position"] , axis = 1)
    target = aa.player_position
    
    X = data
    Y = target
    
    # Logistic regression implementation
    
    # We divide the "train" data in training and testing in order to test the algorithms
    
    X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2)
    
    # We scale the data
    
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    
    # Define the algorithm that we are going to use
    
    algorithm = KNeighborsClassifier(n_neighbors = 5 , metric = "minkowski" , p = 2)
    
    # We train the model
    
    algorithm.fit(X_train , Y_train)
    
    # We realize the prediction
    
    Y_pred = algorithm.predict(X_test)
    
    # We verify the confusion matrix
    
    matrix = confusion_matrix(Y_test , Y_pred)
    
    print("Confusion matrix")
    print(matrix)
    
    # We calculate the precision of the model
    
    precision = precision_score(Y_test , Y_pred)
    
    # We calculate the accuracy of the model
    
    accuracy = accuracy_score(Y_test , Y_pred)
    
    # We calculate the recall of the model
    
    recall = recall_score(Y_test , Y_pred)
    
    # We calculate F1 score
    
    scoref1 = f1_score(Y_test , Y_pred)
    
    # We calculate the ROC - AUC score of the model
    
    roc_auc = roc_auc_score(Y_test , Y_pred)
    print("Model's ROC - AUC score:")
    print(roc_auc)
    
    print("Model's precision for total" , i , "is:" , precision)
    print("Model's accuracy for total" , i , "is:" , accuracy)
    print("Model's recall for total" , i , "is:" , recall)
    print("Model's F1 score for total" , i , "is:" , scoref1)
    print("Model's ROC - AUC score for total" , i , "is:" , roc_auc)



#%% TRY TO PREDICT IF A PLAYER IS GK AND DO THE SAME FOR CB, CM AND ST (SUPPORT VECTOR MACHINE)

# Maybe it is good to activate warnings back again. They are unactivated because it is easier for visualization
import warnings
warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


df = df15[["attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" ,
             "defending_standing_tackle" , "defending_sliding_tackle" ,
             "goalkeeping_diving" , "goalkeeping_handling" , "goalkeeping_kicking" , "goalkeeping_positioning" , "goalkeeping_reflexes" ,
             "player_positions"]].dropna()

pos = ["ST"]
#pos = ["GK" , "CB" , "CM" , "ST"]

for i in pos:

    bb = df.loc[df["player_positions"].str.contains(i)]
    bb["player_position"] = 1
    bb = bb.drop(["player_positions"] , axis = 1)
    cc = df.loc[~df["player_positions"].str.contains(i)]
    cc["player_position"] = 0
    cc = cc.drop(["player_positions"] , axis = 1)
    
    aa = bb.append(cc)
    
    data = aa.drop(["player_position"] , axis = 1)
    target = aa.player_position
    
    X = data
    Y = target
    
    # Logistic regression implementation
    
    # We divide the "train" data in training and testing in order to test the algorithms
    
    X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2)
    
    # We scale the data
    
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    
    # Define the algorithm that we are going to use
    
    algorithm = SVC(kernel = "linear")
    
    # We train the model
    
    algorithm.fit(X_train , Y_train)
    
    # We realize the prediction
    
    Y_pred = algorithm.predict(X_test)
    
    # We verify the confusion matrix
    
    matrix = confusion_matrix(Y_test , Y_pred)
    
    print("Confusion matrix")
    print(matrix)
    
    # We calculate the precision of the model
    
    precision = precision_score(Y_test , Y_pred)
    
    # We calculate the accuracy of the model
    
    accuracy = accuracy_score(Y_test , Y_pred)
    
    # We calculate the recall of the model
    
    recall = recall_score(Y_test , Y_pred)
    
    # We calculate F1 score
    
    scoref1 = f1_score(Y_test , Y_pred)
    
    # We calculate the ROC - AUC score of the model
    
    roc_auc = roc_auc_score(Y_test , Y_pred)
    print("Model's ROC - AUC score:")
    print(roc_auc)
    
    print("Model's precision for" , i , "is:" , precision)
    print("Model's accuracy for" , i , "is:" , accuracy)
    print("Model's recall for" , i , "is:" , recall)
    print("Model's F1 score for" , i , "is:" , scoref1)
    print("Model's ROC - AUC score for" , i , "is:" , roc_auc)

# Second part

df = df15.append(df16).append(df17).append(df18).append(df19).append(df20).append(df21)

df = df[["attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" ,
             "defending_standing_tackle" , "defending_sliding_tackle" ,
             "goalkeeping_diving" , "goalkeeping_handling" , "goalkeeping_kicking" , "goalkeeping_positioning" , "goalkeeping_reflexes" ,
             "player_positions"]].dropna()

pos = ["ST"]
#pos = ["GK" , "CB" , "CM" , "ST"]

for i in pos:

    bb = df.loc[df["player_positions"].str.contains(i)]
    bb["player_position"] = 1
    bb = bb.drop(["player_positions"] , axis = 1)
    cc = df.loc[~df["player_positions"].str.contains(i)]
    cc["player_position"] = 0
    cc = cc.drop(["player_positions"] , axis = 1)
    
    aa = bb.append(cc)
    
    data = aa.drop(["player_position"] , axis = 1)
    target = aa.player_position
    
    X = data
    Y = target
    
    # Logistic regression implementation
    
    # We divide the "train" data in training and testing in order to test the algorithms
    
    X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2)
    
    # We scale the data
    
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    
    # Define the algorithm that we are going to use
    
    algorithm = SVC(kernel = "linear")
    
    # We train the model
    
    algorithm.fit(X_train , Y_train)
    
    # We realize the prediction
    
    Y_pred = algorithm.predict(X_test)
    
    # We verify the confusion matrix
    
    matrix = confusion_matrix(Y_test , Y_pred)
    
    print("Confusion matrix")
    print(matrix)
    
    # We calculate the precision of the model
    
    precision = precision_score(Y_test , Y_pred)
    
    # We calculate the accuracy of the model
    
    accuracy = accuracy_score(Y_test , Y_pred)
    
    # We calculate the recall of the model
    
    recall = recall_score(Y_test , Y_pred)
    
    # We calculate F1 score
    
    scoref1 = f1_score(Y_test , Y_pred)
    
    # We calculate the ROC - AUC score of the model
    
    roc_auc = roc_auc_score(Y_test , Y_pred)
    print("Model's ROC - AUC score:")
    print(roc_auc)
    
    print("Model's precision for total" , i , "is:" , precision)
    print("Model's accuracy for total" , i , "is:" , accuracy)
    print("Model's recall for total" , i , "is:" , recall)
    print("Model's F1 score for total" , i , "is:" , scoref1)
    print("Model's ROC - AUC score for total" , i , "is:" , roc_auc)

#%% TRY TO PREDICT IF A PLAYER IS GK (NAIVE BAYES)

# Maybe it is good to activate warnings back again. They are unactivated because it is easier for visualization
import warnings
warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


df = df15[["attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" ,
             "defending_standing_tackle" , "defending_sliding_tackle" ,
             "goalkeeping_diving" , "goalkeeping_handling" , "goalkeeping_kicking" , "goalkeeping_positioning" , "goalkeeping_reflexes" ,
             "player_positions"]].dropna()

pos = ["ST"]
#pos = ["GK" , "CB" , "CM" , "ST"]

for i in pos:

    bb = df.loc[df["player_positions"].str.contains(i)]
    bb["player_position"] = 1
    bb = bb.drop(["player_positions"] , axis = 1)
    cc = df.loc[~df["player_positions"].str.contains(i)]
    cc["player_position"] = 0
    cc = cc.drop(["player_positions"] , axis = 1)
    
    aa = bb.append(cc)
    
    data = aa.drop(["player_position"] , axis = 1)
    target = aa.player_position
    
    X = data
    Y = target
    
    # Logistic regression implementation
    
    # We divide the "train" data in training and testing in order to test the algorithms
    
    X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2)
    
    # We scale the data
    
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    
    # Define the algorithm that we are going to use
    
    algorithm = GaussianNB()
    
    # We train the model
    
    algorithm.fit(X_train , Y_train)
    
    # We realize the prediction
    
    Y_pred = algorithm.predict(X_test)
    
    # We verify the confusion matrix
    
    matrix = confusion_matrix(Y_test , Y_pred)
    
    print("Confusion matrix")
    print(matrix)
    
    # We calculate the precision of the model
    
    precision = precision_score(Y_test , Y_pred)
    
    # We calculate the accuracy of the model
    
    accuracy = accuracy_score(Y_test , Y_pred)
    
    # We calculate the recall of the model
    
    recall = recall_score(Y_test , Y_pred)
    
    # We calculate F1 score
    
    scoref1 = f1_score(Y_test , Y_pred)
    
    # We calculate the ROC - AUC score of the model
    
    roc_auc = roc_auc_score(Y_test , Y_pred)
    print("Model's ROC - AUC score:")
    print(roc_auc)
    
    print("Model's precision for" , i , "is:" , precision)
    print("Model's accuracy for" , i , "is:" , accuracy)
    print("Model's recall for" , i , "is:" , recall)
    print("Model's F1 score for" , i , "is:" , scoref1)
    print("Model's ROC - AUC score for" , i , "is:" , roc_auc)

# Second part

df = df15.append(df16).append(df17).append(df18).append(df19).append(df20).append(df21)

df = df[["attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" ,
             "defending_standing_tackle" , "defending_sliding_tackle" ,
             "goalkeeping_diving" , "goalkeeping_handling" , "goalkeeping_kicking" , "goalkeeping_positioning" , "goalkeeping_reflexes" ,
             "player_positions"]].dropna()

pos = ["ST"]
#pos = ["GK" , "CB" , "CM" , "ST"]

for i in pos:

    bb = df.loc[df["player_positions"].str.contains(i)]
    bb["player_position"] = 1
    bb = bb.drop(["player_positions"] , axis = 1)
    cc = df.loc[~df["player_positions"].str.contains(i)]
    cc["player_position"] = 0
    cc = cc.drop(["player_positions"] , axis = 1)
    
    aa = bb.append(cc)
    
    data = aa.drop(["player_position"] , axis = 1)
    target = aa.player_position
    
    X = data
    Y = target
    
    # Logistic regression implementation
    
    # We divide the "train" data in training and testing in order to test the algorithms
    
    X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2)
    
    # We scale the data
    
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    
    # Define the algorithm that we are going to use
    
    algorithm = GaussianNB()
    
    # We train the model
    
    algorithm.fit(X_train , Y_train)
    
    # We realize the prediction
    
    Y_pred = algorithm.predict(X_test)
    
    # We verify the confusion matrix
    
    matrix = confusion_matrix(Y_test , Y_pred)
    
    print("Confusion matrix")
    print(matrix)
    
    # We calculate the precision of the model
    
    precision = precision_score(Y_test , Y_pred)
    
    # We calculate the accuracy of the model
    
    accuracy = accuracy_score(Y_test , Y_pred)
    
    # We calculate the recall of the model
    
    recall = recall_score(Y_test , Y_pred)
    
    # We calculate F1 score
    
    scoref1 = f1_score(Y_test , Y_pred)
    
    # We calculate the ROC - AUC score of the model
    
    roc_auc = roc_auc_score(Y_test , Y_pred)
    print("Model's ROC - AUC score:")
    print(roc_auc)
    
    print("Model's precision for total" , i , "is:" , precision)
    print("Model's accuracy for total" , i , "is:" , accuracy)
    print("Model's recall for total" , i , "is:" , recall)
    print("Model's F1 score for total" , i , "is:" , scoref1)
    print("Model's ROC - AUC score for total" , i , "is:" , roc_auc)

#%% TRY TO PREDICT IF A PLAYER IS GK (DECISSION TREE CLASSIFIERS)

# Maybe it is good to activate warnings back again. They are unactivated because it is easier for visualization
import warnings
warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


df = df15[["attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" ,
             "defending_standing_tackle" , "defending_sliding_tackle" ,
             "goalkeeping_diving" , "goalkeeping_handling" , "goalkeeping_kicking" , "goalkeeping_positioning" , "goalkeeping_reflexes" ,
             "player_positions"]].dropna()

pos = ["ST"]
#pos = ["GK" , "CB" , "CM" , "ST"]

for i in pos:

    bb = df.loc[df["player_positions"].str.contains(i)]
    bb["player_position"] = 1
    bb = bb.drop(["player_positions"] , axis = 1)
    cc = df.loc[~df["player_positions"].str.contains(i)]
    cc["player_position"] = 0
    cc = cc.drop(["player_positions"] , axis = 1)
    
    aa = bb.append(cc)
    
    data = aa.drop(["player_position"] , axis = 1)
    target = aa.player_position
    
    X = data
    Y = target
    
    # Logistic regression implementation
    
    # We divide the "train" data in training and testing in order to test the algorithms
    
    X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2)
    
    # We scale the data
    
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    
    # Define the algorithm that we are going to use
    
    algorithm = DecisionTreeClassifier(criterion = "entropy")
    
    # We train the model
    
    algorithm.fit(X_train , Y_train)
    
    # We realize the prediction
    
    Y_pred = algorithm.predict(X_test)
    
    # We verify the confusion matrix
    
    matrix = confusion_matrix(Y_test , Y_pred)
    
    print("Confusion matrix")
    print(matrix)
    
    # We calculate the precision of the model
    
    precision = precision_score(Y_test , Y_pred)
    
    # We calculate the accuracy of the model
    
    accuracy = accuracy_score(Y_test , Y_pred)
    
    # We calculate the recall of the model
    
    recall = recall_score(Y_test , Y_pred)
    
    # We calculate F1 score
    
    scoref1 = f1_score(Y_test , Y_pred)
    
    # We calculate the ROC - AUC score of the model
    
    roc_auc = roc_auc_score(Y_test , Y_pred)
    print("Model's ROC - AUC score:")
    print(roc_auc)
    
    print("Model's precision for" , i , "is:" , precision)
    print("Model's accuracy for" , i , "is:" , accuracy)
    print("Model's recall for" , i , "is:" , recall)
    print("Model's F1 score for" , i , "is:" , scoref1)
    print("Model's ROC - AUC score for" , i , "is:" , roc_auc)

# Second part

df = df15.append(df16).append(df17).append(df18).append(df19).append(df20).append(df21)

df = df[["attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" ,
             "defending_standing_tackle" , "defending_sliding_tackle" ,
             "goalkeeping_diving" , "goalkeeping_handling" , "goalkeeping_kicking" , "goalkeeping_positioning" , "goalkeeping_reflexes" ,
             "player_positions"]].dropna()

pos = ["ST"]
#pos = ["GK" , "CB" , "CM" , "ST"]

for i in pos:

    bb = df.loc[df["player_positions"].str.contains(i)]
    bb["player_position"] = 1
    bb = bb.drop(["player_positions"] , axis = 1)
    cc = df.loc[~df["player_positions"].str.contains(i)]
    cc["player_position"] = 0
    cc = cc.drop(["player_positions"] , axis = 1)
    
    aa = bb.append(cc)
    
    data = aa.drop(["player_position"] , axis = 1)
    target = aa.player_position
    
    X = data
    Y = target
    
    # Logistic regression implementation
    
    # We divide the "train" data in training and testing in order to test the algorithms
    
    X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2)
    
    # We scale the data
    
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    
    # Define the algorithm that we are going to use
    
    algorithm = DecisionTreeClassifier(criterion = "entropy")
    
    # We train the model
    
    algorithm.fit(X_train , Y_train)
    
    # We realize the prediction
    
    Y_pred = algorithm.predict(X_test)
    
    # We verify the confusion matrix
    
    matrix = confusion_matrix(Y_test , Y_pred)
    
    print("Confusion matrix")
    print(matrix)
    
    # We calculate the precision of the model
    
    precision = precision_score(Y_test , Y_pred)
    
    # We calculate the accuracy of the model
    
    accuracy = accuracy_score(Y_test , Y_pred)
    
    # We calculate the recall of the model
    
    recall = recall_score(Y_test , Y_pred)
    
    # We calculate F1 score
    
    scoref1 = f1_score(Y_test , Y_pred)
    
    # We calculate the ROC - AUC score of the model
    
    roc_auc = roc_auc_score(Y_test , Y_pred)
    print("Model's ROC - AUC score:")
    print(roc_auc)
    
    print("Model's precision for total" , i , "is:" , precision)
    print("Model's accuracy for total" , i , "is:" , accuracy)
    print("Model's recall for total" , i , "is:" , recall)
    print("Model's F1 score for total" , i , "is:" , scoref1)
    print("Model's ROC - AUC score for total" , i , "is:" , roc_auc)

#%% TRY TO PREDICT IF A PLAYER IS GK (RANDOM FOREST CLASSIFIERS)

# Maybe it is good to activate warnings back again. They are unactivated because it is easier for visualization
import warnings
warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


df = df15[["attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" ,
             "defending_standing_tackle" , "defending_sliding_tackle" ,
             "goalkeeping_diving" , "goalkeeping_handling" , "goalkeeping_kicking" , "goalkeeping_positioning" , "goalkeeping_reflexes" ,
             "player_positions"]].dropna()

pos = ["ST"]
#pos = ["GK" , "CB" , "CM" , "ST"]

for i in pos:

    bb = df.loc[df["player_positions"].str.contains(i)]
    bb["player_position"] = 1
    bb = bb.drop(["player_positions"] , axis = 1)
    cc = df.loc[~df["player_positions"].str.contains(i)]
    cc["player_position"] = 0
    cc = cc.drop(["player_positions"] , axis = 1)
    
    aa = bb.append(cc)
    
    data = aa.drop(["player_position"] , axis = 1)
    target = aa.player_position
    
    X = data
    Y = target
    
    # Logistic regression implementation
    
    # We divide the "train" data in training and testing in order to test the algorithms
    
    X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2)
    
    # We scale the data
    
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    
    # Define the algorithm that we are going to use
    
    algorithm = RandomForestClassifier(n_estimators = 10 , criterion = "entropy")
    
    # We train the model
    
    algorithm.fit(X_train , Y_train)
    
    # We realize the prediction
    
    Y_pred = algorithm.predict(X_test)
    
    # We verify the confusion matrix
    
    matrix = confusion_matrix(Y_test , Y_pred)
    
    print("Confusion matrix")
    print(matrix)
    
    # We calculate the precision of the model
    
    precision = precision_score(Y_test , Y_pred)
    
    # We calculate the accuracy of the model
    
    accuracy = accuracy_score(Y_test , Y_pred)
    
    # We calculate the recall of the model
    
    recall = recall_score(Y_test , Y_pred)
    
    # We calculate F1 score
    
    scoref1 = f1_score(Y_test , Y_pred)
    
    # We calculate the ROC - AUC score of the model
    
    roc_auc = roc_auc_score(Y_test , Y_pred)
    print("Model's ROC - AUC score:")
    print(roc_auc)
    
    print("Model's precision for" , i , "is:" , precision)
    print("Model's accuracy for" , i , "is:" , accuracy)
    print("Model's recall for" , i , "is:" , recall)
    print("Model's F1 score for" , i , "is:" , scoref1)
    print("Model's ROC - AUC score for" , i , "is:" , roc_auc)

# Second part

df = df15.append(df16).append(df17).append(df18).append(df19).append(df20).append(df21)

df = df[["attacking_crossing", "attacking_finishing" , "attacking_heading_accuracy" , "attacking_short_passing", "attacking_volleys" ,
             "skill_dribbling", "skill_curve" , "skill_fk_accuracy" , "skill_long_passing" , "skill_ball_control" ,
             "movement_acceleration", "movement_sprint_speed" , "movement_agility" , "movement_reactions" , "movement_balance" ,
             "mentality_aggression" , "mentality_interceptions" , "mentality_positioning" , "mentality_vision" , "mentality_penalties" ,
             "defending_standing_tackle" , "defending_sliding_tackle" ,
             "goalkeeping_diving" , "goalkeeping_handling" , "goalkeeping_kicking" , "goalkeeping_positioning" , "goalkeeping_reflexes" ,
             "player_positions"]].dropna()

pos = ["ST"]
#pos = ["GK" , "CB" , "CM" , "ST"]

for i in pos:

    bb = df.loc[df["player_positions"].str.contains(i)]
    bb["player_position"] = 1
    bb = bb.drop(["player_positions"] , axis = 1)
    cc = df.loc[~df["player_positions"].str.contains(i)]
    cc["player_position"] = 0
    cc = cc.drop(["player_positions"] , axis = 1)
    
    aa = bb.append(cc)
    
    data = aa.drop(["player_position"] , axis = 1)
    target = aa.player_position
    
    X = data
    Y = target
    
    # Logistic regression implementation
    
    # We divide the "train" data in training and testing in order to test the algorithms
    
    X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2)
    
    # We scale the data
    
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    
    # Define the algorithm that we are going to use
    
    algorithm = RandomForestClassifier(n_estimators = 10 , criterion = "entropy")
    
    # We train the model
    
    algorithm.fit(X_train , Y_train)
    
    # We realize the prediction
    
    Y_pred = algorithm.predict(X_test)
    
    # We verify the confusion matrix
    
    matrix = confusion_matrix(Y_test , Y_pred)
    
    print("Confusion matrix")
    print(matrix)
    
    # We calculate the precision of the model
    
    precision = precision_score(Y_test , Y_pred)
    
    # We calculate the accuracy of the model
    
    accuracy = accuracy_score(Y_test , Y_pred)
    
    # We calculate the recall of the model
    
    recall = recall_score(Y_test , Y_pred)
    
    # We calculate F1 score
    
    scoref1 = f1_score(Y_test , Y_pred)
    
    # We calculate the ROC - AUC score of the model
    
    roc_auc = roc_auc_score(Y_test , Y_pred)
    print("Model's ROC - AUC score:")
    print(roc_auc)
    
    print("Model's precision for total" , i , "is:" , precision)
    print("Model's accuracy for total" , i , "is:" , accuracy)
    print("Model's recall for total" , i , "is:" , recall)
    print("Model's F1 score for total" , i , "is:" , scoref1)
    print("Model's ROC - AUC score for total" , i , "is:" , roc_auc)

#%% TESTING








#%%


                        #             ___________
                        #             |         |
                        #             |         |
                        #             |         |
                        #             |         |
                        #             |_________|
                        #      _______|_________|_______
                        
                            #~~~~~~~~~~~$$$$
                            #~~~~~~~~~~$$$$$$
                            #~~~~~~~~~.$$$**$$
                            #~~~~~~~~~$$$'~~`$$
                            #~~~~~~~~$$$'~~~~$$
                            #~~~~~~~~$$$~~~~.$$
                            #~~~~~~~~$$~~~~..$$
                            #~~~~~~~~$$~~~~.$$$
                            #~~~~~~~~$$~~~$$$$
                            #~~~~~~~~~$$$$$$$$
                            #~~~~~~~~~$$$$$$$
                            #~~~~~~~.$$$$$$*
                            #~~~~~~$$$$$$$'
                            #~~~~.$$$$$$$
                            #~~~$$$$$$'`$
                            #~~$$$$$*~~~$$
                            #~$$$$$~~~~~$$.$..
                            #$$$$$~~~~$$$$$$$$$$.
                            #$$$$~~~.$$$$$$$$$$$$$
                            #$$$~~~~$$$*~`$~~$*$$$$
                            #$$$~~~`$$'~~~$$~~~$$$$
                            #3$$~~~~$$~~~~$$~~~~$$$
                            #~$$$~~~$$$~~~`$~~~~$$$
                            #~`*$$~~~~$$$~~$$~~:$$
                            #~~~$$$$~~~~~~~$$~$$'
                            #~~~~~$$*$$$$$$$$$'
                            #~~~~~~~~~~````~$$
                            #~~~~~~~~~~~~~~~`$
                            #~~~~~~~~..~~~~~~$$
                            #~~~~~~$$$$$$~~~~$$
                            #~~~~~$$$$$$$$~~~$$
                            #~~~~~$$$$$$$$~~~$$
                            #~~~~~~$$$$$'~~.$$
                            #~~~~~~~'*$$










