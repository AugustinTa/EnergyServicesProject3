import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import pandas as pd
import plotly.express as px

# Charger les données géographiques des régions françaises
regions_geojson = gpd.read_file('https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions.geojson')

# Créer un DataFrame avec les noms des régions et les valeurs associées
data = {
    'Region': ['Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté', 'Bretagne', 'Centre-Val de Loire',
               'Corse', 'Grand Est', 'Hauts-de-France', 'Île-de-France', 'Normandie', 'Nouvelle-Aquitaine',
               'Occitanie', 'Pays de la Loire', 'Provence-Alpes-Côte d\'Azur'],
    'Valeur': [10, 20, 15, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
}

dico_regions={'Auvergne-Rhône-Alpes':'Auvergne-Rhône-Alpes','Bourgogne-Franche-Comté':'Bourgogne-Franche-Comté','Bretagne':'Bretagne','Centre-Val-de-Loire':'Centre-Val de Loire','Grand-Est':'Grand Est','Hauts-de-France':'Hauts-de-France','Ile-de-France':'Île-de-France','Normandie':'Normandie','Nouvelle-Aquitaine':'Nouvelle-Aquitaine','Occitanie':'Occitanie','PACA':'Provence-Alpes-Côte d\'Azur','Pays-de-la-Loire':'Pays de la Loire'}

df = pd.DataFrame(data)

def plot_map(df):

    # Fusionner les données géographiques avec les données du DataFrame
    merged = regions_geojson.merge(df, how='left', left_on='nom', right_on='Region')

    fig = px.choropleth(merged,
                        geojson=merged.geometry,
                        locations=merged.index,
                        color="Valeur",
                        projection="mercator",
                        title="Carte de la France consommation prochaines 24h en MWh"
                    )

    fig.update_geos(fitbounds="locations", visible=False)
    return fig
    # fig.show()
# fig1=plot_map(df)
# fig1.show()

# import load_models as lm

# data={'Region':['Corse'],'Valeur':[0]}

# regions=['Auvergne-Rhône-Alpes','Bourgogne-Franche-Comté','Bretagne','Centre-Val-de-Loire','Grand-Est','Hauts-de-France','Ile-de-France','Normandie','Nouvelle-Aquitaine','Occitanie','PACA','Pays-de-la-Loire']
# date_test='2022-07-19 00:00:00'

# for region in regions:
#     value=lm.predict_next_hour_cons(region,date=date_test,df_met=lm.df_meteo,df_pow=lm.df_power,power_input={'Consommation-1':lm.df_power[region].loc[date_test, 'Consommation']})
#     # value=1
#     data['Region'].append(dico_regions[region])
#     data['Valeur'].append(value)
# df = pd.DataFrame(data)
# fig1=plot_map(df)
# fig1.show()