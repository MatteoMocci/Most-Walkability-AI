import geopandas as gpd
import pandas as pd
import osmnx as ox
import math
import networkx as nx
import re
import json
import osmapi as osm
import os
import subprocess
from osm_and_street_view_download import get_graph_and_photos
import utm
from shapely.geometry import Point



def protocol(shapefile_path):
    get_pictures = False
    post_process = True
    shapefile = gpd.read_file(shapefile_path)
    shapefile.to_crs("EPSG:4326", inplace=True)
    if get_pictures:
        data_retrieval(shapefile)                                                # Lo shapefile in input viene usato per estrarre le immagini streetview e satellitari
                                                                                 # Il modello prende le immagini e restituisce delle predizioni di camminabilità
    if post_process:
        create_shapefile(shapefile)                                              # I risultati vengono processati per generare uno shapefile di output

def data_retrieval(shapefile):
    print("Getting Streetview images...")
    # get_streetview(shapefile)
    create_csv()
    print("Getting Satellite images...")
    get_sat()


def get_photo_points_and_headings(shapefile, road_network):
    query_points = []
    polygon = shapefile.geometry.unary_union

    for u, v, data in road_network.edges(data=True):
        if data.get('geometry') and data['geometry'].within(polygon):
            lat0, lon0, lat1, lon1 = getLongestEdge(road_network, u, v)
            (x0, y0, zone, north) = utm.from_latlon(lat0, lon0)
            (x1, y1, zone, north) = utm.from_latlon(lat1, lon1)
            p0 = Point(x0, y0, 0)
            p1 = Point(x1, y1, 0)

            mid = utm.to_latlon(p0.x + (p1.x - p0.x) * .5, 
                   p0.y + (p1.y - p0.y) * .5, zone, north)
            
            heading = math.atan2(p1.y - p0.y, p1.x - p0.x)
            heading = heading * (180.0 / math.pi)
            query_points.append((mid,heading))
    return query_points


'''
Restituisce l'edge più lungo compreso tra i nodi u e v
'''
def getLongestEdge(g, u, v):
    geo = nx.get_edge_attributes(g, 'geometry')
    if (u,v) in geo:
      text = re.findall(r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", geo[(u,v)])
      np = len(text)
      maxL = 0
      index = 0
      for i in range(0, np-2, 2):
         lon0 = float(text[i])
         lat0 = float(text[i+1])
         lon1 = float(text[i+2])
         lat1 = float(text[i+3])
         l = (lon0-lon1)*(lon0-lon1) + (lat0-lat1)*(lat0-lat1)
         if l>maxL:
           maxL = l
           index = i
      lon0 = text[index]
      lat0 = text[index + 1]
      lon1 = text[index + 2]
      lat1 = text[index + 3]
    else:
      lat0 = float(g.nodes[u]['y'])
      lon0 = float(g.nodes[u]['x'])
      lat1 = float(g.nodes[v]['y'])
      lon1 = float(g.nodes[v]['x'])

    return float(lat0), float(lon0), float(lat1), float(lon1)

def get_streetview(shapefile):
    folder_path = os.getcwd() + "/road_network.gml"
    (minx, miny, maxx, maxy) = shapefile.total_bounds
    bbox = (miny, maxy, maxx, minx)
    G = get_graph_and_photos(path = folder_path, bbox = bbox)
    return G

def get_sat():
    create_csv()
    return cesium_blueprint('points.csv')

def create_csv():
    osm_lat_lon = [name.split('_') for name in os.listdir('streetview')]
    points_data = []
    api = osm.OsmApi()
    i = 1
    for point in osm_lat_lon:
        lat_p = float(point[1])
        lon_p = float(point[2]) 
        G = ox.graph_from_point((lat_p,lon_p))                      # Genera il grafo relativo al punto
        ne = ox.nearest_edges(G,lon_p,Y=lat_p)          # Ottieni l'arco del grafo più vicino al punto, ossia la strada a cui il punto appartiene
        p0 = api.NodeGet(ne[0])                         # Ottieni i due punti che rappresentano gli estremi della strada. L'informazione ottenuta è il nodo dell'id, quindi per ottenere le informazioni sui punti è usata l'api OSM
        p1 = api.NodeGet(ne[1])
        lat1 = math.radians(p0['lat'])
        lon1 = math.radians(p0['lon'])
        lat2 = math.radians(p1['lat'])
        lon2 = math.radians(p1['lon'])
        delta_lon = lon2 - lon1
        x = math.sin(delta_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon))
        heading = (math.degrees(math.atan2(x, y)) + 360) % 360
        heading = (heading - 90) % 360

        points_data.append(
        {
            'lon': lon_p,
            'lat': lat_p,
            'heading': heading
        })

        print(f"Processing {i}/{len(osm_lat_lon)}")
        i+=1
    pd.DataFrame(points_data).to_csv('points.csv')
    data = pd.DataFrame(points_data).to_dict(orient='index')
    
    # Write the dictionary to a JSON file
    with open('points.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
def find_unreal_editor():
        common_paths = [
            "C:/Program Files/Epic Games/UE_5.0",
            "C:/Program Files/Epic Games/UE_5.1",
            "C:/Program Files/Epic Games/UE_5.2",
            "C:/Program Files/Epic Games/UE_5.3",
            "C:/Program Files/Epic Games/UE_5.4",
            "C:/Program Files/Epic Games/UE_5.5"
        ]

        for base_path in common_paths:
            for root, dirs, files in os.walk(base_path):
                if "UnrealEditor.exe" in files:
                    return os.path.join(root, "UnrealEditor.exe")
        
        raise FileNotFoundError("Unreal Editor executable not found on this system.")

def cesium_blueprint(csv):
    # Find Unreal Editor
    unreal_editor_path = find_unreal_editor()

    # Path to your project
    project_path = os.path.join(os.path.abspath(os.getcwd()),"Cesium/Cesium.uproject")

    exec_cmds = "ce BeginPlay"

    window_mode = "-Windowed"
    resolution = "-ResX=1280 -ResY=720"

    # Run Unreal Editor
    subprocess.run([
        unreal_editor_path,
        project_path,
        "-game",
        f"-ExecCmds={exec_cmds}",
        window_mode,
        resolution
    ])

    rename_photos(csv)

def rename_photos(csv):
    s = "00000"
    df = pd.read_csv(csv)
    for index, row in df.iterrows():
        lat = row['lat'] 
        lon = row['lon']
        lat = round(lat, 6)
        lon = round(lon, 6)
        new_filename = f'Cesium/Saved/Screenshots/WindowsEditor/{index}_{lat}_{lon}.png'
        old_filename = f'Cesium/Saved/Screenshots/WindowsEditor/ScreenShot{s}.png'
        if os.path.exists(old_filename):
            os.rename(old_filename, new_filename)
        s = str(int(s) + 1).zfill(len(s)) # incremento stringa + 1 mantenendo gli zeri

def create_shapefile():
    df = pd.read_csv('comb_predictions.csv')
    df['geometry'] = df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)

    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.set_crs(epsg=4326, inplace=True)
    gdf.to_file('output_shapefile.shp')

    print(f"✅ Shapefile created with {len(df)} points")


if __name__ == '__main__':
    create_shapefile()