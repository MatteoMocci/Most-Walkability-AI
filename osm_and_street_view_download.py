
# from OSMParser import read_osm
import google_streetview.api
import shapefile
import math
import numpy as np
import csv
import time
import re
from PIL import Image
import utm
from vectors import Point, Vector
from random import shuffle
import shutil
import os
import progressbar
import networkx
import googlemaps
import aggdraw
from googleplaces import GooglePlaces
#from pysal.cg import RTree, Rect
from rtree import index
from rtree.index import Rtree
import osmnx as ox
import shapely as sh
import json
import matplotlib.pyplot as plt
import geopandas as gp
from geopy.geocoders import Nominatim
import geopy
import networkx as nx



requests = 0

keys = []
notAvailables = []

'''
Questa funzione salva un'immagine utilizzando l'api di google maps. Inoltre, restituisce True se l'immagine è stata salvata, False se l'immagine è duplicata o ci sono stati errori nella richiesta all'API
'''
def saveImage(index, shp_index, p0, p1, heading, alpha, walk, csvfile):
    # Define parameters for street view api
    params = [{
        'size': '640x640',  # max 640x640 pixels
        'location': '40.7225,8.5517',
        'heading': '20.0',
        'pitch': '0.0',
        'key': '',
        'fov': 360
    }]
    # AIzaSyAz_hLf92j3vhkqm - XAAunjifXiwooni7E
    # AIzaSyCDVKtvxKZaes0 - lpeiEyW19c - co8lOAbY

    if heading > 360:
        heading -= 360

    if heading < 0:
        heading += 360

    x = p0.x + (p1.x - p0.x) * alpha
    y = p0.y + (p1.y - p0.y) * alpha

    (lat, lon) = utm.to_latlon(x, y, 32, 'N')
    # 32 alghero - sassari
    # 29 lisbona

    imageName = "downloads/image"

    params[0]['location'] = str(lat) + ',' + str(lon)
    params[0]['heading'] = str(heading)
    print(params[0]['location'])
    return False

    results = google_streetview.api.results(params)
    if results.metadata[0]['status'] != "OK":
        return False

    results.download_links(imageName)

    if results.metadata[0]['copyright'] != '© Google, Inc.':
        return False

    realLat = results.metadata[0]['location']['lat']
    realLon = results.metadata[0]['location']['lng']
    (realX, realY, zone, north) = utm.from_latlon(realLat, realLon)
    rp = Point(realX, realY, 0)
    a = Vector.from_points(p0, p1)
    l = a.magnitude()
    a = a.multiply(1.0 / l)
    b = Vector.from_points(p0, rp)
    xl = b.dot(a)
    if xl > l - 1.0 or xl < 1.0:
        shutil.rmtree("downloads/image")
        return False

    imageName += "/gsv_0.jpg"

    if (not os.path.exists(imageName)):
        shutil.rmtree("downloads/image")
        print("Unable to download image")
        return False

    image = Image.open(imageName)

    width = image.size[0]
    height = image.size[1]

    new_image = Image.new('RGB', (width, height))
    x_offset = 0
    new_image.paste(image, (x_offset, 0))

    key0 = str(shp_index) + "_" + str(int(lat * 10000000) / 10000000.0) + "_" + str(
        int(lon * 10000000) / 10000000.0) + "_" + str(int(heading * 100) / 100.0)

    if key0 in keys:
        print("skipped duplicate")
        shutil.rmtree("downloads/image")
        return False

    keys.append(key0)

    key = key0 + "_" + str(index)

    new_image.save("dataset/" + key + ".jpg", "JPEG")
    csvfile = open('dataset/data.csv', 'a')
    csvfile.write(str(shp_index) + ';' + key + ';' + str(walk) + '\n')
    shutil.rmtree("downloads/image")
    return True

'''
Data una lista di archi, questa funzione effettua delle chiamate alla funzione saveImage
'''
def processList(listOfArcs, alphas, index, maxSize, processed):
    csvfile = open('dataset\data.csv', 'a')

    # go=False
    i = 0
    with progressbar.ProgressBar(max_value=len(listOfArcs)) as bar:
        for (shp_index, shape, walk) in listOfArcs:

            # if shp_index==3364:
            # go=True

            # if not go:
            # continue
            bar.update(i)
            i = i + 1
            np = len(shape.points)
            for i0 in range(0, np - 1):
                i1 = i0 + 1
                p0 = Point(shape.points[i0][0], shape.points[i0][1], 0)
                p1 = Point(shape.points[i1][0], shape.points[i1][1], 0)

                heading = math.atan2(p1.y - p0.y, p1.x - p0.x)
                heading = 90 - heading * (180.0 / math.pi)

                v = Vector.from_points(p0, p1)

                for alpha in alphas:
                    if not saveImage(index, shp_index, p0, p1, heading + 45, alpha, walk, csvfile):
                        index += 1
                        processed += 1
                    if not saveImage(index, shp_index, p0, p1, heading + 180 + 45, alpha, walk, csvfile):
                        index += 1
                        processed += 1

                print('processed=' + str(processed))

                i0 = i1
                i1 = i1 + 1

                if maxSize > 0 and processed > maxSize:
                    return index

    return index


def getCoordinates(g, nodes):
    """
    Extract (lon, lat) coordinate pairs from nodes in an osmgraph
    Parameters
    ----------
    g : networkx graph
    nodes : iterable of node ids

    Returns
    -------
    List of (lon, lat) coordinate pairs

    """
    c = [(g.node[n]['lon'], g.node[n]['lat']) for n in nodes]
    return c

'''
Questa funzione ricava una lista d'archi dallo shapefile il cui path è passato come variabile, questi archi sono poi passati alla funzione processList che prende singolarmente le foto
'''
def downloadPhotosFromShapefile(file):
    sf = shapefile.Reader(file)
    alphas = [0.5]
    shapes = sf.shapes()
    shp_index = 0
    data = []
    for shape in shapes:
        data.append((shp_index, shape, 0.0))
        shp_index += 1
    processList(data, alphas, 0, -1, 0)

'''
Questa funzione salva l'immagine di un arco con field-of-view pari a 90°. Questa funzione non è richiamata nel blocco di nessun'altra funzione. Al suo posto viene utilizzata la funzione saveImageOfEdge360 che combina delle immagini
con fov 90° per ottenere la visione a 360°
'''
def saveImageOfEdge(osmId, osmNode0, osmNode1, p0, p1, heading, alpha, walk, folder, ZN, ZL):

    global requests
    ZN =23
    ZL = 'K'
    # print('Trying to download image for edge ' + osmId)
    # Define parameters for street view api
    params = [{
        'size': '640x640',  # max 640x640 pixels
        'location': '40.7225,8.5517',
        'heading': '20.0',
        'pitch': '0.0',
        'fov': 90,
        'key': ''
    }]

    if heading > 360:
        heading -= 360

    if heading < 0:
        heading += 360

    x = p0.x + (p1.x - p0.x) * alpha
    y = p0.y + (p1.y - p0.y) * alpha

    (lat, lon) = utm.to_latlon(x, y, ZN, ZL)

    k = str(lat) + str(lon) + str(heading)
    if k in notAvailables:
        # print("skipped not available")
        return False

    imageName = "downloads/image"

    params[0]['location'] = str(lat) + ',' + str(lon)
    params[0]['heading'] = str(heading)

    results = google_streetview.api.results(params)

    requests = requests + 1

    if results.metadata[0]['status'] != "OK":
        return False

    realLat = results.metadata[0]['location']['lat']
    realLon = results.metadata[0]['location']['lng']

    filename = osmId + "_" + str(int(realLat * 10000000) / 10000000.0) + "_" + str(
        int(realLon * 10000000) / 10000000.0) + "_" + str(int(heading * 100) / 100.0)
    filename = filename.replace(", ", "EE")

    if filename in keys:
        # print("skipped duplicate pre-download")
        return False

    if results.metadata[0]['copyright'] != '© Google, Inc.':
        csvfile = open(folder + '/' + 'notavailable.csv', 'a')
        csvfile.write(str(lat) + ';' + str(lon) + ';' + str(heading) + '\n')
        csvfile.close()
        return False

    results.download_links(imageName)

    imageName += "/gsv_0.jpg"

    if (not os.path.exists(imageName)):
        shutil.rmtree("downloads/image")
        print("Unable to download image")
        return False

    image = Image.open(imageName)

    width = image.size[0]
    height = image.size[1]

    new_image = Image.new('RGB', (width, height))
    x_offset = 0
    new_image.paste(image, (x_offset, 0))

    filename = osmId + "_" + str(int(realLat * 1000000) / 1000000.0) + "_" + str(
        int(realLon * 1000000) / 1000000.0) + "_" + str(int(heading * 100) / 100.0)
    filename = filename.replace(", ", "EE")

    if filename in keys:
        print("skipped duplicate post-download", filename)
        shutil.rmtree("downloads/image")
        return False

    keys.append(filename)

    new_image.save(folder + '/' + filename + ".jpg", "JPEG")
    csvfile = open(folder + '/' + 'data.csv', 'a')
    osmId = osmId.replace(", ", "EE")
    csvfile.write(str(osmId) + ';' + str(osmNode0) + ';' + str(osmNode1) + ';' + filename + ';' + str(walk) + '\n')
    csvfile.close()
    shutil.rmtree("downloads/image")
    return True

'''
Questa funzione salva l'immagine di un arco con field of view pari a 360° (combinando 4 immagini da 90).
Importante: cambiare il valore di key (è invalido) 
'''
def saveImageOfEdge360(osmId, osmNode0, osmNode1, p0, p1, streetHeading, alpha, walk, folder, ZN, ZL, footprints, rTreeFootprints, simulate):
    from shapely.geometry.point import Point

    global requests

    key = open('key.txt').read()

    # print('Trying to download image for edge ' + osmId)
    # Define parameters for street view api
    params = [{
        'size': '640x640',  # max 640x640 pixels
        'location': '40.7225,8.5517',
        'heading': '20.0',
        'pitch': '0.0',
        'fov': 120,
        'key': key
    }]

    if os.path.exists("downloads/image"):
      shutil.rmtree("downloads/image")

    x = p0.x + (p1.x - p0.x) * alpha
    y = p0.y + (p1.y - p0.y) * alpha

    #(x, y, zone, north) = utm.from_latlon(40.84177, 9.40521)


    buildings = list(rTreeFootprints.intersection((x - 50, y - 50, x + 50, y + 50), objects=True))
    
    '''
    if len(buildings) < 3:
        print(f"Not urbanized: {len(buildings)}")
        return 0

    area = 0
    for building in buildings:
        shape = building.object
        area = area + shape.area
    if area < 250:
        print(f"Not urbanized area: {area}")
        return 0
    '''
    #print("Urbanized")

    (lat, lon) = utm.to_latlon(x, y, ZN, ZL)

    k = str(lat) + str(lon)
    if k in notAvailables:
        #print("skipped not available")
        return 0

    filename = "";
    images = []
    for i in range(0,4): #numero chiamate api
        heading = streetHeading
        if i % 2 == 0:
            heading -= 45
        else:
            heading += 45
        
        if i > 1:
            heading += 180

        if heading > 360:
            heading -= 360
        if heading < 0:
            heading += 360

        params[0]['location'] = str(lat) + ',' + str(lon)
        params[0]['heading'] = str(heading)

        if not simulate:
            results = google_streetview.api.results(params)
            if results.metadata[0]['status'] != "OK":
                return 0

            realLat = results.metadata[0]['location']['lat']
            realLon = results.metadata[0]['location']['lng']
        else:
            realLat = lat
            realLon = lon

        filename = osmId + "_" + str(int(realLat * 10000000) / 10000000.0) + "_" + str(int(realLon * 10000000) / 10000000.0)
        filename = filename.replace(", ", "EE")

        if filename in keys:
            print("skipped duplicate pre-download")
            return -1

        if not simulate and results.metadata[0]['copyright'] != '© Google, Inc.' and results.metadata[0]['copyright'] != '© Google':
            csvfile = open(folder + '/' + 'notavailable.csv', 'a')
            csvfile.write(str(lat) + ';' + str(lon) + '\n')
            csvfile.close()
            return 0

        requests = requests + 1
        if not simulate:
          results.download_links("downloads/image")
          if not os.path.exists("downloads/image/gsv_0.jpg"):
             shutil.rmtree("downloads/image")
             print("Unable to download image")
             return 0
          os.rename("downloads/image/gsv_0.jpg", "downloads/image/img_" + str(i) + ".jpg");


          imageName = "downloads/image/img_" + str(i) + ".jpg"
          Image.open(imageName).save(folder + '/' + filename + '_' + str(i) + ".jpg", "JPEG")
    '''
    if not simulate:
        for i in range(0, 4):
          widths, heights = zip(*(i.size for i in images))
          total_width = sum(widths)
          max_height = max(heights)

        new_image = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
          new_image.paste(im, (x_offset, 0))
          x_offset += im.size[0]
        new_image.save(folder + '/' + filename + ".jpg", "JPEG")

    keys.append(filename)
    '''

    csvfile = open(folder + '/' + 'data.csv', 'a')
    osmId = osmId.replace(", ", "EE")
    csvfile.write(str(osmId) + ';' + str(osmNode0) + ';' + str(osmNode1) + ';' + filename + ';' + str(walk) + '\n')
    csvfile.close()
    if not simulate:
      shutil.rmtree("downloads/image")
      time.sleep(0.1)
    return 1




def saveImage360fromLatLon(id, osmId, lat, lon, streetHeading, walkScore, folder):

    global requests

    # print('Trying to download image for edge ' + osmId)
    # Define parameters for street view api
    params = [{
        'size': '640x640',  # max 640x640 pixels
        'location': '40.7225,8.5517',
        'heading': '20.0',
        'pitch': '0.0',
        'fov': 90,
        'key': 'xxx'
    }]

    if os.path.exists("downloads/image"):
      shutil.rmtree("downloads/image")

    filename = "";
    heading = streetHeading - 45
    for i in range(0,4):

        if heading > 360:
            heading -= 360
        if heading < 0:
            heading += 360

        k = str(lat) + str(lon) + str(heading)
        if k in notAvailables:
            print("skipped not available")
            return False

        params[0]['location'] = str(lat) + ',' + str(lon)
        params[0]['heading'] = str(heading)

        results = google_streetview.api.results(params)


        if results.metadata[0]['status'] != "OK":
            return False

        realLat = results.metadata[0]['location']['lat']
        realLon = results.metadata[0]['location']['lng']

        filename = str(osmId) + "_" + str(int(realLat * 10000000) / 10000000.0) + "_" + str(int(realLon * 10000000) / 10000000.0)
        filename = filename.replace(", ", "EE")

        if filename in keys:
            print("skipped duplicate pre-download")
            return False

        if results.metadata[0]['copyright'] != '© Google, Inc.' and results.metadata[0]['copyright'] != '© Google':
            csvfile = open(folder + '/' + 'notavailable.csv', 'a')
            csvfile.write(str(lat) + ';' + str(lon) + ';' + str(heading) + '\n')
            csvfile.close()
            return False

        results.download_links("downloads/image")

        requests = requests + 1

        if not os.path.exists("downloads/image/gsv_0.jpg"):
            shutil.rmtree("downloads/image")
            print("Unable to download image")
            return False

        os.rename("downloads/image/gsv_0.jpg", "downloads/image/img_" + str(i) + ".jpg");

        heading += 90


    images = []
    for i in range(0, 4):
      imageName = "downloads/image/img_" + str(i) + ".jpg"
      images.append(Image.open(imageName))

    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_image = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
      new_image.paste(im, (x_offset, 0))
      x_offset += im.size[0]
    new_image.save(folder + '/' + filename + ".jpg", "JPEG")


    if filename in keys:
        print("skipped duplicate post-download", filename)
        shutil.rmtree("downloads/image")
        return False

    keys.append(filename)

    csvfile = open(folder + '/' + 'data360.csv', 'a')
    #csvfile.write(str(osmId) + ';' + str(0) + ';' + str(0) + ';' + filename + ';' + str(walkScore) + '\n')
    csvfile.write(str(id) + ',' + filename + ',' + "CA_SS_AHO" + ',' + str(walkScore) + ", 0" + '\n')
    csvfile.close()
    shutil.rmtree("downloads/image")
    time.sleep(0.1)
    return True





'''
Questa funzione prende un dataset .csv con path "csvdatafile" e nella cartella con path "outputFolder" salva le immagini con fov 360° relative ai punti indicati
'''
def convertTo360(csvdatafile, outputFolder):
    import time
    from PIL import Image
    import functools

   #scan folder and creates a single image from th four images of a point
   #the class of the resulting image is the maximum class of the composing images
    dataset = {}
    with open(csvdatafile) as csvfile:
       reader = csv.reader(csvfile, delimiter=',')
       count = 0
       for row in reader:

           filename = row[1]
           cl = int(row[7])
           xy = filename.split('_')
           lat = float(xy[1])
           lon = float(xy[2])

           photoNum = int(xy[4])

           id = str(xy[0]) + '_' + str(lat) + '_' + str(lon)

           count = count + 1
           print( str(count)  + ' ->  ' + id )

           osmId = xy[0]
           c = 0
           nf = 0
           if id in dataset:
               nf = dataset[id][0]
               c = dataset[id][2]
               head = dataset[id][3]
           else:
               dataset[id] = (0, 0, ' ', ' ', ' ', ' ')

           if nf==0:
             head = str(float(xy[3]) - 45.0)
             print('Heading= ' + head)

           c = max(float(cl), float(c))
           nf = nf + 1
           dataset[id] = (nf, osmId, c, head, lat, lon)

    incomplete = 0
    total = 0
    for key, element in dataset.items():
       if element[0]>0:
         print('Heading= ' + element[3])
         streetHeading = float(element[3])
         walkScore = int(element[2])
         osmId = float(element[1])
         lat = float(element[4])
         lon = float(element[5])
         total += 1
         #scarica le quattro immagini e crea l'immagine unica
         print('Saving ' + str(total))
         saveImage360fromLatLon(total, osmId, lat, lon, streetHeading, walkScore, outputFolder)
       else:
         incomplete += 1

    print('Created ' + str(total) + ' records\n')
    print('Found ' + str(incomplete) + ' points with less than 4 photos\n')


'''
Questa funzione converte le 4 immagini di un punto in una sola immagine
'''
def convert360To180(csvdatafile, inputFolder, outputFolder):
    import time
    from PIL import Image
    import functools

    # scan folder and creates a single image from th four images of a point
    # the class of the resulting image is the maximum class of the composing images
    dataset = {}
    with open(csvdatafile) as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        id = 0
        for row in reader:
            filename = row[3]
            setname = row[0]
            walkScore = 0
            dummy = int(row[4])

            image360 = Image.open(inputFolder + '/' + filename + '.jpg')
            width, height = image360.size

            csvfileOut = open(outputFolder + '/' + 'data180.csv', 'a')

            box = (0, 0, width/2, height)
            new_image1 = image360.crop(box)
            new_image1.save(outputFolder + "/" + filename + "_1.jpg", "JPEG")
            csvfileOut.write(str(id) + ',' + filename+'_1' + ',' + setname + ',' + str(walkScore) + ", 0" + '\n')
            id = id + 1

            box = (width / 2, 0, width, height)
            new_image2 = image360.crop(box)
            new_image2.save(outputFolder + "/" + filename + "_2.jpg", "JPEG")
            csvfileOut.write(str(id) + ',' + filename+'_2' + ',' + setname + ',' + str(walkScore) + ", 0" + '\n')
            csvfileOut.close()
            id = id + 1


'''
Questa funzione legge un file osm e lo trasforma in grafo
'''
def createGraphFromOSM(file):
    # create graph
    g = read_osm(file)
    basefile = os.path.splitext(file)[0]
    networkx.write_gml(g, basefile + '.gml')


'''
Questa funzione crea un grafo da un file gml
'''
def createGraphFromGraphML(file, outfile):
    # create graph
    g = networkx.read_graphml(file)
    networkx.set_edge_attributes(g, name='imageFlag', values=0)
    networkx.set_edge_attributes(g, name='walkability', values=-1)
    basefile = os.path.splitext(file)[0]
    networkx.write_graphml(g, outfile)

'''
Definisce la distanza tra 3 punti
'''
def distps(v, w, p):
    import numpy as np
    p1 = np.asarray([v.x, v.y])
    p2 = np.asarray([w.x, w.y])
    p3 = np.asarray([p.x, p.y])
    d = np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)
    return d


'''
Restituisce l'edge che si trova in mezzo rispetto a quelli specificati
'''
def getCentralEdge(g, u, v):
    geo = networkx.get_edge_attributes(g, 'geometry')
    if (u,v) in geo:
      text = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", geo[(u,v)])
      np = len(text)/2
      i = int(np/2)-1
      lon0 = text[2*i]
      lat0 = text[2*i+1]
      lon1 = text[2*i+2]
      lat1 = text[2*i+3]
    else:
      lat0 = float(g.nodes[u]['y'])
      lon0 = float(g.nodes[u]['x'])
      lat1 = float(g.nodes[v]['y'])
      lon1 = float(g.nodes[v]['x'])

    return float(lat0), float(lon0), float(lat1), float(lon1)


'''
Restituisce l'edge più lungo compreso tra i nodi u e v
'''
def getLongestEdge(g, u, v):
    geo = networkx.get_edge_attributes(g, 'geometry')
    if (u,v) in geo:
      text = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", geo[(u,v)])
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


'''
Crea un rtree partendo da un grafo?
'''
def createRTree(g):
    #t = RTree()
    p = index.Property()
    idx = index.Index(properties=p)

    i = 0

    for u, v in g.edges():
        lat0 = float(g.nodes[u]['y'])
        lon0 = float(g.nodes[u]['x'])
        lat1 = float(g.nodes[v]['y'])
        lon1 = float(g.nodes[v]['x'])

        (x0, y0, ZN, ZL) = utm.from_latlon(lat0, lon0)
        (x1, y1, ZN, ZL) = utm.from_latlon(lat1, lon1)
        #print(x0, y0, x1, y1)
        #r = Rect(x0-5, y0-5, x1+5, y1+5)
        v = (u, v)
        # print(obj, r.coords())
        #t.insert(obj, r)
        idx.insert(i, (min(x0-50,x1+50), min(y0-50, y1+50), max(x0-50, x1+50), max(y0-50, y1+50)), obj=v)
        i = i + 1

    return idx

'''
Crea la footprint di un rTree partendo da un grafo?
'''
def createFootprintRTree(gdf):
    #t = RTree()
    p = index.Property()
    idx = index.Index(properties=p)

    i = 0

    for shape in gdf['geometry']:
        x0, y0, x1, y1 = shape.bounds
        eps = 2
        idx.insert(i, (min(x0-eps,x1+eps), min(y0-eps, y1+eps), max(x0-eps, x1+eps), max(y0-eps, y1+eps)), obj=shape)
        i = i + 1
    return idx


'''
Restituisce un edge nel grafo che corrisponde alla longitudine e latitudine specificata
'''
def findEdge(g, idx, lon, lat):
    (xp, yp, ZN, ZL) = utm.from_latlon(lat, lon)

    minDist = 1.0E10
    uMin = 0
    vMin = 0
    toll = 10.0
    i = 0

    #res = [r.leaf_obj() for r in rtree.query_point((xp, yp)) if r.is_leaf()]
    #res = rtree.intersection((xp-1, yp-1, xp+1, yp +1))
    res = list(idx.intersection((xp-50, yp-50, xp+50, yp +50), objects=True))

    if len(res) == 0:
        print('No results from rTree query ' + str(lat) + ' ' + str(lon))
        exit()
        #for u, v, d in g.edges(data=True):
        #    res.append(object)

    for edge in res:
        u = edge.object[0]
        v = edge.object[1]

        lat0, lon0, lat1, lon1 = getLongestEdge(g, u, v)

        (x0, y0, zone, north) = utm.from_latlon(lat0, lon0)
        (x1, y1, zone, north) = utm.from_latlon(lat1, lon1)
        p0 = Point(x0, y0, 0)
        p1 = Point(x1, y1, 0)
        p = Point(xp, yp, 0)
        d2 = distps(p0, p1, p)
        if d2 < minDist:
            minDist = d2
            uMin = u
            vMin = v
    #print(xp, yp, uMin, vMin, " -> Min dist=", minDist)
    return uMin, vMin, minDist

'''
Questa funzione trasforma un'angolo in modo che abbia un'intervallo tra 0 e 360°, aggiunendo o sottraendo 360 se l'angolo è più piccolo o più grande di 360°
'''
def normalizeAngle(alpha):
    if alpha > 360:
        alpha -= 360
    if alpha < 0:
        alpha += 360
    return alpha


'''
Un altra funzione che trova un arco nel grafo, utilizzando l'osmid e i nodi stessi al posto della longitudine e latitudine
'''
def findEdgeByOSMIdUAndV(g, OsmId, un, vn):
  ids = networkx.get_edge_attributes(g, 'osmid')
  for (u,v) in ids:
      sameNodes = ((u == un) and (v == vn)) or ((u == vn) and (v == un))
      cosmId = ids[(u,v)].replace(", ", "EE")
      if OsmId == cosmId and sameNodes:
          return (u,v)

  return (None, None)


'''
Questa funzione scarica le foto dal grafo ML
'''
def downloadPhotosFromGraphML(file, footPrints, Simulation, maxRequests):

    global requests

    folder, tail = os.path.split(file)

    for f in os.listdir(folder):
        if f.endswith(".jpg"):
            f = os.path.splitext(f)[0]
            keys.append(f)

    # load graph
    mgd = networkx.read_graphml(file)

    gud = mgd.to_undirected()
    g = networkx.Graph(gud)

    print('Creating RTree...')
    fpRtree = createFootprintRTree(footPrints)
    rtree = createRTree(g)
    print('done')

    networkx.set_edge_attributes(g, name='imageFlag', values=0)

    print('The graph is composed of ' + str(len(g.edges)) + ' edges')

    csv_dataset = folder + '/notavailable.csv'
    if os.path.isfile(csv_dataset):
        with open(csv_dataset) as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                lat = row[0]
                lon = row[1]
                key = str(lat) + str(lon)
                notAvailables.append(key)

    # reconstruction of already downloaded photos

    csv_dataset = folder + '/data.csv'
    nPhotos = 0
    nAttributed = 0
    if os.path.isfile(csv_dataset):
        with open(csv_dataset) as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                nPhotos = nPhotos + 1
                print(nPhotos)
                OsmId = row[0]
                u = row[1]
                v = row[2]
                filename = row[3]
                keys.append(filename)

                '''
                xy = filename.split('_')
                lat = float(xy[1])
                lon = float(xy[2])
                #u, v, d = findEdge(g, rtree, lon, lat)

                u, v = findEdgeByOSMIdUAndV(g, OsmId, u, v)

                if u is None and v is None:
                    print("arc not found")
                    exit()
                if not u in g.nodes:
                    print("node" + u + " not found")
                    exit()
                if not v in g.nodes:
                    print("node" + v + " not found")
                    exit()
                if not g.has_edge(u, v):
                    print("arc (" + u +"," + v + ") not found")
                    exit()
                '''

                d = g.get_edge_data(u, v)
                if d is not None:
                   d['imageFlag'] = 1
                   nAttributed = nAttributed + 1
                else:
                   print("arc (" + u + "," + v + ") not found")
                   exit()



    print('number of already downloaded photos=', nPhotos, "(", nAttributed, " attributed to arcs)")

    i = 0
    with progressbar.ProgressBar(max_value=len(g.edges)) as bar:
        for u, v, data in g.edges(data=True):
            bar.update(i)
            i = i + 1

            if requests>=maxRequests:
                print("Reached maximum number of downloads")
                exit()

            uId = u
            vId = v
            edgeId = data['osmid']

            lat0, lon0, lat1, lon1 = getLongestEdge(g, u, v)
            #print(edgeId  + " " + str(lat0) + " " + str(lon0) + " " + str(lat1) + " " + str(lon1))

            (x0, y0, zone, north) = utm.from_latlon(lat0, lon0)
            (x1, y1, zone, north) = utm.from_latlon(lat1, lon1)

            p0 = Point(x0, y0, 0)
            p1 = Point(x1, y1, 0)
            dist = math.dist((x0,y0),(x1,y1))
            a = Vector.from_points(p0, p1)
            # l = a.magnitude()


            heading = math.atan2(p1.y - p0.y, p1.x - p0.x)
            heading = heading * (180.0 / math.pi)
            if data['imageFlag'] == 0: # se l'immagine non è stata scaricata
                #print('No images for osmid=', edgeId)
                if dist < 100:
                    alpha = 0.5
                    res = saveImageOfEdge360(edgeId, uId, vId, p0, p1, heading, alpha, 0, folder, zone, north, footPrints, fpRtree, Simulation)     #   scarica l'immagine dell'edge
                else:
                    alpha = 0.3
                    res = saveImageOfEdge360(edgeId, uId, vId, p0, p1, heading, alpha, 0, folder, zone, north, footPrints, fpRtree, Simulation)
                    alpha = 0.7
                    res = saveImageOfEdge360(edgeId, uId, vId, p0, p1, heading, alpha, 0, folder, zone, north, footPrints, fpRtree, Simulation)
                    
                if res==1:
                    data['imageFlag'] = 1   # l'immagine è stata scaricata, segnalo
                    #print('--- Saved new image for osmid=', edgeId)
                elif res !=-1:              # se l'immagine non è stata trovata, prova con alpha < 0.5
                    alpha = 0.3
                    res=saveImageOfEdge360(edgeId, uId, vId, p0, p1, heading, alpha, 0, folder, zone, north, footPrints, fpRtree, Simulation)
                    if res == 1:           #segna se l'immagine è stata scaricata
                        data['imageFlag'] = 1
                        #print('--- Saved new image for osmid=', edgeId)
                    elif res != -1:        # se neanche con alpha < 0.5 è stata scaricata, prova con alpha > 0.5
                        alpha = 0.7
                        if saveImageOfEdge360(edgeId, uId, vId, p0, p1, heading, alpha, 0, folder, zone, north, footPrints, fpRtree, Simulation)==1:
                            data['imageFlag'] = 1 # segna se scaricata
                            #print('--- Saved new image for osmid=', edgeId)
                        else:               # se neanche con alpha < 0.5 è stata scaricata, prova con alpha > 0.5
                            print("Unable to download or not urbanized: " + str(i)) 
                    else:
                        print("Already downloaded " + str(i))
                else:
                    print("Already downloaded " + str(i))
            else:
                print("Already downloaded " + str(i))
    return g





'''
Questa funzione scarica le foto da un shapefile, la differenza rispetto all'altra funzione è che permette di stabilire un valore w (walkability?) che condiziona le aree che vengono selezionate per fare le foto. 
Questa w è legata al valore associato ad ogni arco, che potrebbe essere un valore di walkability assegnato prendendo uno score dal dataset
Chiedere a Trunfio il significato della W
'''
def downloadPhotosFromShapefileAndAttributeW(file):
    sf = shapefile.Reader(file)
    '''
    fields = sf.fields
    for field in fields:
        print(field)
    '''
    records = sf.records()
    min_w = 1.0E10
    max_w = 0
    nr = 0
    for record in records:
        w = 1.0 - record[0]
        min_w = min(min_w, w)
        max_w = max(max_w, w)
        nr += 1
    print("min=" + str(min_w))
    print("max=" + str(max_w))

    walks = []
    for record in records:
        w = 1.0 - record[0]
        w -= min_w
        w /= (max_w - min_w)
        w = round(w, 1)
        walks.append(w)

    shapes = sf.shapes()
    records = sf.records()

    # 0-0.2  0.2-0.4  0.4-0.6  0.6-0.8  0.8-1.0
    low = []
    average = []
    good = []
    high = []

    shp_index = 0
    for shape, walk in zip(shapes, walks):
        if (walk > 0.0 and walk <= 0.25):
            low.append((shp_index, shape, walk))
        elif (walk > 0.25 and walk <= 0.45):
            print(walk)
            average.append((shp_index, shape, walk))
        elif (walk > 0.45 and walk <= 0.8):
            good.append((shp_index, shape, walk))
        elif (walk > 0.8 and walk <= 1.0):
            high.append((shp_index, shape, walk))

        shp_index += 1

    print(len(low) * 4)
    print(len(average) * 4)
    print(len(good) * 4)
    print(len(high) * 4 * 5)

    shuffle(low)
    shuffle(average)
    shuffle(good)
    shuffle(high)

    index = 2827
    # alphas=[0.1, 0.25, 0.45, 0.6, 0.7, 0.85, 0.95]
    alphas = [0.2, 0.45, 0.6, 0.7, 0.80]
    print('high')
    # index = processList(high, alphas, index, -1, 0)
    alphas = [0.5]
    # print('good')
    # index = processList(good, alphas, index, 2000, 0)
    print('average')
    index = processList(average, alphas, index, 2000, 0)
    print('low')
    index = processList(low, alphas, index, 2000, 0)

    exit()

    alphas = [0.5]
    index = 0
    # index = processList(low, alphas, index, -1, 0)
    index = processList(average, alphas, index, -1, 0)
    index = processList(good, alphas, index, -1, 0)
    alphas = [0.1, 0.25, 0.45, 0.6, 0.7, 0.85, 0.95]
    index = processList(high, alphas, index, -1, 0)

'''
Distanza tra 3 punti
'''
def distance(p0, p1, p2):  # p3 is the point
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    nom = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denom = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
    result = nom / denom
    return result

'''
Questa funzione utilizza il dataset in formato csv per assegnare agli archi del grafo un punteggio di walkability
'''
def attributeScoreToArcs(shfilein, shfileout, csvdatafile):
    # read the deep network output

    dataset = {}
    pdataset = {}
    with open(csvdatafile) as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            osmId = row[0]
            n1 = row[1]
            n2 = row[2]
            id = osmId + '_' + n1 + '_' + n2
            filename = row[3]
            cl = row[4]
            xy = filename.split('_')
            lat = float(xy[1])
            lon = float(xy[2])
            (x, y, ZN, ZL) = utm.from_latlon(lat, lon)

            v0 = v1 = v2 = v3 = v4 = 0

            if id in dataset:
                v0 = int(dataset[id][2])
                v1 = int(dataset[id][3])
                v2 = int(dataset[id][4])
                v3 = int(dataset[id][5])
                v4 = int(dataset[id][6])

            if   cl == '0': v0 += 1
            elif cl == '1': v1 += 1
            elif cl == '2': v2 += 1
            elif cl == '3': v3 += 1
            elif cl == '4': v4 += 1
            dataset[id] = (x, y, v0, v1, v2, v3, v4, filename)
            pdataset[id] = (x, y, cl, filename)

 ##   sf = shapefile.Reader(shfilein)
  #  fields = sf.fields
    # for field in fields:
    #    print(field)
 #   shapes = sf.shapes()
#    records = sf.records()

 #   print('There are ' + str(len(shapes)) + ' shapes')

    w = shapefile.Writer()
    # w.field('averageClass', 'N', decimal=2)
    # w.field('maxClass', 'N')
    # w.field('nPhotos', 'N')

    w.field('Class', 'N')
    w.field('PhotoId', 'C')
    for key, element in dataset.items():
        w.point(element[0], element[1])
        v0 = int(element[2])
        v1 = int(element[3])
        v2 = int(element[4])
        v3 = int(element[5])
        v4 = int(element[6])
        cl = 0
        if v0 > v1 and v0 > v2 and v0 > v3 and v0 > v4: cl = 0
        if v1 >= v0 and v1 > v2 and v1 > v3 and v1 > v4: cl = 1
        if v2 >= v0 and v2 >= v1 and v2 > v3 and v2 > v4: cl = 2
        if v3 >= v0 and v3 >= v1 and v3 >= v2 and v3 > v4: cl = 3
        if v4 >= v0 and v4 >= v1 and v4 >= v2 and v4 >= v3: cl = 4

        if v4>0: cl=4
        elif v3>0: cl=3
        elif v2>0: cl=2
        elif v1>0: cl=1
        elif v0>0: cl=0

        w.record(int(cl), str(element[7]))
    w.save(shfileout)
    exit()

    for key, element in dataset.items():
        w.point(element[0], element[1])
        w.record(float(element[2]), int(element[3]), int(element[4]))
    w.save(shfileout)
    exit()

    i = 0
    with progressbar.ProgressBar(max_value=len(shapes)) as bar:
        for shaperec in sf.iterShapeRecords():
            i = i + 1
            w.shape(shaperec.shape)

            npoints = len(shaperec.shape.points)
            i0 = 0
            i1 = 1
            p1 = np.array([shaperec.shape.points[i0][0], shaperec.shape.points[i0][1]])
            p2 = np.array([shaperec.shape.points[i1][0], shaperec.shape.points[i1][1]])

            cl = 0;
            for datapoint in dataset:
                p3 = np.array([datapoint[0], datapoint[1]])
                if ((min(p1[0], p2[0]) < p3[0] and p3[0] < max(p1[0], p2[0]) and min(p1[1], p2[1]) < p3[1] and p3[
                    1] < max(p1[1], p2[1]))):
                    d = distance(p1, p2, p3)
                    if (d < 1.0):
                        cl = max(cl, datapoint[2])
                        break
            bar.update(i)
            w.record(shaperec.record[0], cl)

    w.save(shfileout)


'''
Questa funzione pare essere chiamata quando non sono disponibili punti (forse per evitare un errore?). Non sembra avere un'utilità concreta, visto che exists restituirà sempre false per come è scritta adesso e che quindi non aprirà neanche i file 
passati come parametro.
Forse è stata usata come funzione di debug?
'''
def notAvailableToPoints(csvdatafile, shfileout):
    import os
    exists = os.path.isfile('/path/to/file')
    if not exists:
        return
    w = shapefile.Writer()
    w.field('Coordinates', 'C')
    with open(csvdatafile) as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            lat = float(row[0])
            lon = float(row[1])
            (x, y, ZN, ZL) = utm.from_latlon(lat, lon)
            print(x, y, ZN, ZL)
            w.point(lon, lat)
            w.record(row[0]+","+row[1])
    w.save(shfileout)

'''
Questa funzione ricava i punti corrispondenti alle foto. Usata nella funzione main() quando downLoad è False
'''
def photosToPoints(csvdatafile, shfileout):
    w = shapefile.Writer()
    w.field('Class', 'N')
    w.field('PhotoId', 'C')
    with open(csvdatafile) as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            osmId = row[0]
            n1 = row[1]
            n2 = row[2]
            id = osmId + '_' + n1 + '_' + n2
            filename = row[3]
            cl = row[4]
            xy = filename.split('_')
            lat = float(xy[1])
            lon = float(xy[2])
            (x, y, ZN, ZL) = utm.from_latlon(lat, lon)
            print(x, y, ZN, ZL)
            w.point(lon, lat)
            w.record(int(cl), str(filename))

    w.save(shfileout)


'''
Questa funzione assegna dei voti ai punti? Nessun blocco successivo la invoca, forse è stata sostituita a un certo punto?
'''
def votesToPoints(csvdatafile, shfileout):
    # read the deep network output

    w = shapefile.Writer()
    w.field('AvgClass', 'N', decimal=2)
    w.field('StdDev', 'N', decimal=2)
    w.field('NumVotes', 'N')
    w.field('PhotoId', 'C')
    with open(csvdatafile) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            id = row[0]
            filename = row[1]
            cl = row[3]
            stdev = row[4]
            nvotes = row[5]
            xy = filename.split('_')
            lat = float(xy[1])
            lon = float(xy[2])
            (x, y, ZN, ZL) = utm.from_latlon(lat, lon)
            w.point(x, y)
            w.record(cl, stdev, nvotes, str(filename))
    w.save(shfileout)



'''
Questa funzione effettua un aggiornamento delle immagini? Nessun blocco la invoca. Il parametro della chiave api è da cambiare perché invalido. Forse riscarica le immagini del dataset inserendo all'interno del csv un'indicazione legata a
quando risalgono queste immagini in termini di data e ora
'''
def refreshImages(csv_dataset):
    import time

    folder, tail = os.path.split(csv_dataset)
    with open(csv_dataset) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            id = row[0]
            filename = row[1]
            arc = row[2]
            setname = row[3]
            w = row[4]
            date1 = time.strftime("%Y-%m-%d %H:%M:%S")
            date2 = time.strftime("%Y-%m-%d %H:%M:%S")

            fullFileName = folder + '/' + filename + ".jpg"

            if (os.path.exists(fullFileName)):
                continue

            xy = filename.split('_')
            lat = float(xy[1])
            lon = float(xy[2])
            heading = float(xy[3])
            print(lat, lon, heading)
            params = [{
                'size': '640x640',  # max 640x640 pixels
                'location': '40.7225,8.5517',
                'heading': '20.0',
                'pitch': '0.0',
                'key': 'xxx',
                'fov': 120
            }]
            # AIzaSyAz_hLf92j3vhkqm - XAAunjifXiwooni7E
            # AIzaSyCDVKtvxKZaes0 - lpeiEyW19c - co8lOAbY

            imageName = "downloads/image"

            params[0]['location'] = str(lat) + ',' + str(lon)
            params[0]['heading'] = str(heading)

            results = google_streetview.api.results(params)
            if results.metadata[0]['status'] != "OK":
                print('Error: ', filename)
                continue

            results.download_links(imageName)

            if results.metadata[0]['copyright'] != '© Google, Inc.':
                print('Indoor or not Google: ', filename)
                continue

            imageName += "/gsv_0.jpg"

            if (not os.path.exists(imageName)):
                shutil.rmtree("downloads/image")
                print("Unable to download image: " + filename)
                continue

            image = Image.open(imageName)

            new_image = Image.new('RGB', (640, 640))
            new_image.paste(image, (0, 0))

            new_image.save(fullFileName, "JPEG")
            newCsvFile = open(folder + '/newdataset.csv', 'a')
            newCsvFile.write(str(id) + ';' + filename + ';' + str(arc) + ';' + str(setname) + ';' + str(w) + ';' + str(
                date1) + ';' + str(date2) + '\n')
            newCsvFile.close()


'''
Questa funzione genera delle immagini tramite PIL che probabilmente dovranno essere visualizzate su interfaccia web, ma che in questa funzione vengono semplicemente salvate.
La funzione genera 4 rettangoli neri sull'immagine. Tutti e 4 partono dall'angolo in alto a sinistra, ma ognuno di loro ha una larghezza diversa, ma stessa altezza
'''
def produceWebImages(csvdatafile, inputFolder, outputFolder):
    from PIL import Image, ImageDraw
    with open(csvdatafile) as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            imageName = row[3]
            image = Image.open(inputFolder + '/' + imageName +'.jpg')
            draw = aggdraw.Draw(image)
            p = aggdraw.Pen("black", 4)
            draw.rectangle((0, 0,   image.size[0]/4, image.size[1]), p)
            draw.rectangle((0, 0,   image.size[0]/2, image.size[1]), p)
            draw.rectangle((0, 0, 3*image.size[0]/4, image.size[1]), p)
            draw.rectangle((0, 0,   image.size[0],   image.size[1]), p)
            draw.flush()
            image.save(outputFolder + '/' + imageName + ".jpg", "JPEG")


'''
Questa funzione effettua un calcolo di una misura (non specificata) su due coppie (lat1, lon1) e (lat2, lon2). Forse si tratta di una qualche distanza?
'''
def measure(lat1, lon1, lat2, lon2):
    R = 6378.137
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(lat1 * math.pi / 180) * math.cos(
        lat2 * math.pi / 180) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d * 1000

'''
Questa funzione utilizza googleplaces per scaricare le immagini e probabilmente ottenere altre informazioni aggiuntive
'''
def downloadPlaces(latMin, latMax, lonMin, lonMax, csvFile):
    import io

    gp = GooglePlaces('AIzaSyD9_MXZD1Cze9unuQlcElkgz-pezGlgEMI')

    dLon = measure(latMin, lonMin, latMin, lonMax)
    dLat = measure(latMin, lonMin, latMax, lonMin)

    print(dLat, dLon)

    nsLat = int(dLat / 100)
    nsLon = int(dLon / 100)

    dLat = (latMax - latMin) / nsLat
    dLon = (lonMax - lonMin) / nsLon

    print(nsLat, nsLon)

    for i in range(0, nsLat):
        for j in range(0, nsLon):

            lat = latMin + i * dLat
            lon = lonMin + j * dLon
            loc = (lat, lon)

            print(i, j, lat, lon)

            places = gp.search(location=loc, radius=100 * math.sqrt(2))

            npt = None
            if 'next_page_token' in places:
                npt = places['next_page_token']

            page = 1
            k = 0
            for res in places['results']:
                k = k + 1

                type = res['types'][0]
                for q in range(0, len(res['types'])):
                    if res['types'][q] != 'point_of_interest':
                        type = res['types'][q]
                        break

                if type == 'locality':
                    continue

                print(k, ' page ', page, '  ', res['id'], res['geometry']['location']['lat'],
                      res['geometry']['location']['lng'], res['name'], type)

                newCsvFile = io.open(csvFile, 'a', encoding="utf-8")
                newCsvFile.write(
                    str(i) + ',' + str(j) + ',' + res['id'] + ',' + str(res['geometry']['location']['lat']) + ',' + str(
                        res['geometry']['location']['lng']) + ',' + res['name'] + ',' + type + '\n')
                newCsvFile.close()

            while npt != None:

                places = gp.search(location=loc, radius=100 * math.sqrt(2), page_token=npt)
                page = page + 1
                for res in places['results']:
                    k = k + 1

                    type = res['types'][0]
                    for q in range(0, len(res['types'])):
                        if res['types'][q] != 'point_of_interest':
                            type = res['types'][q]
                            break

                    if type == 'locality':
                        continue

                    print(k, ' page ', page, '  ', res['id'], res['geometry']['location']['lat'],
                          res['geometry']['location']['lng'], res['name'], type)
                    newCsvFile = io.open(csvFile, 'a', encoding="utf-8")
                    newCsvFile.write(str(i) + ',' + str(j) + ',' + res['id'] + ',' + str(
                        res['geometry']['location']['lat']) + ',' + str(res['geometry']['location']['lng']) + ',' + res[
                                         'name'] + ',' + type + '\n')
                    newCsvFile.close()

                if not 'next_page_token' in places:
                    break

                npt = places['next_page_token']

    return

    gmaps = googlemaps.Client(key='xxx')
    loc = (39.2305, 9.1192)
    # loc = (45.4585, 9.1873)
    places = gmaps.places(type=['restaurants', 'train_station', 'bus_station', 'subway_station', 'transit_station'],
                          location=loc, radius=1000, language='it')
    print(places)
    npt = None
    if 'next_page_token' in places:
        npt = places['next_page_token']
    i = 0
    for result in places['results']:
        i = i + 1
        print(i, ' ', result['name'])
    while npt != None:

        print('npt=', npt)
        places = gmaps.places(type=['restaurants', 'train_station', 'bus_station', 'subway_station', 'transit_station'],
                              location=loc, radius=1000, language='it', page_token=npt)
        print(places)
        for result in places['results']:
            i = i + 1
            print(i, ' ', result['name'])
        npt = places['next_page_token']

'''
Questa funzione aggiunge attributi a un dataset nel path datasetFileName e scrive questa nuova versione in un altro path outputDatasetFileName
'''
def addPlacesAttributes(datasetFileName, outputDatasetFileName):
    import time

    open(outputDatasetFileName, 'w').close()

    go = False
    with open(datasetFileName) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in reader:
            id = row[0]
            filename = row[1]

            arc = row[2]
            setname = row[3]
            wr = row[4]
            cl = int(row[7])
            pre = int(row[8])
            err = int(row[9])
            date1 = time.strftime("%Y-%m-%d %H:%M:%S")
            date2 = time.strftime("%Y-%m-%d %H:%M:%S")

            xy = filename.split('_')
            lat = float(xy[1])
            lon = float(xy[2])
            heading = float(xy[3])
            gp = GooglePlaces('AIzaSyD9_MXZD1Cze9unuQlcElkgz-pezGlgEMI')
            loc = (lat, lon)
            places = gp.search(location=loc, radius=100)
            p100 = len(places['results'])

            places = gp.search(location=loc, radius=250)
            p250 = len(places['results'])

            places = gp.search(location=loc, radius=500)
            p500 = len(places['results'])

            print(p100, p250, p500)
            # print(places['results'])

            newCsvFile = open(outputDatasetFileName, 'a')
            newCsvFile.write(str(id) + ',' + filename + ',' + str(arc) + ',' + str(setname) + ',' + str(wr) + ',' + str(
                date1) + ',' + str(date2) + ',' + str(cl) + ',' + str(pre) + ',' + str(err) + ',' + str(
                p100) + ',' + str(p250) + ',' + str(p500) + '\n')
            newCsvFile.close()

'''
Questa funzione carica il file csv legato al dataset di places
'''
def loadPlaces(datasetFileName, places):
    with open(datasetFileName, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            key = row[0]
            lat = row[1]
            lon = row[2]
            name = row[3]
            type = row[4]
            if type == 'neighborhood':
                continue
            places[key] = (lat, lon, name, type)
    return places

'''
Penso che restituisce i numeri n1,n2,n3 di posti che distano r1,r2,r3 dal punto (lat,lon)
'''
def findNumberOfPlaces(places, r1, r2, r3, lat, lon):
    n1 = n2 = n3 = 0
    for key, place in places.items():
        latp = float(place[0])
        lonp = float(place[1])
        d = measure(lat, lon, latp, lonp)
        if d <= r1:
            n1 = n1 + 1
            print(place[2])
        if d <= r2:
            n2 = n2 + 1

        if d <= r3:
            n3 = n3 + 1
    return n1, n2, n3


'''
Questa funzione effettua una selezione di posto rispetto a quelli inseriti in fileOut. Forse questa è la famosa funzione che filtra in base alle aree urbane?
'''
def filterPlaces(fileIn, fileOut):
    places = {}
    q = 0
    with open(fileIn, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            q = q + 1
            i = row[0]
            print(i)
            j = row[1]
            print(j)
            key = row[2]
            lat = row[3]
            lon = row[4]
            name = row[5]
            type = row[6]
            print(lat, lon, name, type)
            places[key] = (lat, lon, name, type)
    newCsvFile = open(fileOut, 'w', encoding='utf-8')
    for key, element in places.items():
        newCsvFile.write(str(key) + ',' + str(element[0]) + ',' + str(element[1]) + ',' + str(element[2]) + ',' + str(
            element[3]) + '\n')
    newCsvFile.close()

'''
Questa funzione prende un dataframe caricato con geopandas e lo trasforma in OSM.
Purtroppo questa funzione è irrilevante ai nostri fini perchè lavora con strade e prende la x e la y, cosa che non si può fare con un poligono come sto lavorando io. Forse questa
funzione lavorava con un altro formato diverso da quello a cui ho a disposizione
'''
def geoDataFrameToOSM(geopandas_data_frame):
    import xml.etree.ElementTree as ET
    osm_root = ET.Element('osm', attrib={
        'version': '0.6',
        'generator': 'custom python script'
        })

    i = -1
    for index, row in geopandas_data_frame.iterrows():
        geometry = row.geometry
        current_node = ET.SubElement(osm_root, 'node', attrib={
            'id': str(i),
            'lat': str(geometry.y),
            'lon': str(geometry.x),
            'changeset': 'false'})

        for column in geopandas_data_frame.loc[:, ['addr:housenumber', 'addr:street']]:
            ET.SubElement(current_node, 'tag', attrib={
                'k': column,
                'v': row[column]})

        i -= 1
    output_file = ET.ElementTree(element=osm_root)
    output_file.write('test.osm')


'''
Questo "Main" probabilmente eseguiva le azioni principali per un progetto per cui è stato creato questo script.
Se download è a True, viene creato il grafo del posto richiesto, salvato come shapefile (tramite la libreria ox) e poi
da questo grafo neviene generato un altro. Se download è a false si cercano di ottenere le footprint e scaricare le foto dal grafo creato
'''
def main():
    place = 'Sardegna'
    datasetFolder = "dataset_" + place
    downLoad = True
    simulation = False
    maxRequests = 28000
    if downLoad:
        if (not os.path.exists(datasetFolder)):
           os.mkdir(datasetFolder)
        G = ox.graph_from_place(place, simplify=False)
        ox.plot_graph(G)
        ox.save_graph_shapefile(G, filename=place)
        G2 = ox.simplify_graph(G)
        ox.save_graphml(G2, filename=place + ".graphml")
        createGraphFromGraphML("data/" + place + ".graphml", datasetFolder + "/" + place + "_u.graphml")
        exit()
    else:
        Fp = ox.features_from_place(place, tags={'building': True}) #cambiato footprints in geometries
        minx, miny, maxx, maxy = Fp.geometry.total_bounds
        print(minx, miny, maxx, maxy)
        #osmFp = ox.osm_footprints_download_OSM(north=maxy, south=miny, east=maxx, west=minx)
        #osm_file = open(datasetFolder + "/" + place+"_footprint.osm", "w")
        #osm_file.write(osmFp[0])
        #osm_file.close()
        #exit()
        Fp = Fp.to_crs({'init': 'epsg:32632'})
        minx, miny, maxx, maxy = Fp.geometry.total_bounds
        print(minx, miny, maxx, maxy)
        downloadPhotosFromGraphML(datasetFolder + "/" + place + "_u.graphml", Fp, simulation, maxRequests)
        photosToPoints(datasetFolder + "/data.csv", datasetFolder + "/" + place + "_photos_points")
        notAvailableToPoints(datasetFolder + "/notavailable.csv", datasetFolder + "/" + place + "_not_available_photos_points")
        exit()


'''
This function gets the .gml file from OSM file by specifying 4 coordinates of the bounding box that encloses the area you would like to download.
To get these coordinates, go to this link: https://www.openstreetmap.org/export#map=16/41.9407/12.5183 and select manually the bounding box
In the left panel, you will see 4 float values: the upper one is the north value. the lower one is the south value, the right one is the east value and the left one is the west value.
The order to get them in the function is north,south,east and west. The function will save the graphml file and the shape file
'''
def get_gml_from_bbox(path,bbox):
    G = ox.graph_from_bbox(bbox=bbox, simplify=True)
    ox.io.save_graphml(G, filepath=path)

'''
This function converts a shapefile in the graphml format. This is useful if you had to manually delete nodes or edges via the Qgis interface and now want the graph back.
To get correctly the files the 'path' parameter is the path to a folder that contains the shapefile of nodes and the shapefile of edges.
'''

def convert_from_shp(path):
    draw_graph = False
    nodes = gp.read_file(f'{path}/nodes.shp').set_index(['osmid'])
    edges = gp.read_file(f'{path}/edges.shp').set_index(['u','v','key'])
    G =  ox.graph_from_gdfs(nodes,edges)
    ox.io.save_graphml(G, filepath='dataset/valdala.graphml')
    if draw_graph:
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos)      
        plt.show()  

'''
This function gets the graph and the photos from StreetView of the corresponding edges. If get_gml is set to true, the graph will be downloaded and saved in .graphml and .shp via the get_gml_from_bbox function.
If get_from_shp is set to true, the graphml will be converted from the shapefile. If this is true, then get_gml must be false, since it is pointless to convert the graph from the shapefile since they get saved
simultaneously. The convert_from_shp function is needed only if the graph was precedently modified via removing nodes or edges in the shapefile via Qgis and then these updates need to be reflected on the .graphml file
'''
def get_graph_and_photos(path, bbox):
    north = 41.9463
    south = 41.9356
    east  = 12.5297
    west = 12.5109
    get_gml = True
    get_from_shp = False
    if get_gml:
        get_gml_from_bbox(path=path,bbox=bbox)
    elif get_from_shp:
        convert_from_shp('valdala')

    fp = ox.features_from_bbox(bbox=(north,south,east,west),tags={'building': True})
    downloadPhotosFromGraphML(path, footPrints = fp, Simulation=False, maxRequests=28000)

def test_api_key():
    g = networkx.read_graphml('dataset/valdala.graphml')
    north = 41.9463
    south = 41.9356
    east  = 12.5297
    west = 12.5109
    footPrints = ox.features_from_bbox(bbox=(north,south,east,west),tags={'building': True})
    Simulation = False
    folder ='dataset'
    networkx.set_edge_attributes(g, name='imageFlag', values=0)
    list_edges = [e for e in g.edges(data=True)]
    u,v, data = list_edges[0]
    uId = u
    vId = v
    edgeId = data['osmid']

    lat0, lon0, lat1, lon1 = getLongestEdge(g, u, v)
    #print(edgeId  + " " + str(lat0) + " " + str(lon0) + " " + str(lat1) + " " + str(lon1))

    (x0, y0, zone, north) = utm.from_latlon(lat0, lon0)
    (x1, y1, zone, north) = utm.from_latlon(lat1, lon1)

    p0 = Point(x0, y0, 0)
    p1 = Point(x1, y1, 0)
    dist = math.dist((x0,y0),(x1,y1))
    a = Vector.from_points(p0, p1)
    fpRtree = createFootprintRTree(footPrints)
    # l = a.magnitude()


    heading = math.atan2(p1.y - p0.y, p1.x - p0.x)
    heading = heading * (180.0 / math.pi)
    if data['imageFlag'] == 0: # se l'immagine non è stata scaricata
        if dist < 100:
            alpha = 0.5
            res = saveImageOfEdge360(edgeId, uId, vId, p0, p1, heading, alpha, 0, folder, zone, north, footPrints, fpRtree, Simulation)     #   scarica l'immagine dell'edge
            if res==1:
                data['imageFlag'] = 1   # l'immagine è stata scaricata, segnalo
                #print('--- Saved new image for osmid=', edgeId)
        else:
            alpha=0.3
            res1 = saveImageOfEdge360(edgeId, uId, vId, p0, p1, heading, alpha, 0, folder, zone, north, footPrints, fpRtree, Simulation)     #   scarica l'immagine dell'edge
            alpha=0.7
            res2 = saveImageOfEdge360(edgeId, uId, vId, p0, p1, heading, alpha, 0, folder, zone, north, footPrints, fpRtree, Simulation)     #   scarica l'immagine dell'edge
            if res1 and res2 == 1:
                data['imageFlag'] = 1
    else:
        print("Already downloaded")

def test_conversion():
    convert360To180(csvdatafile='dataset/data.csv',inputFolder='dataset',outputFolder='dataset180')

if __name__ == "__main__":
    main()
    #test_conversion()
    # test_api_key()
    # get_graph_and_photos()