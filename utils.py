import csv
from multiprocessing import cpu_count
from itertools import product
import numpy as np

cpus = cpu_count()

env = {r[0]: r[1] for r in csv.reader(open('settings.csv'))}

M = int(env['gridsize'])

def proc_ton(x, shiptype):
  if shiptype == 'B':
    if 0<x<3: return '3'
    elif x<5: return '5'
    elif x<10: return '10'
    elif x<50: return '50'
    elif x<100: return '100'
    elif 100<=x: return '100P'
    else: return 'NT'
  else:
    if 0<x<20: return '20'
    elif x<100: return '100'
    elif x<300: return '300'
    elif x<500: return '500'
    elif x<1000: return '1000'
    elif x<10000: return '10000'
    elif 10000<=x: return '10000P'
    else: return 'NT'

def ll2tiderc(lat, lon):
  ridx = int((39-lat)*2)
  cidx = int((lon-124)*2)
  return (ridx, cidx)

def ll2idx(lat, lon):
  ridx = int((int(env['maxlat'])-lat)*M)
  cidx = int((lon-int(env['minlon']))*M)
  return (ridx, cidx)

def idx2ll(ridx, cidx):
  lat = int(env['maxlat']) - ((ridx + 0.5) / M)
  lon = ((cidx + 0.5) / M) + int(env['minlon'])
  return [lat, lon]

cslist = 'CMAT SMAT'.split()
tidelist = list('LMH')
tonlist = dict(B='3 5 10 50 100 100P NT'.split(), NB='20 100 300 500 1000 10000 10000P NT'.split())
ttlist = [f'B_{t}' for t in tonlist['B']] + [f'NB_{t}' for t in tonlist['NB']]

matlist = cslist[:]
matlist += ['_'.join(t) for t in product(cslist, tidelist)]
matlist += ['_'.join(t) for t in product(cslist, tidelist, ttlist)]

def AStarPostSmoothing(b, p, distcap = int(M/100)):
  if len(p)==0: return []
  n = len(p)-1
  k = 0
  previ = 0
  t = []
  t.append(p[0])
  for i in range(1, n-1):
    # print(i)
    if not lineOfSight(b, t[k], p[i+1]): k += 1; t.append(p[i]); previ = i
    else:
      if i-previ>=distcap: k += 1; t.append(p[i]); previ = i
  k += 1; t.append(p[n])
  return t

def lineOfSight(b, p1, p2):
  n = 4  # Steps per unit distance
  dxy = int((np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)) * n)
  i = np.rint(np.linspace(p1[0], p2[0], dxy)).astype(int)
  j = np.rint(np.linspace(p1[1], p2[1], dxy)).astype(int)
  has_collision = np.any(b[i, j])
  return ~has_collision

def nnz(a,x,y):
  if a[x,y]>0: return x, y
  tmp = a[x,y]
  a[x,y] = 0
  r,c = np.nonzero(a)
  a[x,y] = tmp
  min_idx = ((r - x)**2 + (c - y)**2).argmin()
  return r[min_idx], c[min_idx]

EUC = [
[99, 70, 99],
[70, 0, 70],
[99, 70, 99],
]

EUC_N = [
[99, 70, 99],
[70, 0, 70],
[0,0,0],
]

EUC_NE = [
[99, 70, 99],
[0, 0, 70],
[0,0,99],
]

EUC_NW = [
[99, 70, 99],
[70, 0, 0],
[99,0,0],
]

EUC_S = [
[0,0,0],
[70, 0, 70],
[99, 70, 99],
]

EUC_SE = [
[0,0,99],
[0, 0, 70],
[99, 70, 99],
]

EUC_SW = [
[99,0,0],
[70, 0, 0],
[99, 70, 99],
]

EUC_E = [
[0, 70, 99],
[0, 0, 70],
[0, 70, 99],
]

EUC_W = [
[99, 70, 0],
[70, 0, 0],
[99, 70, 0],
]
