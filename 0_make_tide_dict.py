import numpy as np
import h5py
import os 
import sys
import datetime
import pickle
import utils as ut

## 1. config
# e.g.) 20220601 ~ 20221130 
# // no folder on 20220719 
# // path to save dictionary pickle files in './proc_data'

# path of folder of water level h5 files (inputs)
datafd = ut.env['datafd']

os.makedirs(f'{datafd}/water', exist_ok=True)

water_level_folder = f'{datafd}/water'
# target year and months
year = '2022'
lddic = {6:30, 7:31, 8:31, 9:30, 10:31, 11:30}
# dates to filter out
no_folder_dates = ['20220719']
# path for saving 1) grid idx dict. 2) daily tide dict. (outputs)
save_path = '.' 


## 2. make day list 

daylist = []
month_list = list(lddic.keys())
start_month = int(month_list[0])
end_month = int(month_list[len(month_list)-1])
for m in range(start_month, end_month+1):
    for d in range(1, lddic[m]+1):
        daylist.append(f'{year}{m:02d}{d:02d}')
# remove dates which don't have folder from daylist
for date in no_folder_dates:
    daylist.remove(date)
if len(daylist) != 0: print('1. Make day list is done!')
else: print('No date in day list')


## 3. make grid name list 

# list h5 filenames in one day folder under the 'water' folder
file_list = []
for (path, dir, files) in os.walk(f'{water_level_folder}/{str(daylist[0])}'):
    for filename in files:
        if '.h5' in filename:
            file_list.append(path + '/' + filename)
# get grid names from h5 filename
grid_name_list =[]
for filename in file_list:
    grid_name = filename.split('/')[-1][11:16]
    grid_name_list.append(grid_name)

if len(grid_name_list) == 108: print('2. Make grid name list is done!')
else: print('Error: the number of grids must be 108')


## 4. make grid:index in (17 x 17 matrix) dictionary 

gidx_north_dict = {}
gidx_west_dict = {}

for grid_name in grid_name_list:
    f = h5py.File(f'{water_level_folder}/{str(daylist[0])}/104KR00KR4_{grid_name}.h5', 'r')
    north_bound_lat = f.attrs['northBoundLatitude']
    gidx_north_dict[grid_name] = north_bound_lat
    west_bound_lon = f.attrs['westBoundLongitude']
    gidx_west_dict[grid_name] = west_bound_lon
    
sorted_gidx_north_dict = {k: v for k, v in sorted(gidx_north_dict.items(), key=lambda item: item[1], reverse=True)}
sorted_gidx_west_dict = {k: v for k, v in sorted(gidx_west_dict.items(), key=lambda item: item[1])}

grid_name_row_dict = {k: int((39.0-v)*2) for k, v in sorted_gidx_north_dict.items()}
grid_name_col_dict = {k: int((v-124.0)*2) for k, v in sorted_gidx_west_dict.items()}

# grid_name_idx_dict is {grid_name: (i, j)}
grid_name_idx_dict = {}
for name in grid_name_row_dict.keys():
    grid_name_idx_dict[name] = [grid_name_row_dict[name], grid_name_col_dict[name]]

if len(grid_name_idx_dict) == 108: print('4. Make grid:index dictionary is done!')
else: print('Error: the number of grids must be 108')


# Function to call: make tide label dictionary {date: (24hours, i, j ) 3d array} 
def make_day_grid_hour_label_dict(date, grid_name):
    dt = date
    grid = grid_name
    temp_pro_dt = datetime.datetime.strptime(dt, '%Y%m%d') + datetime.timedelta(days=1)
    pro_dt = temp_pro_dt.strftime('%Y') + temp_pro_dt.strftime('%m') + temp_pro_dt.strftime('%d')

    # mat_51_list: each element is (51 x 51) points in 1 grid on 1 day
    mat_51_list = []
    f = h5py.File(f'{water_level_folder}/{dt}/104KR00KR4_{grid}.h5', 'r')
    for group_idx in [i for i in '10,11,12,13,14,15,16,17,18,19,20,21,22,23,24'.split(',')]:
        # group010 is 0 o'clock of dt, group23 is 14 o'clock of dt
        mat_51 = f.get(f'WaterLevel/WaterLevel.01/Group_0{group_idx}/values')[()]
        mat_51_list.append(mat_51)
    f = h5py.File(f'{water_level_folder}/{pro_dt}/104KR00KR4_{grid}.h5', 'r')
    for group_idx in [i for i in '01,02,03,04,05,06,07,08,09'.split(',')]:
        # group001 is 15 o'clock of dt(a day before pro_dt), group009 is 23 o'clock of dt(a day before pro_dt)
        mat_51 = f.get(f'WaterLevel/WaterLevel.01/Group_0{group_idx}/values')[()]
        mat_51_list.append(mat_51)

    mat_51_array = np.array(mat_51_list) # 3d array (24 x 51 x 51), each table index means o'clock of dt
    mat_2601_array = np.reshape(mat_51_array, (24, 2601)) # 2d array (24 x 2601) , each table index means o'clock of dt
    day_grid_hourly_avg_wl = np.nanmean(mat_2601_array, axis=1) # 1d array (24,) , each index means o'clock of dt

    # figure out high/mid/low tide hours with (24, ) 1d array

    prev_diff = np.insert(np.flip(np.diff(np.flip(day_grid_hourly_avg_wl))), 0 , 0)
    pro_diff = np.insert(np.diff(day_grid_hourly_avg_wl), (len(day_grid_hourly_avg_wl)-1), 0)

    temp_low = np.intersect1d(np.where(prev_diff > 0), np.where(pro_diff > 0))
    temp_high = np.intersect1d(np.where(prev_diff < 0), np.where(pro_diff < 0))

    low_hours = []
    high_hours = []

    for i in temp_low:
        low_hours.append(i-1)
        low_hours.append(i)
        low_hours.append(i+1)

    for i in temp_high:
        high_hours.append(i-1)
        high_hours.append(i)
        high_hours.append(i+1)

    grid_hour_label_dict = {} # (24 x 1 x 1) of cube
    for hour in range(24):
        if hour in high_hours: grid_hour_label_dict[hour] = 1
        elif hour in low_hours: grid_hour_label_dict[hour] = -1
        else: grid_hour_label_dict[hour] = 0
    
    return grid_hour_label_dict


## 5. execute main to make tide label dictionary

print('5. Making tide label dictionary ...')

except_date_list = []
for date in no_folder_dates:
    temp_prev_dt = datetime.datetime.strptime(date, '%Y%m%d') - datetime.timedelta(days=1)
    prev_dt = temp_prev_dt.strftime('%Y') + temp_prev_dt.strftime('%m') + temp_prev_dt.strftime('%d')
    except_date_list.append(prev_dt)

# result is {date: (24hours, i, j ) 3d array} 
result = {}
for date in daylist:
    if date in except_date_list: continue
    # cube is 3d array. 
    # 24 layers means 24 hours, and an element in each (17 x 17) matrix is the tide label of a grid 
    cube = np.full((24,17,17), np.nan) 
    for grid in grid_name_list:
        a = make_day_grid_hour_label_dict(date, grid)
        for hour in a.keys():
            cube[hour, grid_name_idx_dict[grid][0], grid_name_idx_dict[grid][1]] = a[hour]
    result[date] = cube

# save result as pickle
name = 'daily_tide_dict'
with open(f"{save_path}/{name}.pickle", "wb") as file:
    pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
if os.path.exists(f"{save_path}/{name}.pickle"): print(f'{name} has been saved!')
else: print(f'{name} is NOT saved.')
