import pandas as pd
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt


class geosteering_dataset:
    def __init__(self):
        pass

    def get_data(self, dataset_type=3):

        data_path_root = '/content/drive/MyDrive/Geosteering/Datasets/'
        df_poro = pd.read_csv(
            data_path_root + f'Dataset {dataset_type}/Groningen dataset_{dataset_type} PROCESSED_PORO.csv',
            low_memory=False, header=None)
        df_Sw = pd.read_csv(
            data_path_root + f'Dataset {dataset_type}/Groningen dataset_{dataset_type} PROCESSED_Sw.csv',
            low_memory=False, header=None)
        df_Vsh = pd.read_csv(
            data_path_root + f'Dataset {dataset_type}/Groningen dataset_{dataset_type} PROCESSED_Vsh.csv',
            low_memory=False, header=None)
        df_perm = pd.read_csv(
            data_path_root + f'Dataset {dataset_type}/Groningen dataset_{dataset_type} PROCESSED_perm.csv',
            low_memory=False, header=None)

        if dataset_type == 2:
            self.scale_x = 10
        elif dataset_type == 3:
            self.scale_x = 3

        # clean and rotate the dataframe before make it an array, and make the numbers as float
        # porosity
        df_poro = (df_poro.iloc[:, 1::2]).astype(float)
        arr_poro = (np.array(df_poro)).T
        self.arr_poro = arr_poro[::-1]

        # water saturation
        df_Sw = (df_Sw.iloc[:, 1::2]).astype(float)
        arr_Sw = (np.array(df_Sw)).T
        self.arr_Sw = arr_Sw[::-1]

        # V_shale
        df_Vsh = (df_Vsh.iloc[:, 1::2]).astype(float)
        arr_Vsh = (np.array(df_Vsh)).T
        self.arr_Vsh = arr_Vsh[::-1]

        # permeability
        df_perm = (df_perm.iloc[:, 1::2]).astype(float)
        arr_perm = (np.array(df_perm)).T
        self.arr_perm = arr_perm[::-1]

        # here we define the values of the parameters m and n from Archi equation
        n = 2  # saturation exponent, which varies from 1.8 to 4.0 but normally is 2.0
        # m = arr_Vsh_scaled # cementation exponent, which varies from 1.7 to 3.0 but normally is 2.0
        m = 2

        # computing of the resestivity profile by  Archi equation

        self.arr_Rt = 0.127 / (arr_poro ** (m) * arr_Sw ** (n))

        self.P = self.arr_perm * (1 - self.arr_Sw) * self.arr_poro

        # computing of the density profile
        rho_f = 1.0  # density of fluid (g/cm3)
        rho_ss = 2.6  # density of sandstones (g/cm3)
        rho_sh = 2.3  # density of shales (g/cm3)

        arr_m_dens = (1 - arr_Vsh) * rho_ss + arr_Vsh * rho_sh  # matrix density accounting for V shale
        self.arr_dens = arr_m_dens - arr_poro * (arr_m_dens - rho_f)  # calculate bulk density

        # computing of the neutron profile
        k = 0.3  # lithological coefficient
        self.arr_neut = self.arr_poro + k * self.arr_Vsh  # calculating of neutron profile

        # computing of the GR profile
        self.arr_gr = 0.1 * ((-100 * self.arr_Vsh ** 2 + 340 * self.arr_Vsh + 49) ** 0.5 - 7)

        # computing of the travel time profile

        tt_ss = 44.4  # travel time for sandstone matrix
        tt_f = 185  # travel time for fluid
        tt_sh = 70  # shales

        self.arr_tt = self.arr_poro * (tt_f - tt_ss) + self.arr_Vsh * (tt_sh - tt_ss) + tt_ss

    def visualize(self, array, title, x_label, y_label, cmap='hsv', size_inches=(20, 2)):
        fig1, ax = plt.subplots()
        fig1.set_size_inches(20, 2)
        plt.title(f'{title}')
        plt.xlabel(f'Distance *%1.0f [m]' % self.scale_x)
        plt.ylabel('Depth')
        shw1 = ax.imshow(array, aspect='auto', interpolation='nearest', cmap='hsv')
        bar = plt.colorbar(shw1)
        bar.set_label('Porosity')

    @staticmethod
    def interp(array, scale=1, method='linear'):
        # methods: nearest, cubic, linear
        x = np.arange(array.shape[1] * scale)[::scale]
        y = np.arange(array.shape[0] * scale)[::scale]
        x = np.arange(array.shape[1])
        x_in_grid, y_in_grid = np.meshgrid(x, y)
        x_out, y_out = np.meshgrid(np.arange(max(x) + 1), np.arange(max(y) + 1))
        array = np.ma.masked_invalid(array)
        x_in = x_in_grid[~array.mask]
        y_in = y_in_grid[~array.mask]
        return scipy.interpolate.griddata((x_in, y_in), array[~array.mask].reshape(-1), (x_out, y_out), method=method)

    @staticmethod
    def add_gaussian_noise(array, percent_of_noised_array=0.001):
        """
        Args:
            array : numpy array
        Return :
            array : numpy array with gaussian noise added
        """
        mean = np.nanmean(array)
        std = np.nanstd(array)

        gaus_noise = np.random.normal(mean, std, array.shape)
        gaus_noise = np.where(gaus_noise < 0, gaus_noise, abs(gaus_noise))

        indices = np.random.choice(gaus_noise.shape[1] * gaus_noise.shape[0], replace=False,
                                   size=int(gaus_noise.size * (1 - percent_of_noised_array)))
        gaus_noise[np.unravel_index(indices, gaus_noise.shape)] = 0
        noised_array = array + gaus_noise
        # print(np.count_nonzero(gaus_noise)/gaus_noise.size)

        return noised_array




