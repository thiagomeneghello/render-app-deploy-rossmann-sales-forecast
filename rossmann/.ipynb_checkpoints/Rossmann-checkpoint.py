import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Rossmann( object ):
    def __init__( self ):
        #self.home_path = ''
        self.comp_distance_scaler = pickle.load(open('pickle/comp_distance_scaler.pkl', 'rb'))
        self.comp_timemonth_scaler = pickle.load(open('pickle/comp_timemonth_scaler.pkl', 'rb'))
        self.promo_timeweek_scaler = pickle.load(open('pickle/promo_timeweek_scaler.pkl', 'rb'))
        self.year_scaler = pickle.load(open('pickle/year_scaler.pkl', 'rb'))
        self.store_type_scaler = pickle.load(open('pickle/store_type_scaler.pkl', 'rb'))

    def data_cleaning( self, df1):
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo',
                    'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                    'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                    'Promo2SinceYear', 'PromoInterval']

        snakecase = lambda x: inflection.underscore(x)
        cols_new = list(map(snakecase, cols_old))
        df1.columns = cols_new

        ## 1.3 Data types
        df1['date'] = pd.to_datetime(df1['date'])

        ## 1.5 Fillout NA
        #competition_distance
        df1['competition_distance'] = [90000 if math.isnan(i) else i for i in df1['competition_distance']]

        #competition_open_since_month
        df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else 
                                                        x['competition_open_since_month'], axis = 1)
        #competition_open_since_year
        df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else
                                                       x['competition_open_since_year'], axis = 1)            
        #promo2_since_week
        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else
                                             x['promo2_since_week'], axis = 1) 
        #promo2_since_year
        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else 
                                             x['promo2_since_year'], axis = 1)     
        #promo_interval  
        month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
        df1['promo_interval'] = df1['promo_interval'].fillna(0)
        df1['month_map'] = df1['date'].dt.month.map(month_map)
        df1['is_promo'] = df1[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval']==0 else
                                                                     1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)

        ## 1.6 Change types

        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int) 
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int) 
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)

        return df1

    def feature_engineering( self, df2 ):
        # year
        df2['year'] = df2['date'].dt.year
        #month
        df2['month'] = df2['date'].dt.month
        #day
        df2['day'] = df2['date'].dt.day
        #week of year
        df2['week_of_year'] = df2['date'].dt.isocalendar().week
        df2['week_of_year'] = df2['week_of_year'].astype(int)
        #year week
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')

        # competition since
        df2['competition_since'] = df2.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'],
                                                                         month=x['competition_open_since_month'],
                                                                         day=1), axis=1)

        df2['competition_time_month'] = ((df2['date'] - df2['competition_since'])/30).apply(lambda x: x.days).astype(int)


        # promo since
        df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)
        df2['promo_since'] = df2['promo_since'].apply(lambda x: datetime.datetime.strptime(x +'-1',
                                                                                           '%Y-%W-%w') - 
                                                                                            datetime.timedelta(days=7))
        df2['promo_time_week'] = ((df2['date'] - df2['promo_since'])/7).apply(lambda x: x.days).astype(int)

        #assortment
        df2['assortment'] = ['basic' if i == 'a' else 'extra' if i == 'b' else 'extended' for i in df2['assortment']]

        #state_holiday
        df2['state_holiday'] = ['public_holiday' if i == 'a' else 'easter_holiday' if i == 'b' else 'christmas' if i == 'c' else 'regular_day' for i in df2['state_holiday']]

        ## 3.1 Filtragem das linhas
        df2 = df2.loc[df2['open'] != 0, :].reset_index()
        ## 3.2 Seleção das colunas
        cols_drop = ['open', 'month_map', 'promo_interval']
        df2 = df2.drop(cols_drop, axis=1)

        return df2

    def data_preparation( self, df5 ):
        df5['competition_distance'] = self.comp_distance_scaler.fit_transform(df5[['competition_distance']].values)
        df5['competition_time_month'] = self.comp_timemonth_scaler.fit_transform(df5[['competition_time_month']].values)
        df5['promo_time_week'] = self.promo_timeweek_scaler.fit_transform(df5[['promo_time_week']].values)
        df5['year'] = self.year_scaler.fit_transform(df5[['year']].values)

        ## 5.3 Transformation
        #state_holiday - One Hot Encoding, bom para estado de coisas
        df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'], dtype='int64')

        #store_type - Label Encoding, bom para variáveis categóricas sem ordem ou relevância
        df5['store_type'] = self.store_type_scaler.fit_transform(df5['store_type'])

        #assortment - Ordinal Encoding, bom para variáveis com intraordem.
        assort_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df5['assortment'] = df5['assortment'].map(assort_dict)

        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x*(2*np.pi/7)))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x*(2*np.pi/7)))

        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x*(2*np.pi/12)))
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x*(2*np.pi/12)))

        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x*(2*np.pi/52)))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x*(2*np.pi/52)))

        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x*(2*np.pi/30)))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x*(2*np.pi/30)))

        cols_selected = ['store', 'promo','store_type', 'assortment', 'competition_distance',
                         'competition_open_since_month','competition_open_since_year','promo2',
                         'promo2_since_week','promo2_since_year','competition_time_month','promo_time_week',
                         'day_of_week_sin','day_of_week_cos','month_cos','month_sin','week_of_year_cos',
                         'week_of_year_sin','day_sin','day_cos']

        return df5[cols_selected]

    def get_prediction(self, model, original_data, test_data):
        #prediction
        pred = model.predict(test_data)
        # join pred into original date
        original_data['predictions'] = np.expm1(pred)

        return original_data.to_json( orient='records', date_format = 'iso')