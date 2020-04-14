#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 23:13:35 2020

@author: Alex
"""

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import OrdinalEncoder




def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns



#Custom Transformer that extracts columns passed as argument to its constructor 
class RedefineDTypes(BaseEstimator, TransformerMixin):
    #Class Constructor 
    def __init__(self):
        pass
    
    #Return self nothing else to do here    
    def fit(self, X, y = None):
        #Detect the type
        cat_feat = []
        cont_feat = []
        bin_feat = []
        
        for column in X :
            
            name = column
            name_type = X[column].dtype
            if name_type == "object" : 
                cat_feat.append(name)
                #print("category", name, name_type)
            
            elif (name_type == "int64") | (name_type == "float64") : 
                n_modalities = X[column].nunique()
                if n_modalities == 2 : 
                    bin_feat.append(name)
                    #print("binary", name, n_modalities)

                else : 
                    cont_feat.append(name)
                    #print("continuous", name, n_modalities)
            
            else : 
                print("unknown type", name, name_type)
            
        self.cat_feat = cat_feat
        self.cont_feat = cont_feat
        self.bin_feat = bin_feat
        
        #print(len(cat_feat+cont_feat+bin_feat))
        
        return self
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        X_out = X.copy()
        X_out[self.cat_feat] = X[self.cat_feat].astype("category")
        X_out[self.bin_feat] = X[self.bin_feat].replace([True, False], [1,0]).astype(float)
        #X_out[self.bin_feat] = X[self.bin_feat].astype("bool")
        X_out[self.cont_feat] = X[self.cont_feat].astype("float")
        
        return X_out
    
    
class Train_Test_Processor(BaseEstimator, TransformerMixin) : 
    def __init__(self) : 
        pass
    
    def fit(self, X, y=None) : 
        return self
    
    def transform(self, X, y=None) : 
        nan_as_category = False
        
        df = X.copy()
        df = df[df['CODE_GENDER'] != 'XNA']
        
        for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
            df[bin_feature], uniques = pd.factorize(df[bin_feature])

        # NaN values for DAYS_EMPLOYED: 365.243 -> nan
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
        
        # Some simple new features (percentages)
        df['DAYS_EMPLOYED'] = -df['DAYS_EMPLOYED']
        df['DAYS_BIRTH'] = -df['DAYS_BIRTH']
        df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
        df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        
        return df
    
    
class Bureau_and_balance(BaseEstimator, TransformerMixin) :
    def __init__(self, bureau_data, bb_data) : 
        #Instead of Loading data on each transform step, i'd rather load in outside,
        #integrate it in the object attributes and then select it.
        self.bureau = bureau_data
        self.bb = bb_data
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None) :
        #Read the whole set
        bureau = self.bureau
        bb = self.bb
        
        #Perhaps the MainSet was computed only on a few raw. So lets no aggregate every
        #Instance if it will not be usefull later. Extract needed index, and aggregate those :
        bureau_reduced = bureau[bureau['SK_ID_CURR'].isin(X.index)]
        skid_bureau_valid = bureau_reduced["SK_ID_BUREAU"].unique()
        
        #Select Bureau ID matching with SK_ID_CURR
        bb_reduced = bb[bb['SK_ID_BUREAU'].isin(skid_bureau_valid)]
        
        def bureau_and_balance(bureau_data, bb_data):
            bb = bb_data
            bureau = bureau_data
            
            bb, bb_cat = one_hot_encoder(bb)
            bureau, bureau_cat = one_hot_encoder(bureau)

            # Bureau balance: Perform aggregations and merge with bureau.csv
            bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
            for col in bb_cat:
                bb_aggregations[col] = ['mean']
            bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
            bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
            bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
            bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
            del bb, bb_agg
            gc.collect()

            # Bureau and bureau_balance numeric features
            num_aggregations = {
                'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
                'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
                'DAYS_CREDIT_UPDATE': ['mean'],
                'CREDIT_DAY_OVERDUE': ['max', 'mean'],
                'AMT_CREDIT_MAX_OVERDUE': ['mean'],
                'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
                'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
                'AMT_CREDIT_SUM_OVERDUE': ['mean'],
                'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
                'AMT_ANNUITY': ['max', 'mean'],
                'CNT_CREDIT_PROLONG': ['sum'],
                'MONTHS_BALANCE_MIN': ['min'],
                'MONTHS_BALANCE_MAX': ['max'],
                'MONTHS_BALANCE_SIZE': ['mean', 'sum']
            }
            # Bureau and bureau_balance categorical features
            cat_aggregations = {}
            for cat in bureau_cat: cat_aggregations[cat] = ['mean']
            for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

            bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
            bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
            
            
            # Bureau: Active credits - using only numerical aggregations
            if 'CREDIT_ACTIVE_Active' in bureau.columns : 
                active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
                active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
                active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
                bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
                del active, active_agg
                gc.collect()
                
            # Bureau: Closed credits - using only numerical aggregations
            if 'CREDIT_ACTIVE_Closed' in bureau.columns : 
                closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
                closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
                closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
                bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
                del closed, closed_agg, bureau

            bureau_agg = bureau_agg.replace(np.inf, np.nan)
            
            gc.collect()
            return abs(bureau_agg)
        
        return bureau_and_balance(bureau_reduced, bb_reduced)
        
    
    
    
    
    
class Previous_Applications(BaseEstimator, TransformerMixin) :
    def __init__(self, prev_data) : 
        self.prev = prev_data
        
    def fit(self, X, y=None) :
        return self
    
    def transform(self, X, y=None) : 
        prev = self.prev
        prev_reduced = prev[prev["SK_ID_CURR"].isin(X.index)]

        def previous_applications(prev_data):
            
            prev = prev_data
            prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
            # Days 365.243 values -> nan
            prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
            prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
            prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
            prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
            prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
            # Add feature: value ask / value received percentage
            prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
            # Previous applications numeric features
            num_aggregations = {
                'AMT_ANNUITY': ['min', 'max', 'mean'],
                'AMT_APPLICATION': ['min', 'max', 'mean'],
                'AMT_CREDIT': ['min', 'max', 'mean'],
                'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
                'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
                'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
                'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
                'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
                'DAYS_DECISION': ['min', 'max', 'mean'],
                'CNT_PAYMENT': ['mean', 'sum'],
            }
            # Previous applications categorical features
            cat_aggregations = {}
            for cat in cat_cols:
                cat_aggregations[cat] = ['mean']

            prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
            prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
            # Previous Applications: Approved Applications - only numerical features
            if 'NAME_CONTRACT_STATUS_Approved' in prev.columns : 
                approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
                approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
                approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
                prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
            # Previous Applications: Refused Applications - only numerical features
            if 'NAME_CONTRACT_STATUS_Refused' in prev.columns : 
                refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
                refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
                refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
                prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
                del refused, refused_agg, approved, approved_agg, prev

            prev_agg = prev_agg.replace(np.inf, np.nan)

            gc.collect()
            return prev_agg
        
        return previous_applications(prev_reduced)
    
    
    
    
    
class Pos_Cash_Balance(BaseEstimator, TransformerMixin) :
    def __init__(self, pos_data) : 
        self.pos = pos_data
        
    def fit(self, X, y=None) :
        return self
    
    def transform(self, X, y=None) : 
        pos = self.pos
        pos_reduced = pos[pos["SK_ID_CURR"].isin(X.index)]

        def pos_cash(pos_data):
            pos = pos_reduced
            pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
            # Features
            aggregations = {
                'MONTHS_BALANCE': ['max', 'mean', 'size'],
                'SK_DPD': ['max', 'mean'],
                'SK_DPD_DEF': ['max', 'mean']
            }
            for cat in cat_cols:
                aggregations[cat] = ['mean']

            pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
            pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
            # Count pos cash accounts
            pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()

            pos_agg = pos_agg.replace(np.inf, np.nan)

            del pos
            gc.collect()
            return pos_agg
    
        return abs(pos_cash(pos_reduced))
    
    
    
class Installments_Payments(BaseEstimator, TransformerMixin) :
    def __init__(self, ins_data) : 
        self.ins = ins_data
        
    def fit(self, X, y=None) :
        return self
    
    def transform(self, X, y=None) : 
        ins = self.ins
        ins_reduced = ins[ins["SK_ID_CURR"].isin(X.index)]

        def installments_payments(ins_data):
            ins = ins_data
            ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
            # Percentage and difference paid in each installment (amount paid and installment value)
            ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
            ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
            # Days past due and days before due (no negative values)
            ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
            ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
            ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
            ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
            # Features: Perform aggregations
            aggregations = {
                'NUM_INSTALMENT_VERSION': ['nunique'],
                'DPD': ['max', 'mean', 'sum'],
                'DBD': ['max', 'mean', 'sum'],
                'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
                'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
                'AMT_INSTALMENT': ['max', 'mean', 'sum'],
                'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
                'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
            }
            for cat in cat_cols:
                aggregations[cat] = ['mean']
            ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
            ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
            # Count installments accounts
            ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
            ins_agg = ins_agg.replace(np.inf, np.nan)
            del ins
            gc.collect()
            return ins_agg
        
        return installments_payments(ins_reduced)
    
    
    
    
    
class Credit_Card(BaseEstimator, TransformerMixin) :
    def __init__(self, ccb_data) : 
        self.ccb = ccb_data
        
    def fit(self, X, y=None) :
        return self
    
    def transform(self, X, y=None) : 
        ccb = self.ccb
        #print(ccb.columns)
        ccb_reduced = ccb[ccb["SK_ID_CURR"].isin(X.index)]
           
        def credit_card_balance(ccb_data):
            ccb = ccb_data
            ccb, cat_cols = one_hot_encoder(ccb, nan_as_category= True)
            # General aggregations
            ccb.drop(['SK_ID_PREV'], axis= 1, inplace = True)
            ccb_agg = ccb.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
            ccb_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in ccb_agg.columns.tolist()])
            # Count credit card lines
            ccb_agg['CC_COUNT'] = ccb.groupby('SK_ID_CURR').size()
            ccb_agg = ccb_agg.replace(np.inf, np.nan)

            del ccb
            gc.collect()
            return ccb_agg
            
        return credit_card_balance(ccb_data=ccb_reduced)
    
    
    
    
    
class OrdinalEncoding(BaseEstimator, TransformerMixin) : 
    def __init__(self) : 
        pass
    
    def fit(self, X, y=None) : 
        return self
    
    def transform(self, X, y=None) :
        #Mask of cat and cont features, to get list (idx type) : ask for index attribute
        cat_feat = (X.dtypes).where(X.dtypes=="category").dropna()
        cont_feat = (X.dtypes).where(X.dtypes!="category").dropna()
        
        X_cont = X.select_dtypes(exclude="category")

        #Subset to get category types
        X_cat = X.select_dtypes(include="category")
        X_cat = X_cat.replace(np.nan, "NOT_SPECIFIED")
        X_cat.columns = cat_feat.index
        
        #Exchange string modalities for labels, unrelated to target values
        oe = OrdinalEncoder()
        X_cat_enc = pd.DataFrame(oe.fit_transform(X_cat),
                                 columns=X_cat.columns,
                                 index=X_cat.index)
        
        X_cat_enc = X_cat_enc.astype("int")
        
        print("mon premier set", X_cat_enc.shape)
        print("mon second set", X_cont.shape)
        X_numerical = pd.concat([X_cat_enc, X_cont], axis=1)
        X_numerical.columns = list(cat_feat.index) + list(cont_feat.index)
        
        return X_numerical
    
    
    
    
class ColumnExtractor(TransformerMixin):
    def __init__(self, cols) : 
        self.cols = cols
    
    def fit(self, X, y=None) : 
        return self
    
    def transform(self, X) : 
        Xcols = X[self.cols]
        return Xcols
    
    
    
from functools import reduce

class CustomFeatureUnion(TransformerMixin, BaseEstimator) : 
    def __init__(self, transformer_list) : 
        self.transformer_list = transformer_list
        
    def fit(self, X, y=None) : 
        for (name, t) in self.transformer_list : 
            t.fit(X,y)
        return self
    
    def transform(self, X) : 
        Xts = [t.transform(X) for _, t in self.transformer_list]
        Xunion = reduce(lambda X1, X2 : pd.merge(X1,X2,left_index=True,
                                                       right_index=True,
                                                       how = "left"), #initially unspecified, thus "inner"
                                                       Xts)
        return Xunion
    
    
    
    
class MIFeatureSelector(BaseEstimator, TransformerMixin) : 
    def __init__(self) : 
        pass
    def fit(self, X, y) : 
        #Mask of cat and cont features, to get list (idx type) : ask for index attribute
        cat_feat = (X.dtypes).where(X.dtypes=="category").dropna()
        cont_feat = (X.dtypes).where(X.dtypes!="category").dropna()
        
        #Subset to get category types
        X_cat = X.select_dtypes(include="category")
        X_cat = X_cat.replace(np.nan, "NOT_SPECIFIED")
        X_cat.columns = cat_feat.index
        
        #Subset to get continuous types
        X_cont = X.select_dtypes(exclude="category")
        X_cont = X_cont.fillna(value=X_cont.median())
        X_cont.columns = cont_feat.index
        
        #Exchange string modalities for labels, unrelated to target values
        oe = OrdinalEncoder()
        X_cat_enc = pd.DataFrame(oe.fit_transform(X_cat),
                                 columns=X_cat.columns,
                                 index=X_cat.index)
        
        X_numerical = pd.concat([X_cat_enc, X_cont], axis=1)
        X_numerical.columns = list(cat_feat.index) + list(cont_feat.index)
        
        scaler = StandardScaler()
        X_std = pd.DataFrame(scaler.fit_transform(X_numerical),
                             columns = X_numerical.columns,
                             index = X_numerical.index)
        
        print(X_std.nunique())
        self.mi = pd.Series(mutual_info_classif(X_std, y, discrete_features='auto'),
                            index=X_numerical.columns).sort_values(ascending=False)
        self.best_features_ = self.mi.head(100)
        
        return self
    
    def transform(self, X, y=None) :
        selected_features = self.best_features_.index
        return X.loc[:, selected_features]
    
    
    
class LassoFeatureSelector(BaseEstimator, TransformerMixin) : 
    def __init__(self) : 
        pass
    
    def fit(self, X, y) :
        X = X.sample(50000, random_state=21)
        #Mask of cat and cont features, to get list (idx type) : ask for index attribute
        cat_feat = (X.dtypes).where(X.dtypes=="category").dropna()
        cont_feat = (X.dtypes).where(X.dtypes!="category").dropna()
        
        #Subset to get category types
        X_cat = X.select_dtypes(include="category")
        X_cat = X_cat.replace(np.nan, "NOT_SPECIFIED")
        X_cat.columns = cat_feat.index
        self.cat_feat = cat_feat.index
        print("forme de X_cat", X_cat.shape)
        
        #Subset to get continuous types
        X_cont = X.select_dtypes(exclude="category")
        X_cont = X_cont.fillna(value=X_cont.median())
        X_cont.columns = cont_feat.index

        print("forme de X_cont", X_cont.shape)

        scaler = StandardScaler()
        X_std = pd.DataFrame(scaler.fit_transform(X_cont),
                             columns = X_cont.columns,
                             index = X_cont.index).dropna()
        
        print("forme de X_std", X_std.shape)

        self.lasso_data = X_std
        
        lasso = Lasso(alpha=0.99, fit_intercept=True, normalize=False,
              precompute=False, copy_X=True, max_iter=3000,
              tol=0.001, warm_start=False, positive=False,
              random_state=None, selection='cyclic')
        
        lasso.fit(X = X_std*1000, y = y[X_std.index])
        #print(lasso.coef_)
        self.coefs = pd.Series(lasso.coef_, index=X_std.columns, dtype=float)
        self.best_features_ = abs(self.coefs).where(self.coefs != 0).dropna()
        
        print("nombre de features continue après lasso:" , self.best_features_.shape)
        self.selected_features = list(self.best_features_.index) + list(X_cat.columns)
        return self
    
    def transform(self, X, y=None) :
        X_out = X.loc[:, self.selected_features]
        X_out.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_out.columns]
        return X_out