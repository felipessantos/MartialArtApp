
# coding: utf-8

# # Treinando RandomForest

# In[1]:

import pandas as pd
from datetime import datetime, timedelta, date
import numpy as np
import csv
import unidecode 
import pandas.core.algorithms as algos
from scipy.stats import kendalltau   
from funcoes_uteis import *
from dateutil.relativedelta import relativedelta

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder


# In[2]:

def diff_month(d1, d2):
    return (d1.year - d2.year)*12 + d1.month - d2.month

def periodicidade(x):
    if x == 'Trienal':
        return 36    
    elif x == 'Anual':
        return 12
    elif x == 'Semestral':
        return 6
    elif x == 'Trimestral':
        return 3
    else: 
        return 1

def marca_base(Perc, x):
    if x >= Perc[(len(Perc)-1)]:
        return len(Perc) +1
    else:
        for i in range(len(Perc)):
            if x < Perc[i]:
                return i + 1


def cria_curva(percentiles, variavel):
    Perc = list()
    for i in range(len(percentiles)):
        Perc.append(np.percentile(variavel, percentiles[i]))
    return Perc    
    
def aux_nome_data(data):
    y = str(data.year)
    m = str(data.month)
    d = str(data.day)
    if data.day < 9:
        d = '0'+str(data.day)
    if data.month < 9:
        m = '0'+str(data.month)
    return y+m+d


# In[3]:

def ArrumaBase_Consumo(fim_janela_feature, janela_booking, df_base):
    aux_janela_feature = fim_janela_feature+ relativedelta(months=-12)
    inicio_janela_booking = fim_janela_feature
    fim_janela_booking = fim_janela_feature+ relativedelta(months=janela_booking)
    df_treino = df_base[(df_base.Instalacao_AnoMes < fim_janela_feature)].copy()
    grade = pd.date_range(start=aux_janela_feature, end=df_treino.Instalacao_AnoMes.max(), freq='MS', normalize=True)
    agg_dict = {'nr_PrecoMensal' : 'sum'}
    df_historico_valor = df_treino[(df_treino.nr_PrecoMensal> 0) & (df_treino.Instalacao_AnoMes >= aux_janela_feature)].groupby(['cd_ChaveCliente', 'Instalacao_AnoMes']).agg(agg_dict)
    novo_indice = [(x, y) for x in df_historico_valor.index.levels[0] for y in grade]
    novo_indice = pd.MultiIndex.from_tuples(novo_indice, names=df_historico_valor.index.names)
    df_historico_valor = df_historico_valor.reindex(novo_indice, fill_value=0)
    df_historico_valor = df_historico_valor.unstack(level=-1)
    df_historico_valor.columns = df_historico_valor.columns.droplevel()
    colunas = ['vlr_m-1','vlr_m-2','vlr_m-3','vlr_m-4','vlr_m-5','vlr_m-6','vlr_m-7','vlr_m-8','vlr_m-9','vlr_m-10','vlr_m-11','vlr_m-12'] 
    df_historico_valor.columns = colunas
    df_historico_valor['vlr_trim-1'] = [v1+v2+v3 for v1,v2,v3 in zip
                                        (df_historico_valor['vlr_m-1'], 
                                         df_historico_valor['vlr_m-2'],  
                                         df_historico_valor['vlr_m-3'])]
    df_historico_valor['vlr_trim-2'] = [v1+v2+v3 for v1,v2,v3 in zip
                                        (df_historico_valor['vlr_m-4'], 
                                         df_historico_valor['vlr_m-5'],  
                                         df_historico_valor['vlr_m-6'])]
    df_historico_valor['vlr_trim-3'] = [v1+v2+v3 for v1,v2,v3 in zip
                                        (df_historico_valor['vlr_m-7'], 
                                         df_historico_valor['vlr_m-8'],  
                                         df_historico_valor['vlr_m-9'])]
    df_historico_valor['vlr_trim-4'] = [v1+v2+v3 for v1,v2,v3 in zip
                                        (df_historico_valor['vlr_m-10'], 
                                         df_historico_valor['vlr_m-11'],  
                                         df_historico_valor['vlr_m-12'])]
    df_historico_valor['vlr_ano'] = [v1+v2+v3+v4 for v1,v2,v3,v4 in zip
                                        (df_historico_valor['vlr_trim-1'], 
                                         df_historico_valor['vlr_trim-2'],  
                                         df_historico_valor['vlr_trim-3'],
                                         df_historico_valor['vlr_trim-4'])]
    agg_dict = {'Provisioning' : 'count'}
    df_historico_qtd = df_treino[df_treino.Instalacao_AnoMes >= aux_janela_feature].groupby(['cd_ChaveCliente', 'Instalacao_AnoMes']).agg(agg_dict)
    novo_indice = [(x, y) for x in df_historico_qtd.index.levels[0] for y in grade]
    novo_indice = pd.MultiIndex.from_tuples(novo_indice, names=df_historico_qtd.index.names)
    df_historico_qtd = df_historico_qtd.reindex(novo_indice, fill_value=0)
    df_historico_qtd = df_historico_qtd.unstack(level=-1)
    df_historico_qtd.columns = df_historico_qtd.columns.droplevel()
    colunas_ts = [c for c in df_historico_qtd.columns if isinstance(c, pd.tslib.Timestamp)]
    colunas = ['qtd_m-1','qtd_m-2','qtd_m-3','qtd_m-4','qtd_m-5','qtd_m-6',
               'qtd_m-7','qtd_m-8','qtd_m-9','qtd_m-10','qtd_m-11','qtd_m-12'] 
    df_historico_qtd.columns = colunas
    df_historico_qtd['qtd_trim-1'] = [q1+q2+q3 for q1,q2,q3 in zip
                                      (df_historico_qtd['qtd_m-1'],
                                       df_historico_qtd['qtd_m-2'],
                                       df_historico_qtd['qtd_m-3'])]
    df_historico_qtd['qtd_trim-2'] = [q1+q2+q3 for q1,q2,q3 in zip
                                      (df_historico_qtd['qtd_m-4'], 
                                       df_historico_qtd['qtd_m-5'],  
                                       df_historico_qtd['qtd_m-6'])]
    df_historico_qtd['qtd_trim-3'] = [q1+q2+q3 for q1,q2,q3 in zip
                                      (df_historico_qtd['qtd_m-7'], 
                                       df_historico_qtd['qtd_m-8'],  
                                       df_historico_qtd['qtd_m-9'])]
    df_historico_qtd['qtd_trim-4'] = [q1+q2+q3 for q1,q2,q3 in zip
                                      (df_historico_qtd['qtd_m-10'], 
                                       df_historico_qtd['qtd_m-11'],  
                                       df_historico_qtd['qtd_m-12'])]
    df_historico_qtd['qtd_ano'] = [v1+v2+v3+v4 for v1,v2,v3,v4 in zip
                                   (df_historico_qtd['qtd_trim-1'], 
                                    df_historico_qtd['qtd_trim-2'],  
                                    df_historico_qtd['qtd_trim-3'],
                                    df_historico_qtd['qtd_trim-4'])]
    df_historico = pd.concat([df_historico_valor, df_historico_qtd], axis=1)
    df_treino['Periodicidade_Meses'] = [periodicidade(x) for x in df_treino.ds_Periodicidade]
    df_treino['idade_prov'] = [diff_month(fim_janela_feature, Inst) for Inst in df_treino.Instalacao_AnoMes]
    df_treino['idade_cli'] = [diff_month(fim_janela_feature, Inst) for Inst in df_treino.Primeiro_Servico_LW_AnoMes]
    df_treino['quantidade_renovacoes_prov'] = [int(id_prov/peri_mes) for 
                                               id_prov, peri_mes in zip (df_treino.idade_prov, df_treino.Periodicidade_Meses)]
    df_treino['Qtd_meses_P_renovacoes'] = [peri_mes-(idade_prov-qtd_renov*peri_mes) for
                                           peri_mes,idade_prov,qtd_renov in 
                                           zip(df_treino.Periodicidade_Meses, 
                                               df_treino.idade_prov, 
                                               df_treino.quantidade_renovacoes_prov)]
    df_treino = df_treino[~df_treino.Servico.isnull()].copy()
    lista = list(df_treino.Servico.unique())
    lista = ['Servico_' + str(i) for i in lista]
    dict_lista = {str(i): 'sum' for i in lista}
    ohe = ['Servico']
    colunas = ['cd_ChaveCliente', 'Servico']
    df_ohe_Servico = pd.get_dummies(df_treino[colunas], columns = ohe,)
    df_ohe_Servico = df_ohe_Servico.groupby('cd_ChaveCliente').agg(dict_lista)
    
    ###################################################################################
    df_treino = df_treino[~df_treino.ds_Periodicidade.isnull()].copy()
    lista = list(df_treino.ds_Periodicidade.unique())
    lista = ['ds_Periodicidade_' + str(i) for i in lista]
    dict_lista = {str(i): 'sum' for i in lista}
    ohe = ['ds_Periodicidade']
    colunas = ['cd_ChaveCliente', 'ds_Periodicidade']
    df_ohe_ds_Periodicidade = pd.get_dummies(df_treino[colunas], columns = ohe,)
    df_ohe_ds_Periodicidade = df_ohe_ds_Periodicidade.groupby('cd_ChaveCliente').agg(dict_lista)
    df_ohe_ds_Periodicidade.drop(df_ohe_ds_Periodicidade.columns[[0]], axis=1, inplace= True)
    
    ###################################################################################
    df_treino = df_treino[~df_treino.Perfil.isnull()].copy()
    lista = list(df_treino.Perfil.unique())
    lista = ['Perfil_' + str(i) for i in lista]
    dict_lista = {str(i): 'sum' for i in lista}
    ohe = ['Perfil']
    colunas = ['cd_ChaveCliente', 'Perfil']
    df_ohe_Perfil = pd.get_dummies(df_treino[colunas], columns = ohe,)
    df_ohe_Perfil = df_ohe_Perfil.groupby('cd_ChaveCliente').agg(dict_lista)
    df_ohe_Perfil.drop(df_ohe_Perfil.columns[[0]], axis=1, inplace= True)
    
    ###################################################################################
    df_treino = df_treino[~df_treino.classificacao.isnull()].copy()
    lista = list(df_treino.classificacao.unique())
    lista = ['classificacao_' + str(i) for i in lista]
    dict_lista = {str(i): 'sum' for i in lista}
    ohe = ['classificacao']
    colunas = ['cd_ChaveCliente', 'classificacao']
    df_ohe_classificacao = pd.get_dummies(df_treino[colunas], columns = ohe,)
    df_ohe_classificacao = df_ohe_classificacao.groupby('cd_ChaveCliente').agg(dict_lista)
    df_ohe_classificacao.drop(df_ohe_classificacao.columns[[0]], axis=1, inplace= True)
    
    ###################################################################################
    df_treino = df_treino[~df_treino.Status.isnull()].copy()
    lista = list(df_treino.Status.unique())
    lista = ['Status_' + str(i) for i in lista]
    dict_lista = {str(i): 'sum' for i in lista}
    ohe = ['Status']
    colunas = ['cd_ChaveCliente', 'Status']
    df_ohe_Status = pd.get_dummies(df_treino[colunas], columns = ohe,)
    df_ohe_Status = df_ohe_Status.groupby('cd_ChaveCliente').agg(dict_lista)
    df_ohe_Status.drop(df_ohe_Status.columns[[0]], axis=1, inplace= True)
    
    ###################################################################################
    df_treino['idade_prov_sum'] = df_treino.idade_prov
    df_treino['idade_cli_sum'] = df_treino.idade_cli
    df_treino['quantidade_renovacoes_prov_sum'] = df_treino.quantidade_renovacoes_prov
    df_treino['Qtd_meses_P_renovacoes_sum'] = df_treino.Qtd_meses_P_renovacoes
    dict_lista = {'Provisioning' : 'count',
                  'nr_PrecoMensal' : 'sum',
                  'fl_ServicoPai' : 'sum',
                  'fl_Dev' : 'max',
                  'Data_Desativacao_flag' : 'sum',
                  'fl_GerenteConta' : 'max',
                  'idade_prov' : 'mean',
                  'idade_cli' : 'mean',
                  'quantidade_renovacoes_prov' : 'mean',
                  'Qtd_meses_P_renovacoes' : 'mean',
                  'idade_prov_sum' : 'sum',
                  'idade_cli_sum' : 'sum',
                  'quantidade_renovacoes_prov_sum' : 'sum',
                  'Qtd_meses_P_renovacoes_sum' : 'sum',
                  'FlagChurn' : 'sum'}
    df_treino = df_treino.groupby('cd_ChaveCliente').agg(dict_lista)
    
    ###################################################################################
    df_treino = pd.concat([df_treino, df_ohe_Servico, 
                           df_ohe_ds_Periodicidade, df_ohe_Perfil, 
                           df_ohe_classificacao, df_ohe_Status], axis=1)
    colunas = ['cd_ChaveCliente']
    df_booking = df_base[colunas][(df_base.Instalacao_AnoMes >= inicio_janela_booking) & 
                                  (df_base.Instalacao_AnoMes < fim_janela_booking)].copy()
    df_booking['booking'] = 1
    dict_lista = {'booking' : 'sum'}
    df_booking = df_booking.groupby('cd_ChaveCliente').agg(dict_lista)
    df_treino = pd.concat([df_treino, df_booking], axis=1, join_axes=[df_treino.index])
    df_treino.booking.fillna(0, inplace= True)
    df_treino['booking'] = [0 if b == 0 else 1 for b in df_treino.booking]
    df_treino = pd.concat([df_treino, df_historico], axis=1)
    df_treino.reset_index(inplace= True)
    df_treino.rename(columns= {'index': 'cd_ChaveCliente'}, inplace= True)
    df_treino.fillna(0,inplace= True)
    return df_treino   


# In[4]:

def CriaRandomForest_Consumo(df_treino):
    X = df_treino.sample(frac = 1).copy()
    y = X.booking.values
    Colunas_Modelo = X.reset_index(drop=True).drop(['booking'], axis = 1).columns
    X = X.reset_index(drop=True).drop(['booking'], axis = 1).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=3)
    auc_list = []
    k = 1
    for train, valid in skf.split(X, y):
#        print('Fold #', k)
#        print("train indices: %s\nvalidation indices %s" % (train, valid))
        clf = RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=5)
        clf.fit(X[train], y[train])
        y_pred = clf.predict_proba(X[valid])
        auc = roc_auc_score(y[valid], y_pred[:,1])
        auc_list.append(auc)
#        print('AUC on fold #', k, ':', auc, '\n')
        k += 1
#    print('Average AUC on', k-1, 'folds:', np.mean(auc_list))
    return clf


# In[5]:

def CriaCluster_Consumo(clf, df_treino):
    df_treino.reset_index(inplace= True)
    X = df_treino[Colunas_Modelo].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y_pred = clf.predict_proba(X)
    y_pred = pd.DataFrame(data=y_pred[:,1])
    y_pred.rename(columns= {0: 'Prob_Consumo'}, inplace= True)
    df_treino = pd.concat([df_treino, y_pred], axis=1)
    percentiles = list()
    inicio = 0
    fim = 100
    salto = 0.25
    aux = round((fim - inicio)/salto)
    for i in range(aux):
        percentiles.append((inicio +i*salto))
    Perc = cria_curva(percentiles, df_treino.Prob_Consumo)
    df_treino['Prob_Consumo_Grupo'] = [marca_base(Perc, x) for x in df_treino.Prob_Consumo]
    dict_lista_aux = {'Provisioning' : 'count',
                      'Prob_Consumo' : 'min',
                      'booking' : 'mean'}
    RESUMO = df_treino.groupby('Prob_Consumo_Grupo').agg(dict_lista_aux)
    RESUMO.sort_values(['Prob_Consumo'], ascending= 0 ,inplace=True)
    RESUMO.Prob_Churn = round(RESUMO.Prob_Consumo, ndigits = 2)
    RESUMO['booking_aux'] = [p*q for p, q in zip (RESUMO.booking, RESUMO.Provisioning)]
    RESUMO['acumulado'] = RESUMO.Provisioning.cumsum()
    RESUMO['Prob_acumulado'] = RESUMO.booking_aux.cumsum()
    RESUMO['booking_acumulado'] = [p/q for p, q in zip (RESUMO.Prob_acumulado, RESUMO.acumulado)]
    RESUMO['Prob_Consumo_aux'] = [p*q for p, q in zip (RESUMO.Prob_Consumo, RESUMO.Provisioning)]
    RESUMO['acumulado'] = RESUMO.Provisioning.cumsum()
    RESUMO['Prob_acumulado'] = RESUMO.Prob_Consumo_aux.cumsum()
    RESUMO['Prob_Consumo_acumulado'] = [p/q for p, q in zip (RESUMO.Prob_acumulado, RESUMO.acumulado)]
    RESUMO['Provisioning_perc'] = RESUMO.Provisioning/RESUMO.Provisioning.sum()
    RESUMO.reset_index(inplace= True)
    lista = RESUMO.booking_acumulado
    Prob_aux1 = RESUMO.loc[0].booking_acumulado
    curva_indice = []
    for i in range(len(lista)):
        Prob_aux2 = lista[i]
        if len(curva_indice) < 9 and Prob_aux1/Prob_aux2 >= 1.25:
            Prob_aux1 = Prob_aux2 
            curva_indice.append(i)           
    Perc = []
    for i in range(len(curva_indice)):
        Perc.append(RESUMO.booking_acumulado.loc[curva_indice[i]])
    Perc = Perc[::-1]
    RESUMO['Cluster'] = [marca_base(Perc, x) for x in RESUMO.booking_acumulado]
    RESUMO.reset_index(inplace= True)
    lista = [RESUMO[RESUMO.Cluster == 1].Prob_Consumo_Grupo.max(),
             RESUMO[RESUMO.Cluster == 2].Prob_Consumo_Grupo.max(),
             RESUMO[RESUMO.Cluster == 3].Prob_Consumo_Grupo.max(),
             RESUMO[RESUMO.Cluster == 4].Prob_Consumo_Grupo.max(),
             RESUMO[RESUMO.Cluster == 5].Prob_Consumo_Grupo.max(),
             RESUMO[RESUMO.Cluster == 6].Prob_Consumo_Grupo.max(),
             RESUMO[RESUMO.Cluster == 7].Prob_Consumo_Grupo.max(),
             RESUMO[RESUMO.Cluster == 8].Prob_Consumo_Grupo.max(),
             RESUMO[RESUMO.Cluster == 9].Prob_Consumo_Grupo.max()]
    df_treino['Cluster'] = [marca_base(lista, x) for x in df_treino.Prob_Consumo_Grupo]
    return df_treino


# In[30]:

def ComparaCluster_DadoCli(df_treino, data, texto):
    fim_janela_feature = data
    aux_Cluster = pd.DataFrame(data=df_treino.Cluster.value_counts()) 
    aux_Cluster.sort_index(inplace= True)
    aux_Cluster['Perc'] = aux_Cluster.Cluster/aux_Cluster.Cluster.sum()
    dict_lista = {'Prob_Consumo' : 'mean',
                  'Provisioning' : 'count',
                  'nr_PrecoMensal' : 'mean',
                  'fl_ServicoPai' : 'mean',
                  'fl_Dev' : 'mean',
                  'fl_GerenteConta' : 'mean',
                  'booking' : 'mean',
                  'idade_prov' : 'mean',
                  'idade_cli' : 'mean',
                  'quantidade_renovacoes_prov' : 'mean',
                  'Qtd_meses_P_renovacoes' : 'mean',
                  'FlagChurn' : 'mean'}
    RESUMO = df_treino.groupby('Cluster').agg(dict_lista)
    colunas = RESUMO.columns
    scaler = StandardScaler()
    X = scaler.fit_transform(RESUMO)
    X = pd.DataFrame(data=X)
    X.columns = colunas
    X['Cluster'] =X.reset_index().index + 1
    X.set_index(['Cluster'], inplace= True)
    NomeCSV = 'RelarotioClusterConsumo'+texto+aux_nome_data(fim_janela_feature)+'.csv'
    RESUMO.to_csv(NomeCSV)
    return RESUMO


def ComparaCluster_DadoProdutos(df_treino, data, texto):
    fim_janela_feature = data
    colunas = list()
    for i in df_treino.columns: 
        if i[:8] == 'Servico_':
            colunas.append(i)
    dict_colunas = {str(i): 'mean' for i in colunas}
    RESUMO = df_treino.groupby('Cluster').agg(dict_colunas)
    RESUMO = RESUMO.T
    RESUMO.sort(10, ascending= 0, inplace= True)
    RESUMO = pd.DataFrame(data=RESUMO)
    NomeCSV = 'RelarotioServicoClusterConsumo'+texto+aux_nome_data(fim_janela_feature)+'.csv'
    RESUMO.to_csv(NomeCSV)    
    return RESUMO


# In[7]:

def AplicaModelo_Base(clf, df_treino, df_atual):
    df_atual.reset_index(inplace= True)
    X = df_atual[Colunas_Modelo].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y_pred = clf.predict_proba(X)
    y_pred = pd.DataFrame(data=y_pred[:,1])
    y_pred.rename(columns= {0: 'Prob_Consumo'}, inplace= True)
    df_atual = pd.concat([df_atual, y_pred], axis=1)
    aux_Cluster = pd.DataFrame(data=df_treino.Cluster.value_counts()) 
    aux_Cluster.sort_index(inplace= True)
    aux_Cluster['Perc'] = aux_Cluster.Cluster/aux_Cluster.Cluster.sum()
    df_atual.sort_values(['Prob_Consumo'], ascending= 1 ,inplace=True)
    df_atual.drop('index', axis= 1, inplace= True)
    df_atual.reset_index(inplace= True, drop= True)
    df_atual.reset_index(inplace= True)
    df_atual.rename(columns= {'index': 'aux_cluster'}, inplace= True)
    lista = [round(aux_Cluster[aux_Cluster.index == 1].Perc.max()*df_atual.aux_cluster.max(),0),
             round(aux_Cluster[aux_Cluster.index == 2].Perc.max()*df_atual.aux_cluster.max(),0),
             round(aux_Cluster[aux_Cluster.index == 3].Perc.max()*df_atual.aux_cluster.max(),0),
             round(aux_Cluster[aux_Cluster.index == 4].Perc.max()*df_atual.aux_cluster.max(),0),
             round(aux_Cluster[aux_Cluster.index == 5].Perc.max()*df_atual.aux_cluster.max(),0),
             round(aux_Cluster[aux_Cluster.index == 6].Perc.max()*df_atual.aux_cluster.max(),0),
             round(aux_Cluster[aux_Cluster.index == 7].Perc.max()*df_atual.aux_cluster.max(),0),
             round(aux_Cluster[aux_Cluster.index == 8].Perc.max()*df_atual.aux_cluster.max(),0),
             round(aux_Cluster[aux_Cluster.index == 9].Perc.max()*df_atual.aux_cluster.max(),0)]
    lista[1] = lista[1] + lista[0]
    lista[2] = lista[2] + lista[1]
    lista[3] = lista[3] + lista[2]
    lista[4] = lista[4] + lista[3]
    lista[5] = lista[5] + lista[4]
    lista[6] = lista[6] + lista[5]
    lista[7] = lista[7] + lista[6]
    lista[8] = lista[8] + lista[7]
    df_atual['Cluster'] = [marca_base(lista, x) for x in df_atual.aux_cluster]
    return df_atual


# In[8]:

def BaseInicial(df_base):
    df_base['Status'] = ['ativo' if s in ['Ativo', 'Atendido', 'Em ativação',
                                          'Aguardando ativação'] else 'inativo'
                         for s in df_base.Status]
    df_base['fl_Dev'] = df_base['fl_Dev'].astype(float,)
    df_base = df_base[(df_base.nr_PrecoMensal > '0,00')].copy()
    df_base.sort_values(['Instalacao'], ascending= 1 ,inplace=True)
    df_base.drop_duplicates(['Provisioning'], keep='last', inplace= True)
    col_datas = ['Data_Desativacao', 'Data_Fim', 'Instalacao', 'dt_Reativacao', 'Primeiro_Servico_LW']
    converte_datetime(df_base, col_datas)
    df_base['Data_Fim_flag'] = [1 if d_fim != datetime(1900,1,1) else 0 for d_fim in df_base.Data_Fim]
    df_base['Data_Desativacao_flag'] = [1 if f_des != datetime(1900,1,1) else 0 for f_des in df_base.Data_Desativacao]
    df_base['Data_Desativacao_flag'] = [1 if f_des != datetime(1900,1,1) else 0 for f_des in df_base.Data_Desativacao]
    df_base['Data_churn'] = df_base[['Data_Fim', 'Data_Desativacao']].min(axis = 1).astype('datetime64[ns]')
    df_base['Data_churn'] = [d_fim if ((d_des < d_rea < d_fim) & f_fim & f_des) else d_chu
                                  for d_des, d_rea, d_fim, f_fim, f_des, d_chu in 
                                  zip(df_base.Data_Desativacao, df_base.dt_Reativacao, 
                                      df_base.Data_Fim, df_base.Data_Fim_flag, 
                                      df_base.Data_Desativacao_flag, df_base.Data_churn)]
    df_base['Data_churn_flag'] = [1 if d > datetime(1900, 1, 1) else 0 for d in df_base.Data_churn]
    #######################################
    df_base = df_base[df_base.Primeiro_Servico_LW >= datetime(1990,1,1)].copy()  #
    #######################################
    df_base['Primeiro_Servico_LW_AnoMes'] = [datetime(d.year, d.month, 1) for d in df_base.Primeiro_Servico_LW]
    df_base['Instalacao_AnoMes'] = [datetime(d.year, d.month, 1) for d in df_base.Instalacao]
    df_base['Data_churn_AnoMes'] = [datetime(d.year, d.month, 1) for d in df_base.Data_churn]
    df_base['fl_ServicoPai'] = [1 if s in ['SIM', 'Sim', 'sim'] else 0 for s in df_base.fl_ServicoPai]
    df_base['fl_GerenteConta'] = [0 if s== 1 else 1 for s in df_base.id_GerenteConta]
    df_base['nr_PrecoMensal'] = [x.replace(',', '.') for x in df_base.nr_PrecoMensal]
    df_base['nr_PrecoMensal'] = df_base.nr_PrecoMensal.astype(float)
    df_base['MesesParaChurn'] = [diff_month(ch, ins) if ch > datetime(1900, 1, 1) else 0
                               for ch, ins in  zip(df_base.Data_churn, df_base.Instalacao)]
    df_base['FlagChurn'] = [1 if d!= 0 else 0 for d in df_base.MesesParaChurn]
    return df_base


# In[9]:

#Churn_Consumo_Recomendacao_18.08.2017
df_base = pd.read_csv('./Churn_Consumo_Recomendacao_dd.mm.yyyy.csv'
                      , error_bad_lines = False
                      , sep=';'
                      , dtype= {7: str}
                      , encoding='latin-1'
                      , header = 0)


# In[10]:

df_base = BaseInicial(df_base)


# In[11]:

fim_janela_feature = df_base.Instalacao_AnoMes.max()+ relativedelta(months=-3)
janela_booking = 3


# In[12]:

df_treino = ArrumaBase_Consumo(fim_janela_feature, janela_booking, df_base)


# In[13]:

Base_Treino = df_treino.drop('cd_ChaveCliente', axis=1).copy()
clf = CriaRandomForest_Consumo(Base_Treino)
Colunas_Modelo = Base_Treino.drop('booking', axis=1).columns
del Base_Treino


# In[14]:

df_treino =  CriaCluster_Consumo(clf, df_treino)


# In[15]:

ComparaCluster_DadoCli =  ComparaCluster_DadoCli(df_treino, df_base.Instalacao.max(), 'BaseTreino')


# In[22]:

ComparaCluster_DadoProdutos = ComparaCluster_DadoProdutos(df_treino, df_base.Instalacao.max(), 'BaseTreino')


# # Aplicando na Base

# ### Atual

# In[24]:

fim_janela_feature = df_base.Instalacao.max()


# In[25]:

df_atual = ArrumaBase_Consumo(fim_janela_feature, janela_booking, df_base)


# In[26]:

df_atual = AplicaModelo_Base(clf, df_treino, df_atual)


# In[27]:

ComparaCluster_DadoCli =  ComparaCluster_DadoCli(df_atual, df_base.Instalacao.max(), 'BaseAtiva')


# In[31]:

ComparaCluster_DadoProdutos = ComparaCluster_DadoProdutos(df_treino, df_base.Instalacao.max(), 'BaseAtiva')


# In[33]:

NomeCSV = 'ProvsProbConsumoCluster'+aux_nome_data(fim_janela_feature)+'.csv'
colunas_interessantes = ['cd_ChaveCliente', 'Prob_Consumo', 'Cluster']
df_atual[colunas_interessantes].to_csv(NomeCSV)    


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



