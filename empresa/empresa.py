import pandas as pd
import pickle
import numpy as np
import json
import os

class PredictEmprestimo(object):
    def __init__(self):
        base_path = os.path.dirname(__file__)
        param_path = os.path.abspath(os.path.join(base_path, '..', 'parameter'))

        self.idade                                  = pickle.load(open(os.path.join(param_path, 'idade_scaler.pkl'), 'rb'))
        self.renda                                  = pickle.load(open(os.path.join(param_path, 'renda_scaler.pkl'), 'rb'))
        self.tempo_emprego                          = pickle.load(open(os.path.join(param_path, 'tempo_emprego_scaler.pkl'), 'rb'))
        self.valor_emprestimo                       = pickle.load(open(os.path.join(param_path, 'valor_emprestimo_scaler.pkl'), 'rb'))
        self.taxa_juros_emprestimo                  = pickle.load(open(os.path.join(param_path, 'taxa_juros_emprestimo_scaler.pkl'), 'rb'))
        self.relacao_emprestimo_renda               = pickle.load(open(os.path.join(param_path, 'relacao_emprestimo_renda_scaler.pkl'), 'rb'))
        self.historico_credito                      = pickle.load(open(os.path.join(param_path, 'historico_credito_scaler.pkl'), 'rb'))
        self.taxa_juros_ajustada_renda              = pickle.load(open(os.path.join(param_path, 'taxa_juros_ajustada_renda_scaler.pkl'), 'rb'))
        self.proporcao_emprestimo_idade             = pickle.load(open(os.path.join(param_path, 'proporcao_emprestimo_idade_scaler.pkl'), 'rb'))
        self.proporcao_emprestimo_tempo_emprego     = pickle.load(open(os.path.join(param_path, 'proporcao_emprestimo_tempo_emprego_scaler.pkl'), 'rb'))
        self.proporcao_emprestimo_historico_credito = pickle.load(open(os.path.join(param_path, 'proporcao_emprestimo_historico_credito_scaler.pkl'), 'rb'))
        self.proporcao_renda_emprestimo             = pickle.load(open(os.path.join(param_path, 'proporcao_renda_emprestimo_scaler.pkl'), 'rb'))
        self.imputer                                = pickle.load(open(os.path.join(param_path, 'imputed_knn.pkl'), 'rb'))

        self.posse_casa                             = pickle.load(open(os.path.join(param_path, 'posse_casa_label_encoder.pkl'), 'rb'))
        self.finalidade_emprestimo                  = pickle.load(open(os.path.join(param_path, 'finalidade_emprestimo_label_encoder.pkl'), 'rb'))
        self.grau_risco_emprestimo                  = pickle.load(open(os.path.join(param_path, 'grau_risco_emprestimo_label_encoder.pkl'), 'rb'))
        self.registro_inadimplencia                 = pickle.load(open(os.path.join(param_path, 'registro_inadimplencia_label_encoder.pkl'), 'rb'))


    def data_cleaning(self, df1):
        num_attributes = df1.select_dtypes(include=['int64', 'float64'])
        cat_attributes = df1.select_dtypes(include=['object', 'category'])

        if num_attributes.isnull().any().any():
            df_imputed = pd.DataFrame(self.imputer.transform(num_attributes), columns=num_attributes.columns)
            df1 = pd.concat([df_imputed, cat_attributes], axis=1)
        else:
         return df1
    
        return df1
    
    
    def feature_engineering(self, df2):
        # Taxa de Juros Ajustada à renda
        df2['taxa_juros_ajustada_renda'] = df2['taxa_juros_emprestimo'] / df2['renda']

        # Proporção do Valor do Empréstimo em Relação à Idade
        df2['proporcao_emprestimo_idade'] = df2['valor_emprestimo'] / df2['idade']

        # Proporção Valor do Empréstimo em relação ao Tempo de Emprego
        df2['tempo_emprego'] = np.where(df2['tempo_emprego'] == 0, 1, df2['tempo_emprego'])
        df2['proporcao_emprestimo_tempo_emprego'] = df2['valor_emprestimo'] / df2['tempo_emprego']

        # Proporção do Valor do Empréstimo em Relação à Histórico de Crédito
        df2['proporcao_emprestimo_historico_credito'] = df2['valor_emprestimo'] / df2['historico_credito']

        # Proporção renda valor do empréstimo
        df2['proporcao_renda_emprestimo'] = df2['renda'] / df2['valor_emprestimo']

        # Categoria Renda
        faixa_renda = [0, 3000, 7000, 50000, float('inf')]
        categorias_renda = ['baixa', 'média baixa', 'média alta', 'alta']
        df2['categoria_renda'] = pd.cut(df2['renda'], bins=faixa_renda, labels=categorias_renda)

        # Categoria Idade
        faixa_idade = [0, 30, 50, float('inf')]
        categorias_idade = ['jovem', 'adulto', 'idoso']
        df2['categoria_idade'] = pd.cut(df2['idade'], bins=faixa_idade, labels=categorias_idade) 
        
        return df2
        

    def data_preparation(self, df3):
        df3['idade'] = self.idade.transform(df3[['idade']])
        df3['renda'] = self.renda.transform(df3[['renda']])
        df3['tempo_emprego'] = self.tempo_emprego.transform(df3[['tempo_emprego']])
        df3['valor_emprestimo'] = self.valor_emprestimo.transform(df3[['valor_emprestimo']])
        df3['taxa_juros_emprestimo'] = self.taxa_juros_emprestimo.transform(df3[['taxa_juros_emprestimo']])
        df3['relacao_emprestimo_renda'] = self.relacao_emprestimo_renda.transform(df3[['relacao_emprestimo_renda']])
        df3['historico_credito'] = self.historico_credito.transform(df3[['historico_credito']])

        df3['taxa_juros_ajustada_renda'] = self.taxa_juros_ajustada_renda.transform(df3[['taxa_juros_ajustada_renda']])
        df3['proporcao_emprestimo_idade'] = self.proporcao_emprestimo_idade.transform(df3[['proporcao_emprestimo_idade']])
        df3['proporcao_emprestimo_tempo_emprego'] = self.proporcao_emprestimo_tempo_emprego.transform(df3[['proporcao_emprestimo_tempo_emprego']])
        df3['proporcao_emprestimo_historico_credito'] = self.proporcao_emprestimo_historico_credito.transform(df3[['proporcao_emprestimo_historico_credito']])
        df3['proporcao_renda_emprestimo'] = self.proporcao_renda_emprestimo.transform(df3[['proporcao_renda_emprestimo']])
        df3['grau_risco_emprestimo'] = self.grau_risco_emprestimo.transform(df3[['grau_risco_emprestimo']])

        df3['posse_casa'] = self.posse_casa.transform(df3['posse_casa'])
        df3['finalidade_emprestimo'] = self.finalidade_emprestimo.transform(df3['finalidade_emprestimo'])
        df3['registro_inadimplencia'] = self.registro_inadimplencia.transform(df3['registro_inadimplencia'])
        
        # categoria_renda
        renda_dict = {
            'baixa': 1,
            'média baixa': 2,
            'média alta': 3,
            'alta': 4
        }

        df3['categoria_renda'] = df3['categoria_renda'].map(renda_dict)
        df3['categoria_renda'] = df3['categoria_renda'].astype(int)

        # categoria_idade
        renda_dict = {
            'jovem': 1,
            'adulto': 2,
            'idoso': 3
        }

        df3['categoria_idade'] = df3['categoria_idade'].map(renda_dict)
        df3['categoria_idade'] = df3['categoria_idade'].astype(int)
        
        cols_selected_boruta= ['renda',
                                'tempo_emprego',
                                'valor_emprestimo',
                                'taxa_juros_emprestimo',
                                'relacao_emprestimo_renda',
                                'posse_casa',
                                'finalidade_emprestimo',
                                'grau_risco_emprestimo',
                                'taxa_juros_ajustada_renda',
                                'proporcao_emprestimo_idade',
                                'proporcao_emprestimo_tempo_emprego',
                                'proporcao_renda_emprestimo']
        
        return df3[cols_selected_boruta]
    
    
    def get_predictions(self, model, test_data, original_data):
        pred = model.predict(test_data)
        original_data['prediction'] = pred
        
        return original_data.to_json(orient='records')