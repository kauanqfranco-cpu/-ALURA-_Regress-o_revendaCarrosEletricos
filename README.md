<h1 align='center'>Análise de Veículos Elétricos: Previsão de Valor de Revenda e Inferência Causal</h1>

<h2>Visão Geral do Projeto</h2>
Este projeto tem como objetivo prever o valor de revenda de veículos elétricos (VEs) em dólares (USD) utilizando um pipeline de machine learning altamente parcimonioso e embasado em inferência causal. Em vez de adotar uma abordagem "caixa-preta" e jogar todas as variáveis disponíveis em um algoritmo, este projeto começa com um Grafo Acíclico Direcionado (DAG) para mapear a realidade econômica do mercado de VEs, prevenindo viés de variável omitida e isolando os verdadeiros impulsionadores da depreciação.

<h2>Objetivos</h2>
Arquitetura Causal: Mapear as relações teóricas entre ano de fabricação, capacidade da bateria, autonomia e valor de revenda para filtrar ruídos estatísticos e colisores perigosos.Higiene e Localização de Dados: Padronizar e localizar as variáveis de um conjunto de dados internacional para o português, garantindo uma integração perfeita em ambientes de negócios no Brasil.Benchmarking Algorítmico: Avaliar ensembles baseados em árvores (XGBoost, Random Forest) contra modelos baseados em margem (SVR) e modelos lineares penalizados (Elastic Net/Ridge-Lasso).Interpretabilidade de Negócios: Extrair e visualizar coeficientes matemáticos para fornecer aos stakeholders o impacto financeiro exato de características específicas do veículo.

<h2>Conjunto de Dados e Variáveis</h2>
O dataset original continha 25 colunas detalhando especificações técnicas, tipos de uso e impactos ambientais. Com base na análise causal do DAG, a matriz de features foi reduzida agressivamente para as causas fundamentais mais robustas do valor de revenda:Modelo (Model)Ano (Manufacturing Year)Autonomia(KM) (Estimated Range)Capacidade_bateria(kWh) (Battery Capacity)Alvo (Target): Valor_revenda(USD) (Resale Value)

<h2>Stack Tecnológico e Arquitetura</h2>
Python 3.x
Inferência Causal: Modelagem com DAG para seleção de features.
Engenharia de Dados: pandas, numpy (Mapeamento de colunas localizado e encoding de strings).
Machine Learning: scikit-learn, xgboostPipeline Anti-Vazamento: Pipeline, ColumnTransformer, StandardScaler, OneHotEncoder
Validação: GridSearchCV otimizado para o Erro Quadrático Médio Negativo (NMSE).
Visualização: matplotlib, seaborn (Mapas de calor contínuos personalizados e gráficos de barras de impacto das features).

<h2>Metodologia e Pipeline</h2>
Análise Exploratória de Dados (EDA): Mapeamento de correlação de postos de Spearman (para capturar depreciação monotônica e não-linear) e F-scores de ANOVA para variância categórica.
Quarentena de Dados: Aplicação rigorosa do train_test_split antes de qualquer transformação matemática para evitar o vazamento de dados (data leakage).
Pré-processamento: Variáveis contínuas padronizadas e variáveis categóricas codificadas com one-hot encoding inteiramente dentro de um ColumnTransformer do Scikit-Learn.
Treinamento de Modelos: Otimização de hiperparâmetros em quatro arquiteturas distintas via validação cruzada (cross-validation) com 5 folds.
Serialização: O pipeline escolhido (Scaler + Regressor) foi serializado usando pickle (modeloFinalRidgeLasso.pkl) para implantação imediata em produção.

<h2>Resultados e Insights de Negócio</h2>
O modelo Ridge-Lasso (Elastic Net) sagrou-se campeão, superando ensembles mais pesados com excelentes métricas de previsão financeira:$R^2$: ~0.904Variância Explicada (EV): ~0.904MAE: ~$1.533RMSE: ~$1.773Descoberta Principal: O alinhamento quase idêntico entre o $R^2$ e a Variância Explicada prova que o modelo possui viés sistemático quase zero (ele superestima a previsão com a exata mesma frequência e magnitude com que a subestima).Impacto das Variáveis: A extração dos coeficientes revelou que a Capacidade da Bateria e o Ano de Fabricação são os principais impulsionadores positivos do valor de revenda de um VE, enquanto a penalidade Lasso reduziu com sucesso as features inúteis para exatamente zero, criando um algoritmo altamente parcimonioso e pronto para produção.
