# SPR-X-Ray-Age-Prediction-Challenge

Este repositório contém arquivos .ipynb dos meus modelos de aprendizado de máquina desenvolvido para o Desafio de Inteligência Artificial SPR-AWS. O dataset é constituído por 22451 arquivos totalizando 19.81 GB contendo arquivos do tipo png e csv.

Os códigos têm como finalidade o treinamento de rede neural usando a biblioteca Keras do TensorFlow.

A versão 1 não está disponibilizada pois foi muito parecida com a versão 2. apenas questão de ajuste de hiperparâmetros. Na versão 1 foi obtido um MAE de 11.74808 e versão 2 um MAE igual a 10.00549. Estes resultados condizem com a simplicidade do modelo (foi apenas um teste rapido para ponto de partida).
Nessas duas primeiras versões foi realizado o treinamento com a 5 folds, e 80% do dataset para treinamento / 20% para validação, o melhor fold foi selecionado e retreinado com praticamente todo o dataset destinado ao treinamento com apenas uma pequena parcela de 3% para validação apenas para observar o retreinamento para garantir que não teria overffiting. Um modelo simples de 3 camadas densas intermediárias intercaladas com camadas de batchNormalization e Dropout.

Já na versão 3 (não disponibilizado ainda pois está em meio ao treinamento), o código principal começa importando bibliotecas necessárias, como NumPy, Matplotlib, Pandas, e outras. Em seguida, define o número de núcleos a serem usados pelo TensorFlow na CPU para não usar 100% da mesma evitando danos pois como sao muitas imagens não tenho memoria suficiente na GPU do meu computador.

A ideia do modelo é simples: é utilizada a técnica de Tranfer Learning, usando como base a pré-treinada EfficientNetB2 congelando seus pesos e adicionando na sua saida camadas completamente conectadas a fim de treinamento e atualização dos pesos. O código também usa uma técnica de aprendizado de taxa de aprendizagem, Cosine Annealing, para ajustar a taxa de aprendizado ao longo do tempo e a técnica de Early Stopping que foi configurado para monitorar a perda na validação (val_loss) e ser paciente por 10 épocas (patience=10). Além disso, o modelo é salvo com os pesos que obtiveram a melhor performance na validação (restore_best_weights=True). Finalmente, o modelo é compilado e treinado com os dados de treinamento e validação fornecidos. O tempo de execução será medido no final.
obs: Em processo de obtenção de resultados.
