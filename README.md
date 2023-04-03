# SPR-X-Ray-Age-Prediction-Challenge

Este repositório contém o código do meu modelo de aprendizado de máquina desenvolvido para o Desafio de Inteligência Artificial SPR-AWS. Usei dezenas de milhares de imagens de radiografias de tórax fornecidas pela SPR para treinar meu modelo.

Para executar o código, é necessário instalar as dependências necessárias e executar o script principal. O script recebe imagens de radiografias de tórax como entrada e prevê o gênero e faixa etária do paciente.

O código tem como finalidade o treinamento de um modelo de rede neural usando a biblioteca Keras do TensorFlow. Ele começa importando bibliotecas necessárias, como NumPy, Matplotlib, Pandas, e outras. Em seguida, define o número de núcleos a serem usados pelo TensorFlow na CPU para não usar 100% da mesma evitando danos pois como sao muitas imagens não tenho memoria suficiente na GPU do meu computador para rodar o tensorflow com tantas imagens.

A ideia do modelo é simples: é utilizada a técnica de Tranfer Learning, usando como base a pré-treinada EfficientNetB2 congelando seus pesos e adicionando na sua saida camadas extritamente ligadas a fim de treinamento e atualização dos pesos. O código também usa uma técnica de aprendizado de taxa de aprendizagem, Cosine Annealing, para ajustar a taxa de aprendizado ao longo do tempo e a técnica de Early Stopping que foi configurado para monitorar a perda na validação (val_loss) e ser paciente por 10 épocas (patience=10). Além disso, o modelo é salvo com os pesos que obtiveram a melhor performance na validação (restore_best_weights=True). Finalmente, o modelo é compilado e treinado com os dados de treinamento e validação fornecidos. O tempo de execução é medido no final.
