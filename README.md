# RPG_MLP-Project
Repositório referente ao trabalho da disciplina de Redes Neurais e Algoritmos Genéticos ministrada pelo professor Daniel Cassar.

**O objetivo do notebook codado em Python é o seguinte:**
- Identifique e otimize os hiperparâmetros de uma rede neural do tipo MLP para resolver um problema de regressão de interesse científico. Espera-se o uso de boas práticas em ciências de dados assim como já apresentado em disciplinas anteriores. Teste ao menos 100 diferentes arquiteturas de rede durante sua otimização de hiperparâmetros. Os dados para treinar o modelo devem ser dados tabulados de interesse científico, não podem ser séries temporais e precisam ser aprovados pelo professor.

Para realizar essa tarefa utilizamos o Jupyter Lab, Redes Neurais PyTorch e um dataset focado na composição do concreto. O trabalho final está em um documento chamado **TrabalhoFinal(HPC).ipynb**, foi rodado no HPC da Ilum Escola de Ciência, usando 10 núcleos de processamento e rodou em menos de 1 hora. A versão do Pandas utilizada foi a 1.5.1, necessária para evitar um erro na criação do dataset final com as arquiteturas de rede.

## Dataset

Como citado acima, cada coluna do dataset **"Civil Engineering: Cement Manufacturing Dataset"** é um dado que contém a composição do concreto, tempo de preparo e força suportada.

O significado das colunas é:

- **Cimento:** É o aglomerante principal do concreto, responsável por unir os materiais. Medido em quilogramas por metro cúbico de mistura.
- **Escória de alto-forno:** Um subproduto da indústria siderúrgica que pode ser usado como adição ao concreto para melhorar suas propriedades. Medido em quilogramas por metro cúbico de mistura.
- **Cinzas volantes:** Resíduo da queima de carvão em usinas termelétricas, utilizado no concreto para melhorar sua durabilidade. Medido em quilogramas por metro cúbico de mistura.
- **Água:** Essencial para a hidratação do cimento e a mistura dos materiais. Medida em quilogramas por metro cúbico de mistura.
- **Superplastificante:** Aditivo que melhora a trabalhabilidade do concreto sem comprometer sua resistência. Medido em quilogramas por metro cúbico de mistura.
- **Agregado graúdo:** Pedras ou britas que compõem o concreto, fornecendo resistência mecânica. Medido em quilogramas por metro cúbico de mistura.
- **Agregado miúdo:** Areia ou pó de pedra que preenche os vazios entre os agregados graúdos, melhorando a trabalhabilidade. Medido em quilogramas por metro cúbico de mistura.
- **Idade:** Tempo em dias desde a mistura do concreto. Varia de 1 a 365 dias.
- **Resistência à compressão do concreto:** Medida em Megapascal (MPa), indica a capacidade do concreto de suportar cargas de compressão.

### Coleta e Tratamento de Dados

O dataset utilizado estava em boas condições, sem valores ausentes (NaN) e já tabulado corretamente. A única etapa necessária de pré-processamento foi a separação dos dados por ponto e vírgula, seguida da normalização das características.

### Metodologia

O código utiliza uma rede neural do tipo Multi-Layer Perceptron (MLP) implementada em PyTorch. A rede é treinada e validada usando K-Fold Cross-Validation para garantir a robustez dos resultados. Várias combinações de hiperparâmetros são testadas para encontrar a melhor configuração possível.

### Estrutura do código:

1. **Importação das bibliotecas:**
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    import pandas as pd
    import numpy as np
    from itertools import product
    ```

2. **Carregamento e preparação dos dados:**
    - Os dados são carregados do arquivo CSV e normalizados.
    - As colunas de atributos e a coluna alvo são separadas.
    - Os dados são convertidos para tensores PyTorch.
    ```python
    data = pd.read_csv('concrete.csv')

    # Separar atributos (X) e target (y)
    X = data.drop(columns=["cement", "slag", "ash", "water", "superplastic", "coarseagg", "fineagg", "age"])
    y = data['strength']

    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Converter para tensores PyTorch
    X = torch.tensor(X_scaled, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32)
    ```

3. **Definição dos hiperparâmetros:**
    ```python
    DIC_ATIVACOES = {
        'ReLU' : nn.ReLU(),
        'Tanh' : nn.Tanh(),
        'Sigmoid': nn.Sigmoid(),
        'Softmax': nn.Softmax(),
        'GELU': nn.GELU()
    }

    random_state = 100                    
    lrs = [0.01, 0.02, 0.05, 0.1, 0.15]   
    epochs = [100, 150, 200]              
    hidden_sizes = [16, 32, 64, 128, 256] 
    dropout_prob = 0.5                    
    k_folds = 5                           
    ```

4. **Definição da estrutura da rede MLP:**
    ```python
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_prob=0.5):
            super(MLP, self).__init__()
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(input_size, hidden_size))
            for _ in range(num_layers):
                self.layers.append(nn.Linear(hidden_size, hidden_size))
                self.layers.append(nn.Dropout(dropout_prob))
            self.layers.append(nn.Linear(hidden_size, output_size))
            self.activation = nn.ReLU()

        def forward(self, x):
            for layer in self.layers[:-1]:
                x = self.activation(layer(x))
            x = self.layers[-1](x)
            return x
    ```

5. **Treinamento e avaliação:**
    ```python
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    hyperparameters_combinations = product(lrs, epochs, hidden_sizes, range(1, 5), DIC_ATIVACOES.items())

    results_df = pd.DataFrame(columns=['Folds number',
                                       'Fold', 
                                       'Random State',
                                       'Dropout probability', 
                                       'Activation function', 
                                       'Learning Rate', 
                                       'Epochs', 
                                       'Layers number', 
                                       'Hidden Size', 
                                       'RMSE'])

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        for lr, num_epochs, hidden_size, num_layers, (ativacao_name, ATIVACAO) in hyperparameters_combinations:
            model = MLP(input_size, hidden_size, output_size, num_layers, dropout_prob)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs.squeeze(), y_train)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                outputs = model(X_test).squeeze()
                mse = mean_squared_error(y_test, outputs)
                RMSE = np.sqrt(mse)

            results_df = results_df.append({'Folds number': k_folds,
                                            'Fold': fold + 1,
                                            'Random State': random_state,
                                            'Dropout probability': dropout_prob,
                                            'Activation function': ativacao_name, 
                                            'Learning Rate': lr,
                                            'Epochs': num_epochs, 
                                            'Layers number': num_layers,
                                            'Hidden Size': hidden_size,
                                            'RMSE': RMSE,
                                        }, ignore_index=True)

    results_df.to_csv('results3.csv', index=False)
    ```

6. **Análise dos resultados:**
    ```python
    results_df = pd.read_csv('results3.csv')

    best_params = results_df.loc[results_df['RMSE'].idxmin()]
    print("\nMelhores parâmetros de RMSE:")
    print(best_params)

    worst_params = results_df.loc[results_df['RMSE'].idxmax()]
    print("\nPiores parâmetros de RMSE:")
    print(worst_params)

    mean_rmse = results_df['RMSE'].mean()
    print("\nMédia total de RMSE:", mean_rmse)

    mean_rmse_activation = results_df.groupby('Activation function')['RMSE'].mean()
    print("\nMédia do RMSE para cada função de ativação:")
    print(mean_rmse_activation)

    mean_rmse_layers = results_df.groupby('Layers number')['RMSE'].mean()
    print("\nMédia do RMSE para cada número de camadas:")
    print(mean_rmse_layers)

    mean_rmse_lr = results_df.groupby('Learning Rate')['RMSE'].mean()
    print("\nMédia de RMSE para cada valor de aprendizagem:")
    print(mean_rmse_lr)

    mean_rmse_epochs = results_df.groupby('Epochs')['RMSE'].mean()
    print("\nMédia de RMSE para cada número de épocas:")
    print(mean_rmse_epochs)

    mean_rmse_hidden_size = results_df.groupby('Hidden Size')['RMSE'].mean()
    print("\nMédia do RMSE para cada tamanho de camada interna:")
    print(mean_rmse_hidden_size)
    ```

### Resultados

A saída do código foi um dataset de 1500 linhas com as seguintes análises:

- **Melhores parâmetros de RMSE:**
    ```plaintext
    Folds number                  5
    Fold                          1
    Random State                100
    Dropout probability         0.5
    Activation function        ReLU
    Learning Rate              0.05
    Epochs                      200
    Layers number                 1
    Hidden Size                 256
    RMSE                   0.106414
    ```

- **Piores parâmetros de RMSE:**
    ```plaintext
    Folds number                  5
    Fold                          1
    Random State                100
    Dropout probability         0.5
    Activation function        ReLU
    Learning Rate              0.15
    Epochs                      100
    Layers number                 4
    Hidden Size                 256
    RMSE                   40.50468
    ```

- **Média total de RMSE:** 16.93279981701

- **Média do RMSE para cada função de ativação:**
    ```plaintext
    Activation function
    GELU        6.895653
    ReLU       11.490613
    Sigmoid    16.953957
    Softmax    27.937414
    Tanh       21.386362
    Name: RMSE, dtype: float64
    ```

- **Média do RMSE para cada número de camadas:**
    ```plaintext
    Layers number
    1    13.570037
    2    15.451817
    3    18.556452
    4    20.152893
    Name: RMSE, dtype: float64
    ```

- **Média de RMSE para cada valor de aprendizagem:**
    ```plaintext
    Learning Rate
    0.01    17.091399
    0.02    16.867574
    0.05    17.103925
    0.10    16.721659
    0.15    16.879442
    Name: RMSE, dtype: float64
    ```

- **Média de RMSE para cada número de épocas:**
    ```plaintext
    Epochs
    100    17.844950
    150    16.841357
    200    16.112093
    Name: RMSE, dtype: float64
    ```

- **Média do RMSE para cada tamanho de camada interna:**
    ```plaintext
    Hidden Size
    16     16.414477
    32     15.560734
    64     15.865818
    128    17.513823
    256    19.309148
    Name: RMSE, dtype: float64
    ```

### Conclusão

A análise dos resultados sugere que a escolha dos hiperparâmetros impactou de forma variada o desempenho da rede neural. Embora o número de camadas não tenha influenciado significativamente o RMSE médio, isso pode ser atribuído à capacidade limitada do conjunto de dados em capturar padrões mais complexos. A ReLU mostrou-se mais eficaz, enquanto a Softmax foi menos adequada. As taxas de aprendizagem não tiveram um efeito expressivo no desempenho, indicando uma robustez do modelo. O aumento do número de épocas contribuiu para a redução gradual do RMSE médio, sugerindo um aprendizado mais eficaz. Modelos com camadas internas menores tenderam a apresentar RMSEs mais baixos, possivelmente evitando overfitting e generalizando melhor. Esses resultados destacam a importância de ajustar cuidadosamente os hiperparâmetros para garantir que a arquitetura da rede neural seja adequada para o problema em questão.

### Referências

1. [Multi-Layer Perceptron (MLP) in PyTorch](https://medium.com/deep-learning-study-notes/multi-layer-perceptron-mlp-in-pytorch-21ea46d50e62)
2. [Inspiration for Model Evaluation](https://gist.github.com/GeorgeSeif/07a7707976a163cfaa94218db45a0046)
3. [Inspiration for Fixing a Recurring Error](https://stackoverflow.com/questions/62726792/pytorch-runtimeerror-expected-dtype-float-but-got-dtype-long)
4. [K-Fold Cross-Validation in Neural Networks](https://www.baeldung.com/cs/k-fold-cross-validation)
5. [General Error Fixing When Plotting Excel](https://stackoverflow.com/questions/71470352/ufunctypeerror-cannot-cast-ufunc-det-input-from-dtypeo-to-dtypefloat64)
6. [General Error Fixing When Plotting Excel](https://stackoverflow.com/questions/55290596/solving-slurm-sbatch-error-batch-job-submission-failed-requested-node-config)
7. [Dropout in PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
8. [K-Fold Cross-Validation with PyTorch](https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md)
9. Chat GPT para correção de erros simples

---

Este README foi atualizado para refletir a versão final do código fornecido, incluindo todos os detalhes e ajustes necessários.
