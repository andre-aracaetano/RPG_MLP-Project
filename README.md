# RPG_MLP-Project
Repositório referente ao trabalho da disciplina de Redes Neurais e Algoritmos Genéticos ministrada pelo professor Daniel Cassar.

**O objetivo do notebook codado em python é o seguinte:**
- Identique e otimize os hiperparâmetros de uma rede neural do tipo MLP para resolver um problema de regressão de interesse cientíco.Espera-se o uso de boas práticas em ciências de dados assim como já apresentado em disciplinas anteriores. Teste ao menos 100 diferentes arquiteturas de rede durante sua otimização de hiperparâmetros. Os dados para treinar o modelo devem ser dados tabulados de interesse cientíco, não podem ser séries temporais e precisam ser aprovados pelo professor.

 Para realizar essa tarefa utilizamos o Jupyter lab, Redes Neurais Pytorch e um dataset focado na composição do concreto.

# Dataset

Como citado acima cada coluna do dataset **"Civil Engineering: Cement Manufacturing Dataset"** é um dado que contem a composição do concreto, tempo de preparo e força suportada.

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

# O processo

O documento "O_processo" inclui quatro redes MLP que criamos antes de chegar-mos na rede final. Embora não sejam o trabalho final, são redes interessantes para resolver outros problemas, ou para acompanhar a nossa jornada.

# A rede neural


# Referências

- Dataset:

1 e 2: redes com pytorch

3: inspiração para como avaliar o modelo

4: inspiração para corrigir um erro recorrente

5: validação cruzada em redes neurais

6 e 7: correção geral de erros ao plotar o excel

8: inpiração para dropout

9: inspiração para validação cruzada

1. https://medium.com/deep-learning-study-notes/multi-layer-perceptron-mlp-in-pytorch-21ea46d50e62
2. https://medium.com/deep-learning-study-notes/multi-layer-perceptron-mlp-in-pytorch-21ea46d50e62
3. https://gist.github.com/GeorgeSeif/07a7707976a163cfaa94218db45a0046
4. https://stackoverflow.com/questions/62726792/pytorch-runtimeerror-expected-dtype-float-but-got-dtype-long
5. https://www.baeldung.com/cs/k-fold-cross-validation
6. https://stackoverflow.com/questions/71470352/ufunctypeerror-cannot-cast-ufunc-det-input-from-dtypeo-to-dtypefloat64
7. https://stackoverflow.com/questions/55290596/solving-slurm-sbatch-error-batch-job-submission-failed-requested-node-config
8. https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
9. https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
