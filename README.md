# RPG_MLP-Project
Repositório referente ao trabalho da disciplina de Redes Neurais e Algoritmos Genéticos ministrada pelo professor Daniel Cassar.

**O objetivo do notebook codado em python é o seguinte:
- Identique e otimize os hiperparâmetros de uma rede neural do tipo MLP para resolver um problema de regressão de interesse cientíco.Espera-se o uso de boas práticas em ciências de dados assim como já apresentado em disciplinas anteriores.Teste ao menos 100 diferentes arquiteturas de rede durante sua otimização de hiperparâmetros. Os dados para treinar o modelo devem ser dados tabulados de interesse cientíco, não podem ser séries temporais e precisam ser aprovados pelo professor.**

 Para realizar essa tarefa utilizamos o Jupyter lab, Redes Neurais Pytorch e um dataset focado na composição do concreto.

1. Dataset

Como citado acima cada coluna do dataset "Civil Engineering: Cement Manufacturing Dataset" é um dado da composição do concreto, tempo de preparo e força suportada. As colunas são:

**Cimento:** É o aglomerante principal do concreto, responsável por unir os materiais. Medido em quilogramas por metro cúbico de mistura.

**Escória de alto-forno:** Um subproduto da indústria siderúrgica que pode ser usado como adição ao concreto para melhorar suas propriedades. Medido em quilogramas por metro cúbico de mistura.

**Cinzas volantes:** Resíduo da queima de carvão em usinas termelétricas, utilizado no concreto para melhorar sua durabilidade. Medido em quilogramas por metro cúbico de mistura.

**Água:** Essencial para a hidratação do cimento e a mistura dos materiais. Medida em quilogramas por metro cúbico de mistura.

**Superplastificante:** Aditivo que melhora a trabalhabilidade do concreto sem comprometer sua resistência. Medido em quilogramas por metro cúbico de mistura.

**Agregado graúdo:** Pedras ou britas que compõem o concreto, fornecendo resistência mecânica. Medido em quilogramas por metro cúbico de mistura.

**Agregado miúdo:** Areia ou pó de pedra que preenche os vazios entre os agregados graúdos, melhorando a trabalhabilidade. Medido em quilogramas por metro cúbico de mistura.

**Idade:** Tempo em dias desde a mistura do concreto.

**Resistência à compressão do concreto:** Medida em Megapascal (MPa), indica a capacidade do concreto de suportar cargas de compressão.
