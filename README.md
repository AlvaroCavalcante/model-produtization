# Produtização de modelos com TFX
Esta documentação visa mostrar o passo a passo seguido no desenvolvimento do projeto, discutindo brevemente a respeito das decisões tomadas. Para melhor visualizar a documentação, recomendo o uso de uma ferramenta de visualização de Markdown. A **Markdown Preview Enhanced** é um exemplo de extenção do Visual Studio Code que permite a visualização desse tipo de documento.

## Instalação e setup do projeto
**Nota:** O projeto foi executado em um ubuntu 20.04 com Python 3.8.

Para instalar as dependências, primeiro é recomendável utilizar um ambiente virtual para isolar as dependências e garantir que não serão enfrentados conflitos com as bibliotecas já instaladas na máquina. Para instalar o virtualenv, basta executar:

```
pip3 install virtualenv 
```
**NOTA:** Caso você **não tenha o pip**, o seguinte comando pode ser utilizado para instalar o mesmo:
```
sudo apt-get install python-pip python-virtualenv python-dev build-essential
```

Feito isso, é possível iniciar um novo ambiente com o seguinte comando:
```
python3 -m venv tfx_pipeline
```
para fazer a ativação do ambiente, é preciso executar:
```
source tfx_pipeline/bin/activate
```
Com o ambiente virtual ativo no terminal, é necessário acessar a raiz do projeto e executar o seguinte comando:
```
pip3 install -r requirements.txt
```
A instalação pode demorar alguns minutos, mas vai garantir que todas as dependências sejam corretamente instaladas. Feito isso, é necessário iniciar o DB do Airflow com o seguinte comando: 

```
airflow db init
```
Com isso, uma pasta com o nome "airflow" deve ter sido criada na pasta pessoal, com os arquivos de configuração do serviço: 

![Image](/assets/airflow.png "Airflow")
Feito isso, execute o seguinte comando em seu terminal:
```
mkdir -p $HOME/airflow/dags/
```
Isso deve criar uma pasta "dags" dentro das configurações do Airflow. Por fim, vamos copiar a pipeline criada aqui para dentro das dags:
```
cp src/tfx_pipepline.py $HOME/airflow/dags/
```
## Execução do projeto
Para executar o script refatorado do treinamento do modelo, basta utilizar o seguinte comando:
```
python3 src/refactored_code.py
```
Se tudo funcionou, você deve ver um print com a precisão, recall e KS do modelo.

Para executar o AirFlow, primeiro é preciso criar um usuário e senha de acesso, o que pode ser feito com o seguinte comando:

```
airflow users  create --role Admin --username admin --email admin --firstname admin --lastname admin --password admin
```

Feito isso, é necessário abrir dois novos terminais (não se esqueça de iniciar o ambiente virtual). No primeiro terminal, é preciso executar:
```
airflow webserver
```
e no segundo:
```
airflow scheduler
```
Pronto, basta navegar para o endereço [localhost:8080](http://localhost:8080) e acessar o Airflow utilizando "admin" como usuário e senha. Caso você não encontre o **"cartola_pro_clients"** na lista de DAGS, execute o seguinte comando: 

```
python3 src/tfx_pipeline.py
```
**NOTA:** É necessário alterar o valor da variável _project_root com o caminho da sua máquina.

Por fim, basta localizar a DAG que deseja executar e dar um trigger, conforme mostrado abaixo:
![Image](/assets/exec_air.png "Airflow")

A pipeline de treinamento e deployment de modelos do TFX já irá iniciar!
## Padronização do código
Assim como em qualquer projeto que preze pelo código limpo, é necessário estabelecer e seguir os padrões e convenções da linguagem ou da equipe de desenvolvimento. Nesse caso, a escrita de todo o código Python foi feita inteiramente em inglês (com exceção de nomes próprios) e utilizando o padrão [PEP8](https://peps.python.org/pep-0008/).

Para auxiliar nessa tarefa, a biblioteca [Pylint](https://pylint.pycqa.org/en/latest/tutorial.html) foi utilizada para avaliar a qualidade geral do código de acordo aos padrões supracitados. Para realizar a avaliação do mesmo, basta executar:

```
pylint src/desired_python_script.py
```
A biblioteca vai sugerir pontos de melhoria e dar um nota geral para o código. Para fins de baseline, a qualidade inicial do código fornecido foi de **6.74**, conforme mostrado na imagem abaixo:

![Image](/assets/initial_code_quality.png "Qualidade de código inicial")

Após algumas melhorias como alterações nas ordens de import, espaçamento das linhas, remoção de código desnecessário e entre outras, o código atingiu uma nota de **8.9**, conforme exibido abaixo:

![Image](/assets/new_code_quality.png "Qualidade do código refatorado")

## Demais alterações e boas práticas
Algumas mudanças foram realizadas no código a fim de melhorar a usabilidade e produtização do mesmo ou adequá-lo às boas práticas. Primeiro, todas as etapas de pré-processamento foram colocadas dentro de funções, para permitir que as mesmas sejam testadas de maneira unitária, além de facilitar a sua chamada a partir de outros scripts/classes.

O parâmetro "random_state" foi adicionado ao método **train_test_split** e também à regressão logística, para favorecer a reprodutibilidade dos resultados.

Os dados de **X_train___** e **X_test___** são agora gerados com o uso do método de **transform**, visto que o uso de **fit_transform** fazia com que os dados se reajustassem a base em questão, o que é um antipadrão por permitir o fit nos conjutos de teste e validação.

O **PCA** foi removido da pipeline, pois a adoção do mesmo não gerou melhorias significativas nos resultados. Além disso, reduzir a dimensionalidade da informação implica na perda de explicabilidade, dificultando o entendimento do modelo.

Uma das formas de reduzir a quantidade de variáveis e simplificar o modelo gerado foi através da adoção do VIF (Variance Inflation Factor). De maneira simplificada, o VIF mede o índice de colinearidade entre as variáveis do dataset, evidenciando as features que poderiam ser excluídas sem gerar grandes impactos na performance do modelo.

O método **get_colinear_features** faz a iteração pelo dataset até trazer todas as features que têm um VIF maior que 3. Ao total, 7 variáveis foram excluídas do modelo, sendo elas: ```log_interacoes_g1, norm_rodadas, rel_pont, dif_melhoria, log_iteracao_atletismo, max_camp, log_anos_desde_criacao```. Ainda assim, os resultados atingidos pelo modelo se mantiveram os mesmos.

Por fim, embora as variáveis remanescentes não sejam colineares, nem todas precisam fazer parte do modelo, visto que algumas delas possuem pouca influência no resultado gerado pelo mesmo. O gráfico abaixo mostra a importância das variáveis na regressão logística:

![Image](/assets/feature_impo.png "Importância das features")

Deste modo, as variáveis menos relevantes foram excluídas progressivamente, reduzindo o conjunto de dados e mantendo o mesmo poder estatístico. Ao final, apenas 6 variáveis foram mantidas, sendo elas: ```anos_como_pro, ceil_avg_3, ceil_avg_4, escalacoes, norm_escalacao, log_tempo_desperd```. É evidente que, o real significado dessas variáveis e a conclusão de que se elas fazem realmente sentido ou não (considerando o contexto de negócio e analises estatísticas mais profundas) é algo a ser avaliado. Ainda assim, os resultados com o conjunto de dados reduzido foi o mesmo.

Para fins de demonstração, o **refactored_code.py** contém a versão melhorada do script inicial.

### Métrica de avaliação
O modelo inicial chega a atingir uma precisão de 81% e uma revocação de 75%, considerando um limiar de 50% na classificação. Ainda que essas métricas possam orientar a capacidade do modelo em discriminar usuários PRO e não PRO, o principal objetivo do projeto, em termos de negócios, é auxiliar na geração de campanhas que consigam incentivar o aumento de inscrições de usuários como PRO e reduzir o "churn" dos que já são. 

Dessa maneira, mais do que apenas adivinhar o valor 0 ou 1, precisamos entender a distribuição probabilística gerada nas previsões para uma tomada de decisão assertiva, visto que os clientes serão segmentados pelos que são PRO e que tem uma baixa probabilidade de continuar como PRO, e clientes que não são PRO e que tem uma alta probabilidade de se tornarem PRO.

Baseado nisso, a métrica de KS (Kolmogorov-Smirnov), pode ser a que melhor se encaixa nesse contexto, visto que a mesma tem por objetivo medir a diferença entre duas distribuições probabilísticas. Essa métrica varia de 0 a 1, sendo que 0 representa uma distribuição idêntica de probabilidades, e 1 representa distribuições diferentes.

Assim sendo, maximizar a métrica de KS faz com que o modelo consiga separar o máximo possível a classe 1 da classe 0, aumentando o intervalo de confiança entre os clientes PRO e não PRO. O método **calculate_ks_score** foi criado para o cálculo da métrica, exibindo um **KS inicial de 0,60**. Na prática, isso significa que as duas classes já possuem um bom nível de separação entre as suas probabilidades. Para mostrar esse efeito de maneira mais visual, é possível verificar que a média dos valores de probabilidade dos assinantes PRO é de **0,73**, enquanto os não PRO é de **0.24**. Ao plotar as duas distribuições, obtemos o seguinte gráfico:

![Image](/assets/distributions.png "Distribuição das previsões")

Onde azul é a distribuição probabilística dos PRO e laranja os não PRO.

# Criação de pipelines com Airflow e TFX
Uma vez que o script de treinamento do modelos está finalizado, é interessante criar uma pipeline que contenha as etapas de pré-processamento, treinamento e deploy, automatizando o processo, garantindo reprodutibilidade e possibilitando utilizar ferramentas para melhorar o ciclo de vida do mesmo. Nesse caso, optei por adotar o TensorFlow Extended (TFX) para a pipeline, visto que é uma solução Open-Source e que não está diretamente atrelada a um cloud provider específico.

O Airflow, por sua vez, foi utilizado para exibição e execução da pipeline, visto que é uma ferramenta bastante adotada pela comunidade. O código fonte com a definição da pipeline pode ser visto no arquivo **tfx_pipeline.py**. Cada um dos componenetes que formam a pipeline serão explicado individualmente nas seções posteriores.

### Input de dados
O primeiro componente do fluxo é o gerador de *Examples* (instâncias de dados), responsável por trazer os dados provenientes de arquivos ou serviços externos para dentro da pipeline. Nessa caso, utilizei o **CsvExampleGen** para ler o arquivo CSV e realizar o split, conversão e particionamento de dados.

Por padrão, os dados são divididos em múltiplos splits para melhorar a paralelização das operações. Cada split é composto por duas partes (~66%) de dados utilizados para treinamento e uma parte (~33%) para teste. Além disso, os dados são convertidos para o formato binário "TFRecord", pois é a forma padrão do TensorFlow manipular os dados.

### Gerador de estatísticas
O gerador de estatísticas (**StatisticsGen**) é um componente que tem por objetivo levantar as principais características dos dados através de suas estatísticas. Para as features numéricas, o componente calcula a média, desvio padrão, valores mínimos e máximos e etc. Para as categóricas, a cardinalidade e a frequência do conjunto de strings é calculada. 

Dessa forma, sempre que a pipeline for executada, para fazer o retreino do modelo, por exemplo, as estatísticas são calculadas nos novos dados e armazenadas para fins de comparação. Com isso, é possível identificar facilmente qualquer tipo de desvio, outlier ou valor incorreto em cada uma das variáveis. A imagem abaixo mostra algumas das estatísticas levantadas no conjunto de dados em questão:

![Image](/assets/stats.png "Estatísticas dos dados")

### Gerador de Schema
Para que os componentes do TensorFlow funcionem corretamente na pipeline, é necessário que as propriedades das variáveis sejam conhecidas, como o tipo de dados, shape, e etc. Por conta disso, o **SchemaGen** é utilizado para definir o Schema das variáveis e permitir que esse schema seja utilizado nas demais etapas de manipulação dos dados. A imagem abaixo ilustra o schema das variáveis no dataset em questão:

![Image](/assets/schema.png "Schema dos dados")

### Validador de dados
Uma vez que as estatísticas das features foram calculadas e seu schema atribuído, é possível utilizar o componente **ExampleValidator**. Esse componente, como o nome sugere, compara as estatísticas calculadas com os dados atuais e verifica se o schema está de acordo aos dados. Caso algum desvio ou anomalia seja encontrada, essa etapa da pipeline irá apresentar falha. Em uma situação de normalidade, esse componente gera os seguintes logs:

![Image](/assets/anomalies.png "ExampleVal")

### Feature engineering
Com os dados validados, o componente Transform é utilizado para realizar o feature engineering. Esse componente, que é parte da biblioteca do TensorFlow Transform, faz a leitura do arquivo **tfx_utils.py** e chama a função **preprocessing_fn**. 

Essa função possui o código fonte responsável por pré-processar as features, de maneira similar ao que foi feito no script **refactored_code.py**, porém, ao invés de utilizar funções pandas e numpy, é possível apenas utilizar a API do TensorFlow para realizar as operações. Isso acontece pelo fato de que todas as operações realizadas no TensorFlow são adicionadas no grafo de execução e organizadas para melhor otimizar os cálculos computacionais.

### Treinamento
De forma similar, o componente **Trainer** também depende do arquivo **tfx_utils.py**, porém, o mesmo faz a chamada da função **run_fn**. Essa função, por sua vez, é responsável por criar os geradores de dados de treinamento e validação, criar a arquitetura do modelo e fazer o salvamento dos checkpoints do modelo treinado.

Nesse caso, a API do Keras foi utilizada para criar um modelo simples de regressão logística. Além disso, a função **_get_serve_tf_examples_fn** foi criada para adicionar o layer de Transform ao modelo, responsável pelo feature engineering. Assim, o pré-processamento de dados passa a ser parte das camadas iniciais do mesmo, não sendo necessário escrever uma função a parte para tratar os dados.


### Avaliação de performance
Uma vez que o modelo foi treinado, a pipeline também deve ter um mecanismo para definir se esse modelo é bom o bastante para ser disponibilizado em produção. Para isso, dois componentes precisam ser utilizados, sendo o **Evaluator** e o **Resolver**. 

O Evaluator possui uma série de configurações e hiperparâmetros que têm por principal objetivo fazer a avaliação do modelo seguindo diferentes aspectos. Além de analisar métricas comuns como precisão, acurácia e revocação, o Evaluator usa de diversas outras medidas estatísticas, além de fazer estratificações nos dados, para avaliar as previsões e resultados do modelo de maneira aprofundada.

O Resolver, por sua vez, é responsável por carregar o último modelo treinado e que foi aceito em produção e avaliá-lo utilizar suas métricas de baseline no evaluator. Dessa forma, um novo modelo só se torna candidato a produção se tiver superado o modelo já implantado.

Caso seja a primeira execução da pipeline, o comparativo se baseia apenas em limiares, por exemplo, uma taxa mínima de precisão para aceitar determinado modelo.

### Deploy 
Finalmente, o **pusher** é o componente responsável por realizar o deploy do modelo (caso esse novo modelo tenha superado o anterior). Por padrão, esse componente gera uma pasta com o nome do modelo e as versões do mesmo, conforme o exemplo abaixo:

![Image](/assets/pusher.png "Deploy do modelo")

Cada uma das versões possui um arquivo saved_model compatível com o TensorFlow e o Keras, o qual pode ser facilmente utilizado para realizar a predição de dados.

## Pipeline completa
A pipeline completa pode ser visualizada abaixo:

![Image](/assets/complete_pipeline.png "Pipeline")

A mesma pode ser executada via trigger manual ou através de um scheduler, executando de tempos em tempos.

## Conclusões
Em relação ao modelo, muito mais análise e otimização poderia ser feita, como por exemplo: utilizar a validação cruzada para treinar em todo o conjunto, adotar um gridsearch ou randomsearch para encontrar parâmetros melhores e também testar outros modelos como florestas aleatórias e redes neurais. 

Ainda assim, o principal foco do trabalho foi na criação de uma pipeline que consiga produtizar o modelo, uma vez que o mesmo foi criado. O script **refactored_code.py** ilustra um script padrão que foi utilizado para realizar a análise exploratória de dados, busca de melhores parâmetros e treinamento. Em outras palavras, esse script representa o output do time de ciência de dados. 

Como MLE, a principal tarefa foi transformar o script em uma pipeline que permita o retreinamento do modelo, garantindo que a mesma possa ser executada de maneira autônoma (inclive subir novas versões de modelo) e que seja monitorada contra possíveis anomalias nos dados.

Com apenas algumas mudanças, como na fonte de input que provavelmente será de um SGBD e não de um CSV e do output, que provavelmente será em algum servidor cloud, a pipeline está pronta para produtizar modelos.