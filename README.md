# Produtização de modelos com TFX
Esta documentação visa mostrar o passo a passo seguido no desenvolvimento do projeto, discutindo brevemente a respeito das decisões tomadas.

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

Para fins de demonstração, o código refatorado se encontra no arquivo **refactored_code.py**.

## Demais alterações e boas práticas
Algumas mudanças foram realizadas no código a fim de melhorar a usabilidade e produtização do mesmo. Primeiro, todas as etapas de pré-processamento foram colocadas dentro de funções, para permitir que as mesmas sejam testadas de maneira unitária, além de facilitar a sua chamada a partir de outros scripts/classes.

O parâmetro "random_state" foi adicionado ao método train_test_split, para garantir que a divisão de dados seja sempre a mesma, favorecendo a reprodutibilidade dos resultados.

Os dados de **X_train___** e **X_test___** são agora gerados com o uso do método de **transform**, visto que o uso de **fit_transform** fazia com que os dados se reajustassem a base em questão, o que é um antipadrão por permitir o fit nos conjutos de teste e validação.

O **PCA** foi removido da pipeline, pois a adoção do mesmo não gerou melhorias significativas nos resultados. Além disso, reduzir a dimensionalidade da informação implica na perda de explicabilidade, dificultando o entendimento do modelo.

Uma das formas de reduzir a quantidade de variáveis e simplificar o modelo gerado foi através da adoção do VIF (Variance Inflation Factor). De maneira simplificada, o VIF mede o índice de colinearidade entre as variáveis do dataset, evidenciando as features que poderiam ser excluídas sem gerar grandes impactos na performance do modelo.

O método **get_colinear_features** faz a iteração pelo dataset até trazer todas as features que têm um VIF maior que 3. Ao total, 7 variáveis foram excluídas do modelo, sendo elas: ```log_interacoes_g1, norm_rodadas, rel_pont, dif_melhoria, log_iteracao_atletismo, max_camp, log_anos_desde_criacao```. Ainda assim, os resultados atingidos pelo modelo se mantiveram os mesmos.

O **refactored_code.py** contém a versão melhorada do script inicial.

### Métrica de avaliação
O modelo inicial chega a atingir uma precisão de 81% e uma revocação de 75%, considerando um limiar de 50% na classificação. Ainda que essas métricas possam orientar a capacidade do modelo em discriminar usuários pró e não pró, o principal objetivo do time, em termos de negócios, é auxiliar na geração de campanhas que consigam incentivar o aumento de inscrições de usuários como pró e reduzir o "churn" dos que já são. 

Dessa maneira, mais do que apenas adivinhar o valor 0 ou 1, precisamos entender a distribuição probabilística gerada nas previsões para uma tomada de decisão assertiva, visto que os clientes serão segmentados pelos que são pró e que tem uma baixa probabilidade de continuar como pró, e clientes que não são pró e que tem uma alta probabilidade de se tornarem pró.

Baseado nisso, a métrica de KS (Kolmogorov-Smirnov), pode ser a que melhor se encaixa nesse contexto, visto que a mesma tem por objetivo medir a diferença entre duas distribuições probabilísticas. Essa métrica varia de 0 a 1, sendo que 0 representa uma distribuição idêntica de probabilidades, e 1 representa distribuições diferentes.

Assim sendo, o objetivo de maximizar a métrica de KS é fazer com que o modelo consiga separar o máximo possível a classe 1 da classe 0, aumentando o intervalo de confiança entre os clientes pró e não pró. O método **calculate_ks_score** foi criado para o cálculo da métrica, exibindo um **KS inicial de 0,60**. Na prática, isso significa que as duas classes já possuem um bom nível de separação entre as suas probabilidades. Para exemplificar, é possível calcular que a média dos valores de probabilidade dos assinantes pró é de **0,73**, enquanto os não pró é de **0.24**. Ao plotar as duas distribuições, obtemos o seguinte gráfico:

![Image](/assets/distributions.png "Qualidade do código refatorado")

Onde azul é a distribuição probabilística dos pró e laranja os não pró.

# Criação de pipelines com Airflow e TFX
Uma vez que todo o fluxo do modelo foi finalizado, é interessante criar uma pipeline que contenha as etapas de pré-processamento, treinamento e deploy, automatizando o processo, garantindo reprodutibilidade e possibilitando utilizar ferramentas para melhorar o ciclo de vida do mesmo. Nesse caso, optei por utilizar o TensorFlow Extended (TFX) para criar a pipeline, visto que é uma solução Open-Source e que não está diretamente atrelado a um cloud provider específico.

O Airflow, por sua vez, foi utilizado para exibição e execução da pipeline, visto que é uma ferramenta bastante adotada pela comunidade.

