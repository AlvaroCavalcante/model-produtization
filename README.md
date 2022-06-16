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

O **refactored_code.py** contém a versão melhorada do script inicial.
