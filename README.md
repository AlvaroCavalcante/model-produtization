# Produtização de modelos com TFX
Esta documentação visa mostrar o passo a passo seguido no desenvolvimento do projeto, discutindo brevemente a respeito das decisões tomadas.

## Padronização do código
Assim como em qualquer projeto que preze pelo código limpo, é necessário estabelecer e seguir os padrões e convenções da linguagem ou da equipe de desenvolvimento. Nesse caso, optei por adotar o padrão [PEP8](https://peps.python.org/pep-0008/) para todo o código Python em questão.

Para isso, a biblioteca [Pylint](https://pylint.pycqa.org/en/latest/tutorial.html) foi utilizada para avaliar a qualidade geral do código de acordo aos padrões supracitados. Para realizar a avaliação do mesmo, basta executar:

```
pylint src/python_script.py
```
A biblioteca vai sugerir pontos de melhoria e dar um nota geral para o código. Para fins de baseline, a qualidade inicial do código fornecido foi de **6.74**, conforme mostrado na imagem abaixo:

![Image](/assets/initial_code_quality.png "Qualidade de código inicial")

Além disso, todo o código será escrito e mantido em inglês.