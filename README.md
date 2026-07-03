# 🐟 Regressão Linear - Biodiversidade de Peixes na Amazônia

Este é um projeto pequeno de autoaprendizagem na área de **Machine Learning**, com foco em **Regressão Linear Múltipla**.

---

## 🎯 Objetivo

O objetivo do modelo é prever o número de espécies de peixes (`Num_Species`) em bacias hidrográficas da Amazônia, com base em características ecológicas como:

- Área da bacia (km²)
- Número de pontos de coleta (`Num_Sites`)
- Número de ocorrências (`Num_Occurrences`)
- Número de famílias (`Num_Families`)
- Número de gêneros (`Num_Genera`)
- Número de espécies endêmicas (`Num_Endemic_Species`)

---

## 📊 Base de dados

O dataset foi construído a partir de dados do artigo:

> *“A database of freshwater fish species of the Amazon Basin”*  
> disponível em: www.nature.com/scientificdata

Os dados foram organizados em formato CSV para facilitar o uso em modelos de Machine Learning.

---

## 🧠 Abordagem

Foi utilizado um modelo de **Regressão Linear Múltipla**, que aprende a relação entre as variáveis ambientais e a biodiversidade registrada nas bacias.

O modelo tenta entender como essas características influenciam a quantidade de espécies presentes em cada região.

---

## ⚙️ Tecnologias e bibliotecas

- Python
- Pandas
- Scikit-learn
- Matplotlib

---

## 📈 Resultados

O modelo apresentou um desempenho muito bom para um projeto de caráter didático:

- **MAE (Erro Médio Absoluto):** ~18 espécies  
- **R² (Coeficiente de determinação):** ~0.995  

Isso indica que o modelo conseguiu explicar bem a variação no número de espécies com base nas variáveis fornecidas.

---

## 📌 Exemplo de previsão:
Bacia: Xingu
Real: 821 espécies
Previsto: ~810 espécies


---

## 🧪 O que foi aprendido

Este projeto foi desenvolvido com foco em aprendizado prático e envolveu:

- Manipulação de dados com Pandas
- Separação de treino e teste
- Normalização de dados (StandardScaler)
- Treinamento de modelo de regressão
- Avaliação com MAE e R²
- Visualização de resultados com Matplotlib
- Inferência baseada em nome de bacia

---

## ⚠️ Observação

Este é um projeto de estudo.  
Os resultados são válidos apenas para o conjunto de dados utilizado e não devem ser interpretados como previsões ecológicas definitivas.

---

## 💡 Conclusão

Apesar de simples, o projeto foi muito útil para consolidar conceitos de Machine Learning e entender como variáveis ambientais podem ser usadas para modelar padrões de biodiversidade.

Além disso, serviu como prática completa de um pipeline de regressão do início ao fim.

