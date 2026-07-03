import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("peixe.csv") # abre o arquivo e transforma numa tabela, guardada na variável df
print(df.head()) # mostra só as primeiras 5 linhas
print(df.isnull().sum()) # conta quantos valores estão faltando em cada coluna (importante, já sabemos que Num_Endemic_Species tem 4 faltando)

df["Num_Endemic_Species"] = df["Num_Endemic_Species"].fillna(0)
# preenche os valores faltando com 0 (o artigo usa "—" quando não há espécie endêmica reportada, então faz sentido tratar como zero)

X = df.drop(columns=["Tributary", "Num_Species"])
# pega a tabela inteira e remove a coluna Tributary (é só o nome, não é medida) e Num_Species (é o que queremos prever) → sobra X
Y = df["Num_Species"]
# pega só a coluna que queremos prever → Y (agora é um número, não uma categoria)
print(X.head())
print(Y.head())

# LabelEncoder só é necessário quando o alvo é uma CATEGORIA (tipo nome de espécie).
# Aqui Y já é um número (quantidade de espécies), não precisa traduzir nada.

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
# test_size=0.2 reserva 20% dos dados pro teste, 80% fica pro treino
# random_state=42 "trava" a aleatoriedade, bom pra comparar resultados depois
# Repara: SEM stratify aqui! stratify serve pra manter proporção de CATEGORIAS
# (tipo garantir que toda espécie rara apareça no teste). Como Y agora é um
# número contínuo, não existe "categoria" pra estratificar.

print("Treino: ", X_train.shape)
print("Teste: ", X_test.shape)

# Normalização

from sklearn.preprocessing import StandardScaler

# IMPORTANTE: fit_transform() SÓ no treino, nunca no teste!
# fit() = calcula média e desvio padrão dos dados
# transform() = aplica essa conta pra normalizar
# Se eu desse fit() no teste também, a normalização "espiaria"
# informação do teste antes da hora -> DATA LEAKAGE.
# Tudo que precisa de fit() só pode olhar pro X_train.
# O X_test só passa por transform(), usando o que já foi aprendido no treino.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_test_scaled[:3])

# Regressão Linear

from sklearn.linear_model import LinearRegression

modelo = LinearRegression()
# Criando o modelo. Repente que aqui não tem max_iter=1000: LinearRegression
# resolve a equação matematicamente de forma direta, não precisa de
# "tentativas" iterativas como a LogisticRegression precisava.
modelo.fit(X_train_scaled, Y_train)
# Aprendizado de verdade. Modelo olha as medidas (X_train_scaled) e o número
# real de espécies (Y_train) e ajusta os parâmetros internos pra tentar
# prever Y_train a partir de X_train_scaled

from sklearn.metrics import r2_score, mean_absolute_error

Y_pred = modelo.predict(X_test_scaled)

print("Previsoes: ", Y_pred)
print("Valores reais: ", Y_test.values)

mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
# mae = em média, quantas espécies de diferença o modelo erra pra mais ou pra menos
# r2 mede o quanto o modelo conseguiu aprender o padrão existente nos dados.
# Quanto mais próximo de 1, melhor.
# Ex.: r2 = 0.85 significa que o modelo explica cerca de 85% da variação
# observada no número de espécies.

print("ERRO MÉDIO ABSOLUTO (MAE): ", mae)
print("O quanto o modelo explica (R2): ", r2)

print("Intercepto: ", modelo.intercept_)

for nome, coef in zip(X.columns, modelo.coef_):
    print(f"{nome}: {coef}")

resultado = pd.DataFrame({
    "Real": Y_test.values,
    "Previsto": Y_pred
})

resultado["Erro"] = resultado["Previsto"] - resultado["Real"]
resultado["Previsto"] = resultado["Previsto"].round(2)
resultado["Erro"] = resultado["Erro"].round(2)

print(resultado)

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(Y_test,Y_pred)

plt.plot(
    [Y_test.min(), Y_test.max()],
    [Y_test.min(), Y_test.max()],
    color="red",
    linestyle="--"
)

plt.xlabel("Núumero real de espécies")
plt.ylabel("Número previsto")
plt.title("Valores reais x previstos")

plt.show()

plt.figure(figsize=(8,5))

plt.scatter(df["Area_km2"], df["Num_Species"])

plt.xlabel("Área (km2)")
plt.ylabel("Número de espécies")
plt.title("Área da bacia x número de espécies")

plt.show()

plt.figure(figsize=(8,5))

plt.scatter(df["Num_Genera"], df["Num_Species"])

plt.xlabel("Número de generos")
plt.ylabel("Número de espécies")
plt.title("Núumero de generos x espécies")

plt.show()

print("=" * 40)
print("Resumo do modelo")
print(f"MAE: {mae:.2f} espécies")
print(f"R2: {r2:.4f}")
print("=" * 40)

def prever_bacia(nome_bacia):

    nome = nome_bacia.strip().lower()
    linha = df[df["Tributary"].str.strip().str.lower() == nome]

    if linha.empty:
        print("Bacia não encontrada.")
        return

    dados = linha.drop(columns=["Tributary", "Num_Species"])

    dados_scaled = scaler.transform(dados)

    previsao = modelo.predict(dados_scaled)

    print(f"Bacia: {nome_bacia}")
    print(f"Real: {linha['Num_Species'].values[0]}")
    print(f"Previsto: {previsao[0]:.2f}")
    
nome = input("Digite o nome da bacia: ")
prever_bacia(nome)