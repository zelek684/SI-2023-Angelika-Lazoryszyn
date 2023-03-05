import numpy as np

plik_diabetes = np.loadtxt("diabetes.txt", dtype="str")
plik_diabetesTyp = np.loadtxt("diabetes-type.txt", dtype="str")
infoData = np.loadtxt("_info-data-discrete.txt", dtype="str")

print("\n***\n")

print("3b znajduje wielkość klas decyzyjnych, szukając wiersza w infoData, który odpowiada atrybutowi diabetes.\n")
x, y = np.where(infoData == "diabetes")
print(f"wielkość klas decyzyjnych = {infoData[int(x)][2]}")

print("\n***\n")

print("3c oblicza wartości minimalne i maksymalne dla każdego atrybutu w diabetes, "
      "gdzie typ atrybutu jest liczbowy (oznaczone jako diabetesType == n).\n")
x1, y1 = np.where(plik_diabetesTyp == "n")
for i in x1:
    temp = np.array(plik_diabetes[:, i], dtype='float')
    min = np.min(temp)
    max = np.max(temp)
    print(f"{plik_diabetesTyp[i][0]}: max = {max}, min = {min}")

print("\n***\n")


print("3d liczba unikalnych atrybutów dla każdego atrybutu w diabetes\n")
for i in range(len(plik_diabetes[0]) - 1):
    print(f"{plik_diabetesTyp[i][0]}: unikalne atrybuty = {len(np.unique(plik_diabetes[:, i]))}")

print("\n***\n")

print("3e unikalne wartości dla każdego atrybutu w diabetes.\n")
for i in range(len(plik_diabetes[0]) - 1):
    print(f"{plik_diabetesTyp[i][0]}: unikalne atrybuty = {np.unique(plik_diabetes[:, i])}")

print("\n****\n")


print("3f  odchylenie standardowe dla każdego atrybutu w diabetes, gdzie typ atrybutu jest "
      "liczbowy, oraz dla całego zbioru danych.\n")
x1, y1 = np.where(plik_diabetesTyp == "n")
wholeSystem = np.array([])
for i in x1:
    temp = np.array(plik_diabetes[:, i], dtype='float')
    wholeSystem = np.append(wholeSystem, temp)
    print(f"{plik_diabetesTyp[i][0]}: std = {np.std(temp)}")
print(f"caly system: std = {np.std(wholeSystem)}")

print("\n***\n")

print("4a obliczam wartość średnią dla pierwszego atrybutu liczbowego w diabetes,"
      " a następnie losowo zastępuje 10% wartości w zestawie danych znakiem zapytania.\n")
x1, y1 = np.where(plik_diabetesTyp == "n")
for i in x1:
    temp = np.array(plik_diabetes[:, i], dtype='float')
    avg = 0
    for j in temp:
        avg += j
    avg = np.round(avg / len(temp), 2)
    print(avg)
    break

dataChoice = np.random.choice([True, False], size=plik_diabetes.shape, p=[0.1, 0.9])
plik_diabetes[dataChoice] = "?"
print(plik_diabetes)

print("4b przeprowadzam skalowanie min-max dla całego zestawu danych diabetes i "
      "wyświetlam wyniki dla trzech różnych zakresów ([-1, 1], [0, 1] i [-10, 10]).\n")

with open('diabetes.txt', 'r') as file:
    wartosc = np.loadtxt(file)


min_wartosc = np.min(wartosc)
max_wartosc = np.max(wartosc)
scal_wartosc = (wartosc - min_wartosc) / (max_wartosc - min_wartosc)

a1 = -1
b1 = 1
normal_wartosc1 = a1 + (b1 - a1) * scal_wartosc

print(normal_wartosc1)

print("\n***\n")

a2 = 0
b2 = 1
normal_wartosc2 = a2 + (b2 - a2) * scal_wartosc

print(normal_wartosc2)

print("\n***\n")

a3 = -10
b3 = 10
normal_wartosc3 = a3 + (b3 - a3) * scal_wartosc

print(normal_wartosc3)

print("\n***\n")

print("4c standaryzuje trzeci atrybut w diabetes, używając normalizacji z-score."
      "przekształcenie każdej wartości w zestawie danych w jej odchylenie standardowe od średniej.\n")

with open('diabetes.txt', 'r') as file:
    wartosc = np.loadtxt(file)

i = 2
a_i = wartosc[:, i]
a_i_mean = np.mean(a_i)
a_i_std = np.std(a_i)
a_i_standaryzacja = (a_i - a_i_mean) / a_i_std

print(a_i_standaryzacja)