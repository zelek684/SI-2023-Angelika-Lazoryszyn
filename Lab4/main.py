import pandas as pd
import numpy as np
from functools import reduce
import itertools


def get_discernibility_matrix(data):
    n = data.shape[0]
    discernibility_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                discernibility_matrix[i, j] = sum(data.iloc[i, :-1] != data.iloc[j, :-1])
    return discernibility_matrix


def get_reduct(decision_system):
    discernibility_matrix = get_discernibility_matrix(decision_system)
    n, m = decision_system.shape
    attributes = set(range(m - 1))
    reduct = set()
    while attributes:
        min_cardinality = float('inf')
        min_element = None
        for attr in attributes:
            cardinality = sum(decision_system.iloc[:, attr].duplicated(keep=False))
            if cardinality < min_cardinality:
                min_cardinality = cardinality
                min_element = attr
        reduct.add(min_element)
        attributes.remove(min_element)
        if np.any(reduce(np.bitwise_and, discernibility_matrix[:, list(reduct)])):
            break
    return reduct


def get_rules(decision_system, reduct):
    unique_rows = decision_system.iloc[:, list(reduct) + [-1]].drop_duplicates()
    rules = []
    for _, row in unique_rows.iterrows():
        conditions = [f"{col} = {row[col]}" for col in reduct]
        rule = f"Jeśli {' i '.join(conditions)}, to dec = {row['dec']}"
        rules.append(rule)
    return rules




# Zad 1

print("\nZadanie 1")
# redukt z a i d zamieniony na b i d
decision_system1 = pd.DataFrame({
    'b': [2, 2, 0, 1],
    'c': [1, 2, 2, 1],
    'd': [0, 1, 1, 1],
    'dec': [0, 1, 2, 1]
})

reduct1 = get_reduct(decision_system1)
print(f"Redukt decyzyjny dla Figury. 1: {reduct1}")

# Zad 2
print("\nZad 2")
rules1 = get_rules(decision_system1, reduct1)
print("\nReguły wygenerowane z otrzymanego reduktu dec:")
for rule in rules1:
    print(rule)

# Zad 3
print("\nZad 3")
decision_system2 = pd.DataFrame({
    'a1': ['wysoka', 'wysoka', 'wysoka', 'wiecej niż srednia', 'wiecej niż srednia', 'wiecej niż srednia', 'wysoka', 'wiecej niż srednia', 'wiecej niż srednia'],
    'a2': ['bliski', 'bliski', 'bliski', 'daleki', 'daleki', 'daleki', 'bliski', 'daleki', 'daleki'],
    'a3': ['średni', 'średni', 'średni', 'silny', 'silny', 'lekki', 'średni', 'lekki', 'lekki'],
    'dec': ['tak', 'tak', 'tak', 'nie pewne', 'nie', 'nie', 'tak', 'nie', 'tak']
})

X1, X2 = 'a1', 'a2'
A = {X2}
B = {X1, X2}

subset_A = decision_system2.loc[:, list(A) + ['dec']].drop_duplicates()
subset_B = decision_system2.loc[:, list(B) + ['dec']].drop_duplicates()

print("Opis dla X2 w odniesieniu do A:")
for _, row in subset_A.iterrows():
    print(f"Jeśli {X2} = {row[X2]}, to dec = {row['dec']}")

print("Opis dla X1 i X2 w odniesieniu do B:")
for _, row in subset_B.iterrows():
    print(f"Jeśli {X1} = {row[X1]} i {X2} = {row[X2]}, to dec = {row['dec']}")

# Zad 4
print("\nZad 4")
reduct2 = get_reduct(decision_system2)
print(f"Redukt decyzyjny dla Figury. 2: {reduct2}")

rules2 = get_rules(decision_system2, reduct2)
print("Reguły wygenerowane z otrzymanego reduktu decyzyjnego dla Figury. 2:")
for rule in rules2:
    print(rule)


###########################################################

# Zad 5
print("\nZad5")
def check_consistency(rules, decision_system):
    for rule in rules:
        for row in decision_system:
            if all(rule[i] == row[i] for i in rule):
                if rule[-1] != row[-1]:
                    return False
    return True

def find_rules(decision_system):
    attributes = len(decision_system[0]) - 1
    rules = []

    while decision_system:
        found = False
        for row in decision_system:
            for length in range(1, attributes + 1):
                if found:
                    break
                for combination in itertools.combinations(range(attributes), length):
                    rule = {i: row[i] for i in combination}
                    rule[-1] = row[-1]
                    if check_consistency([rule], decision_system):
                        rules.append(rule)
                        decision_system = [r for r in decision_system if not all(rule[i] == r[i] for i in rule)]
                        found = True
                        break

    return rules


decision_system = [
    [1, 1, 1, 1, 3, 1, 1],
    [1, 1, 1, 1, 3, 2, 1],
    [1, 1, 1, 3, 2, 1, 0],
    [1, 1, 1, 3, 3, 2, 1],
    [1, 1, 2, 1, 2, 1, 0],
    [1, 1, 2, 1, 2, 2, 1],
    [1, 1, 2, 2, 3, 1, 0],
    [1, 1, 2, 2, 4, 1, 1]
]

rules = find_rules(decision_system)
print("Reguły:")
for rule in rules:
    print(rule)