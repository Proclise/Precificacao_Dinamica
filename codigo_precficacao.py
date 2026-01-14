import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# 1. Simulação de Dados de Histórico de Vendas
data = {
    'urgencia_lead': [1, 5, 3, 10, 2, 8, 4, 7, 9, 6], # Escala 1-10
    'projetos_em_aberto': [2, 10, 5, 8, 3, 9, 4, 7, 8, 6], # Capacidade da agência
    'dia_da_semana': [1, 5, 3, 2, 1, 4, 3, 5, 2, 4], # 1=Seg, 5=Sex
    'preco_final_fechado': [5000, 7500, 5500, 9000, 4800, 8200, 5200, 7000, 8800, 6500]
}

df = pd.DataFrame(data)

# 2. Treinamento do Modelo de IA
# O objetivo é aprender qual preço o mercado aceitou pagar nessas condições
X = df.drop('preco_final_fechado', axis=1)
y = df['preco_final_fechado']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 3. Função de Sugestão de Preço (O Motor de Precificação)
def sugerir_preco(urgencia, projetos, dia):
    input_data = pd.DataFrame([[urgencia, projetos, dia]], 
                              columns=['urgencia_lead', 'projetos_em_aberto', 'dia_da_semana'])
    preco_base = model.predict(input_data)[0]
    
    # Lógica de Negócio: Se a capacidade está no limite (projetos > 8), subimos 10%
    if projetos > 8:
        preco_base *= 1.10
        
    return preco_base

# 4. Exemplo de Uso em Tempo Real
novo_lead_urgente = sugerir_preco(urgencia=10, projetos=9, dia=2)
lead_baixa_demanda = sugerir_preco(urgencia=2, projetos=2, dia=1)

print(f"Preço sugerido para Lead Crítico (Alta Demanda): R$ {novo_lead_urgente:.2f}")
print(f"Preço sugerido para Lead Frio (Baixa Demanda): R$ {lead_baixa_demanda:.2f}")

# 5. Visualização do Impacto: Importância das Variáveis
importancias = model.feature_importances_
plt.barh(['Urgência', 'Capacidade', 'Sazonalidade'], importancias)
plt.title('Fatores que mais influenciam o Preço Final')
plt.show()
