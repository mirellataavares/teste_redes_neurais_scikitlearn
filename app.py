import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(
    page_title="IA - Análise de Treinamento", page_icon="📊", layout="wide"
)


st.title("📊 Painel Automatizado de Treinamento Corporativo")
st.subheader("Análise de Qualidade de Serviços com Inteligência Artificial")
st.markdown(
    """
Este sistema utiliza uma **Rede Neural Perceptron Multicamadas (MLP)** para analisar o volume de reclamações 
proporcional aos atendimentos e categorizar automaticamente a performance dos seus serviços.
"""
)

st.divider()




@st.cache_resource
def treinar_rede_neural():
    # Base de conhecimento para a IA: proporção de reclamações (0.00 a 1.00)
    X_train = np.array([[0.05], [0.12], [0.25], [0.38], [0.50], [0.65]])
    # Classes correspondentes: 0=Excelente, 1=Bom, 2=Ruim, 3=Muito Ruim
    y_train = np.array([0, 0, 1, 1, 2, 3])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    mlp = MLPClassifier(
        hidden_layer_sizes=(5,),
        max_iter=1000,
        random_state=42,
        learning_rate_init=0.01,
    )
    mlp.fit(X_train_scaled, y_train)

    return mlp, scaler



mlp_model, data_scaler = treinar_rede_neural()

st.sidebar.header("Configurações de Dados")
arquivo_carregado = st.sidebar.file_uploader(
    "Passo 1: Faça o upload do arquivo CSV", type=["csv"]
)


if arquivo_carregado is not None:
    try:
      
        df = pd.read_csv(arquivo_carregado)

       
        colunas_obrigatorias = ["servico", "total_atendimentos", "reclamacoes"]
        if not all(col in df.columns for col in colunas_obrigatorias):
            st.error(
                "❌ Erro de Formato: O CSV deve conter as colunas: 'servico', 'total_atendimentos' e 'reclamacoes'."
            )
        else:
            st.success("✅ Arquivo carregado com sucesso!")

         
            if st.button(
                "Passo 2: Ativar Função de Análise (Rede Neural)",
                type="primary",
            ):

                with st.spinner("A Rede Neural está processando os dados..."):
                   
                    df["taxa_reclamacao"] = (
                        df["reclamacoes"] / df["total_atendimentos"]
                    )

                    X_test = df[["taxa_reclamacao"]].values
                    X_test_scaled = data_scaler.transform(X_test)
                    predicoes = mlp_model.predict(X_test_scaled)

               
                    status_map = {
                        0: "Serviço Excelente",
                        1: "Serviço Bom",
                        2: "Serviço Ruim",
                        3: "Serviço Muito Ruim",
                    }
                    df["Avaliação"] = [status_map[p] for p in predicoes]

                   
                    pior_servico = df.sort_values(
                        by="taxa_reclamacao", ascending=False
                    ).iloc[0]

         
                st.divider()

           
                st.error(
                    f"### 🚨 Diagnóstico Crítico\n"
                    f"O serviço que apresenta o pior desempenho e **necessita de treinamento imediato** é: "
                    f"**{pior_servico['servico']}** (Taxa de reclamação: {pior_servico['taxa_reclamacao']*100:.1f}%)"
                )

              
                df_exibicao = df.copy()
                df_exibicao["Taxa de Reclamações"] = (
                    df_exibicao["taxa_reclamacao"] * 100
                ).map("{:.1f}%".format)
                df_exibicao = df_exibicao.rename(
                    columns={
                        "servico": "Serviço",
                        "total_atendimentos": "Total de Atendimentos",
                        "reclamacoes": "Reclamações",
                    }
                )

            
                st.markdown("### 📋 Relatório de Classificação Geral")
                st.dataframe(
                    df_exibicao[
                        [
                            "Serviço",
                            "Total de Atendimentos",
                            "Reclamações",
                            "Taxa de Reclamações",
                            "Avaliação",
                        ]
                    ],
                    use_container_width=True,
                )

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {str(e)}")
else:
    st.info(
        "💡 Aguardando dados. Por favor, utilize a barra lateral à esquerda para fazer o upload do seu relatório em formato CSV."
    )
