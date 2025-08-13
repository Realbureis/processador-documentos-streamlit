import streamlit as st
from PIL import Image
import cv2
import numpy as np
import io
from streamlit_image_coordinates import streamlit_image_coordinates

# --- CONSTANTES DE IMPRESSÃO ---
DPI = 300
CARD_WIDTH_IN = 3.375  # Polegadas - Tamanho padrão de cartão CR80
CARD_HEIGHT_IN = 2.125  # Polegadas
TARGET_WIDTH_PX = int(CARD_WIDTH_IN * DPI)
TARGET_HEIGHT_PX = int(CARD_HEIGHT_IN * DPI)

# --- Configuração da Página ---
st.set_page_config(page_title="Processador de Documentos", page_icon="🖼️", layout="wide")


# --- FUNÇÃO DE CORREÇÃO DE PERSPECTIVA ---
def corrigir_perspectiva(imagem_original_cv, pontos_lista):
    pontos_np = np.array(pontos_lista, dtype="float32")
    (tl, tr, br, bl) = pontos_np

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pontos_np, dst)
    imagem_retificada = cv2.warpPerspective(imagem_original_cv, M, (maxWidth, maxHeight))

    return imagem_retificada


# --- Inicialização do Estado da Sessão ---
if 'pontos_frente' not in st.session_state: st.session_state.pontos_frente = []
if 'pontos_verso' not in st.session_state: st.session_state.pontos_verso = []

# --- Título e Instruções ---
st.title("✂️ Editor e Processador de Documentos")
st.info(
    "**Instruções:** Faça o upload da FRENTE e do VERSO. Depois, clique nos 4 cantos de cada documento na seguinte ordem: **1.** Sup. Esquerdo, **2.** Sup. Direito, **3.** Inf. Direito, **4.** Inf. Esquerdo.")

# --- Interface de Upload ---
col1, col2 = st.columns(2)
with col1:
    st.header("Frente do Documento")
    img_frente_upload = st.file_uploader("Selecione a imagem da FRENTE", type=["jpg", "jpeg", "png"], key="frente")
with col2:
    st.header("Verso do Documento")
    img_verso_upload = st.file_uploader("Selecione a imagem do VERSO", type=["jpg", "jpeg", "png"], key="verso")

st.markdown("---")

# --- Interface de Marcação de Pontos ---
if img_frente_upload and img_verso_upload:
    col_edit1, col_edit2 = st.columns(2)

    with col_edit1:
        st.subheader(f"FRENTE: Marque os 4 cantos ({len(st.session_state.pontos_frente)}/4)")
        img_frente_pil = Image.open(img_frente_upload)
        coords_frente = streamlit_image_coordinates(img_frente_pil, width=400, key="frente_coords")

        if coords_frente and len(st.session_state.pontos_frente) < 4:
            if coords_frente not in st.session_state.pontos_frente:
                st.session_state.pontos_frente.append(coords_frente)
                st.rerun()

        st.write(f"Pontos marcados: {st.session_state.pontos_frente}")
        if st.button("Resetar Pontos da Frente", key="reset_frente"):
            st.session_state.pontos_frente = []
            st.rerun()

    with col_edit2:
        st.subheader(f"VERSO: Marque os 4 cantos ({len(st.session_state.pontos_verso)}/4)")
        img_verso_pil = Image.open(img_verso_upload)
        coords_verso = streamlit_image_coordinates(img_verso_pil, width=400, key="verso_coords")

        if coords_verso and len(st.session_state.pontos_verso) < 4:
            if coords_verso not in st.session_state.pontos_verso:
                st.session_state.pontos_verso.append(coords_verso)
                st.rerun()

        st.write(f"Pontos marcados: {st.session_state.pontos_verso}")
        if st.button("Resetar Pontos do Verso", key="reset_verso"):
            st.session_state.pontos_verso = []
            st.rerun()

    st.markdown("---")

    if len(st.session_state.pontos_frente) == 4 and len(st.session_state.pontos_verso) == 4:
        if st.button("Processar e Juntar Imagens", use_container_width=True, type="primary"):
            with st.spinner("Processando... ✨"):
                # --- LÓGICA DE ESCALONAMENTO DOS PONTOS ---
                largura_original_frente, _ = img_frente_pil.size
                ratio_frente = largura_original_frente / 400
                pontos_frente_final = [(p['x'] * ratio_frente, p['y'] * ratio_frente) for p in
                                       st.session_state.pontos_frente]

                largura_original_verso, _ = img_verso_pil.size
                ratio_verso = largura_original_verso / 400
                pontos_verso_final = [(p['x'] * ratio_verso, p['y'] * ratio_verso) for p in
                                      st.session_state.pontos_verso]

                # --- LÓGICA DE PROCESSAMENTO OPENCV ---
                img_frente_upload.seek(0)
                img_frente_cv = cv2.imdecode(np.asarray(bytearray(img_frente_upload.read()), dtype=np.uint8), 1)
                img_verso_upload.seek(0)
                img_verso_cv = cv2.imdecode(np.asarray(bytearray(img_verso_upload.read()), dtype=np.uint8), 1)

                frente_corrigida = corrigir_perspectiva(img_frente_cv, pontos_frente_final)
                verso_corrigido = corrigir_perspectiva(img_verso_cv, pontos_verso_final)

                # --- REDIMENSIONAMENTO PARA IMPRESSÃO ---
                frente_para_impressao = cv2.resize(frente_corrigida, (TARGET_WIDTH_PX, TARGET_HEIGHT_PX))
                verso_para_impressao = cv2.resize(verso_corrigido, (TARGET_WIDTH_PX, TARGET_HEIGHT_PX))

                # --- JUNÇÃO FINAL ---
                imagem_final = cv2.hconcat([frente_para_impressao, verso_para_impressao])

                st.subheader("🎉 Resultado Final")
                st.image(cv2.cvtColor(imagem_final, cv2.COLOR_BGR2RGB),
                         caption="Frente e Verso Processados e Prontos para Impressão", use_container_width=True)

                _, buf = cv2.imencode(".jpg", imagem_final, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                bytes_imagem = io.BytesIO(buf)

                st.download_button(label="Baixar Imagem Processada (.jpg)", data=bytes_imagem,
                                   file_name="documento_processado.jpg", mime="image/jpeg", use_container_width=True)