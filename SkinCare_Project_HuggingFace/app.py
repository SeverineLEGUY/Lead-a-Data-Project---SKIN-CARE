import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import os
from GradCam import generate_gradcam

# Charger les modèles
model1 = tf.keras.models.load_model("model1.h5")
model2 = tf.keras.models.load_model("model2.h5")

classes = {
    6: 'akiec - kératoses actiniques',
    5: 'bcc - carcinome basocellulaire',
    0: 'bkl - kératoses séborrhéiques',
    2: 'df - dermatofibromes',
    3: 'mel - melanoma',
    1: 'nv - névus mélanocytaire',
    4: 'vasc - lésions vasculaires'
}

def predict_image(image):
    image_resized_model1 = image.resize((256, 256))
    img_array_model1 = np.array(image_resized_model1).reshape(-1, 256, 256, 3) / 255.0
    result_model1 = model1.predict(img_array_model1)[0][0]
    proba_malin = round(result_model1 * 100, -1)

    fig, ax = plt.subplots(figsize=(8, 1.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    gradient = np.linspace(0, 1, 100)
    ax.imshow([gradient], aspect='auto', cmap='RdYlGn_r', extent=[0, 100, 0, 1])
    ax.axvline(proba_malin, color='black', linewidth=2)
    ax.text(proba_malin + 1, 0.6, f'{proba_malin}%', fontsize=18, color='black')
    ax.text(0, 1.05, "Bénin", fontsize=12, color='black')
    ax.text(100, 1.05, "Malin", fontsize=12, color='black', ha='right')
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    result_image = Image.open(buf)

    image_resized_model2 = image.resize((128, 128))
    img_array_model2 = np.array(image_resized_model2).reshape(-1, 128, 128, 3) / 255.0
    result_model2 = model2.predict(img_array_model2)[0]
    top_3_classes_idx = np.argsort(result_model2)[::-1][:3]
    top_3_classes_prob = result_model2[top_3_classes_idx]
    top_3_text = "\n".join([
    f"{classes[i]} : {prob * 100:.1f}%" 
    for i, prob in zip(top_3_classes_idx, top_3_classes_prob)
    ])


    gradcam_image = generate_gradcam(image_resized_model1)

    return result_image, proba_malin, result_model2, top_3_text, gradcam_image

# --- Interface Streamlit ---
st.set_page_config(layout="wide")

# Titre centré
st.markdown("<h1 style='text-align: center;'>🔎 Skin Care - Analyse des grains de beauté 🔍</h1>", unsafe_allow_html=True)

# Texte centré
st.markdown("<p style='text-align: center;'>Soumettez une image et obtenez une prédiction du caractère bénin/malin, ainsi qu'une classification dermatologique.</p>", unsafe_allow_html=True)

# Ajouter un espace vide sous le titre
st.markdown("<br>", unsafe_allow_html=True)

# Ajouter une barre horizontale noire
st.markdown("<hr style='border: 1px solid black;'>", unsafe_allow_html=True)

# Créer deux colonnes
col1, col2 = st.columns([1, 2])

# Ajouter une barre verticale entre les colonnes
st.markdown(
    """
    <style>
        .css-ffhzg2 {  /* Cible la classe des colonnes Streamlit */
            border-left: 2px solid black;  /* Définir une barre verticale noire */
        }
    </style>
    """, unsafe_allow_html=True)

with col1:
    st.subheader("📥 Import manuel ou Webcam")
    
    # Ajouter le bouton pour la webcam
    camera_image = st.camera_input("Prenez une photo")

    uploaded_file = st.file_uploader("Ou choisissez une image JPG...", type="jpg")

    st.markdown("---")
    st.subheader("📁 Ou utilisez un exemple")
    example_files = ["Exemple1.jpg", "Exemple2.jpg", "Exemple3.jpg", "Exemple4.jpg", "Exemple5.jpg", "Exemple6.jpg"]
    selected_example = st.selectbox("Choisissez un exemple :", ["-- Aucun --"] + example_files)

    image = None
    if camera_image is not None:
        # Convertir l'image de la webcam en format PIL
        image = Image.open(camera_image)
        st.image(image, caption="Photo capturée via la webcam", use_container_width=True)
    elif uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image importée", use_container_width=True)
    elif selected_example != "-- Aucun --":
        image_path = os.path.join("examples", selected_example)
        image = Image.open(image_path)
        st.image(image, caption=f"Exemple : {selected_example}", use_container_width=True)

    # Ajouter une barre horizontale ici
    st.markdown("<hr>", unsafe_allow_html=True)

with col2:
    if image is not None:
        result_image, proba_malin, result_model2, top_3_text, gradcam_image = predict_image(image)

        # Section 1 : Résultat global
        st.markdown("### 🧾 Résultat global")

        # Calcul du risque
        def calculate_risk(proba_malin, result_model2):
            # Risque élevé si la jauge bénin/malin > 40% ou si la probabilité des classes akiec + bcc + mel > 40%
            risk_high = proba_malin > 50 or np.sum(result_model2[[6, 5, 3]]) > 0.30

            # Risque faible si la jauge bénin/malin < 11% et si la probabilité des classes akiec + bcc + mel < 10%
            risk_low = proba_malin < 11 and np.sum(result_model2[[6, 5, 3]]) < 0.10

            if risk_high:
                return "Risque élevé", "red", "Notre application a détecté un risque élevé. Nous vous recommandons de prendre un rendez-vous aussi vite que possible chez un professionnel de santé, médecin traitant ou dermatologue."
            elif risk_low:
                return "Risque faible", "green", "Le risque détecté est faible, mais il est toujours recommandé de surveiller vos grains de beauté régulièrement."
            else:
                return "Risque modéré", "orange", "Le risque est modéré. Il est conseillé de consulter un professionnel de santé pour un suivi, surtout si des changements sont observés."

        # Calcul du risque à partir des prédictions
        risk_text, risk_color, risk_message = calculate_risk(proba_malin, result_model2)

        # Affichage du risque
        st.markdown(f"#### <span style='font-size: 30px; color: {risk_color};'>{risk_text}</span>", unsafe_allow_html=True)

        # Affichage du message spécifique en fonction du niveau de risque
        st.markdown(f"<span style='color: {risk_color};'>{risk_message}</span>", unsafe_allow_html=True)

        st.markdown("---")

        # Section 2 : Jauge
        st.markdown("### 🩺 Jauge de probabilité bénin / malin")
        st.image(result_image, use_container_width=False)
        st.markdown("---")

        # Section 3 : Top 3
        st.markdown("### 🔍 Top 3 des classes prédites")
        st.markdown(f"<div style='font-size:20px; white-space: pre-wrap;'>{top_3_text}</div>", unsafe_allow_html=True)
        
        # Affichage horizontal des images des classes avec proba > 10%
        st.markdown("#### 📸 Exemples des classes détectées (>10%)")
        
        # Trouver les classes avec proba > 10%
        high_proba_classes = [(idx, proba) for idx, proba in enumerate(result_model2) if proba > 0.10]
        
        # Créer autant de colonnes que de classes à afficher
        cols = st.columns(len(high_proba_classes))
        
        # Afficher chaque image dans sa colonne, avec largeur fixe (~1/4 de col2 ≈ 200px)
        for col, (idx, proba) in zip(cols, high_proba_classes):
            class_code = classes[idx].split(' - ')[0]
            class_label = classes[idx]
            image_path = os.path.join("classes", f"{class_code}.jpg")
            if os.path.exists(image_path):
                with col:
                    st.image(image_path, caption=class_label, width=200)

        st.markdown("---")

        # Section 4 : Grad-CAM centrée
        st.markdown("### 🧠 Visualisation Grad-CAM / Zones qui ont impacté l'analyse")
        centered_col = st.columns([1, 2, 1])[1]
        with centered_col:
            st.image(gradcam_image, width=300)
        st.markdown("---")

        # Section 5 : Conseil Skincare
        st.markdown("### 💡 Conseils Skincare")
        st.write(
            "💡 Ce modèle vous donne un aperçu du risque associé à l’image et propose une classification dermatologique automatisée.<br> "
            "👨‍⚕️ Cette application ne remplace en aucun cas l'avis d'un professionnel de santé.<br>"
            "👩‍⚕️ Consultez un dermatologue en cas de doute ou de changement rapide.<br>"
            "🔆 Appliquez une crème solaire à large spectre tous les jours, même en hiver.<br>"
            "📅 Surveillez vos grains de beauté tous les 3 mois (ABCD : Asymétrie, Bords, Couleur, Diamètre).<br>"
            "🧴 Choisissez des produits de soin adaptés à votre type de peau et à vos besoins spécifiques (peau sèche, grasse, sensible, etc.).<br>"
            "💧 Hydratez votre peau régulièrement avec des crèmes et sérums adaptés pour maintenir une barrière cutanée saine.<br>"
            "🚶‍♂️ Évitez une exposition excessive au soleil, surtout entre 12h et 16h, lorsque les rayons UV sont les plus forts.<br>"
            "🧑‍⚕️ Si vous remarquez un changement dans un grain de beauté (forme, couleur, taille), consultez immédiatement un professionnel de santé.<br>"
            "🍏 Adoptez une alimentation équilibrée riche en antioxydants (fruits, légumes, acides gras essentiels) pour soutenir la santé de votre peau.", unsafe_allow_html=True
        )