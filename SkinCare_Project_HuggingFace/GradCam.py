import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as keras_image
from matplotlib import cm

# Charger le modèle pré-entrainé

def generate_gradcam(img):
    model = load_model('model1.h5')
    # Prétraiter l'image
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension pour le batch
    img_array /= 255.0  # Normaliser les pixels entre 0 et 1

    # Définir la couche de convolution et les noms des couches du classificateur
    base_model = model.layers[0]
    last_conv_layer_name = base_model.layers[-1].name

    classifier_layer_names = [layer.name for layer in model.layers][1:]

    def make_gradcam_heatmap(img_array, base_model, model, last_conv_layer_name, classifier_layer_names):
        last_conv_layer = base_model.get_layer(last_conv_layer_name)
        last_conv_layer_model = tf.keras.Model(inputs=base_model.inputs, outputs=last_conv_layer.output)

        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer_name in classifier_layer_names:
            x = model.get_layer(layer_name)(x)
        classifier_model = tf.keras.Model(classifier_input, x)

        with tf.GradientTape() as tape:
            last_conv_layer_output = last_conv_layer_model(img_array)
            tape.watch(last_conv_layer_output)
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        grads = tape.gradient(top_class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(last_conv_layer_output, axis=-1)
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        return heatmap

    # Générer la heatmap
    heatmap = make_gradcam_heatmap(img_array, base_model, model, last_conv_layer_name, classifier_layer_names)

    # Convertir la heatmap pour affichage
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Convertir la heatmap en image et redimensionner
    jet_heatmap = keras_image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[2], img_array.shape[1]))  # Redimensionner à la taille originale
    jet_heatmap = keras_image.img_to_array(jet_heatmap)

    # Superposer la heatmap sur l'image originale
    superimposed_img = jet_heatmap * 0.003 + img_array[0]
    superimposed_img = keras_image.array_to_img(superimposed_img)

    return superimposed_img
