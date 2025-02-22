import tensorflow as tf
import tensorflow_hub as hub


# # Load pretrained audio model from TensorFlow Hub (YAMNet as an example)
# yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'  # Or TRILL model
# yamnet_model = hub.load(yamnet_model_handle)

# Function to extract embeddings
def extract_embedding(wav, model):
    """ Pass audio through the pretrained model to get embeddings """
    scores, embeddings, spectrogram = model(wav)
    return list(embeddings)

# Process the dataset to get embeddings
def preprocess_dataset(dataset, model):
    embedding_list = []
    
    label_list = []
    for wav, label in dataset:
        embeddings = extract_embedding(wav, model)
        #mean_embedding = tf.reduce_mean(embeddings, axis=0)  # Aggregate embeddings
        embedding_list.extend(embeddings)
        label_list.extend([label] * len(embeddings))
    return tf.stack(embedding_list), tf.convert_to_tensor(label_list)
