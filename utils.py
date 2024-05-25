import vec2text
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from sklearn.cluster import KMeans



def get_gtr_embeddings(text_list,
                       encoder: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer) -> torch.Tensor:

    inputs = tokenizer(text_list,
                       return_tensors="pt",
                       max_length=128,
                       truncation=True,
                       padding="max_length",).to("cuda")

    with torch.no_grad():
        model_output = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        hidden_state = model_output.last_hidden_state
        embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])

    return embeddings



def get_kmeans(embeddings,k):
    """
    input : embedding列表,k类
    return(list) :   k个中心embedding
    """
    
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(embeddings.cpu())
    centers = kmeans.cluster_centers_
    return centers
