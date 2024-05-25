import vec2text
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from sklearn.cluster import KMeans
from utils import * 
from datasets import load_dataset
import time 

encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
corrector = vec2text.load_pretrained_corrector("gtr-base")

data_files = {
    "train": "news_text/data/train-00000-of-00001.parquet",
    "test": "news_text/data/test-00000-of-00001.parquet"
}
dataset = load_dataset("parquet",data_files=data_files)
shuffled_train = dataset['train'].shuffle(seed=42)
text_120k = [data["text"] for data in shuffled_train]

text_1k = text_120k[:10000]
#text_120k = ["123","456","sjakl","sdja","sdjkal"]

start = time.time()

embeddings = get_gtr_embeddings(text_1k, encoder, tokenizer)


center_embeddings = get_kmeans(embeddings,4)



print(embeddings.shape)
print(embeddings.dtype)

center_embeddings = torch.from_numpy(center_embeddings).to(torch.float32)
print(center_embeddings.shape)
#center_embeddings = [torch.from_numpy(center_embeddings)


 
answers =  vec2text.invert_embeddings(
    embeddings=center_embeddings.cuda(),
    corrector=corrector,
    num_steps=20,
)

for answer in answers:
    print(answer)
    print("\n\n")

end = time.time()
print("time",end-start)

# ['Jack Morris Morris is a PhD student at  Cornell Tech in New York City ',
# 'It was the best of times, it was the worst of times, it was the age of wisdom, it was the epoch of foolishness']



