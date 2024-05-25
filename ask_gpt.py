from openai import OpenAI
from datasets import load_dataset
import time 
from collections import Counter

client = OpenAI()


data_files = {
    "train": "news_text/data/train-00000-of-00001.parquet",
    "test": "news_text/data/test-00000-of-00001.parquet"
}

dataset = load_dataset("parquet",data_files=data_files)
shuffled_train = dataset['train'].shuffle(seed=42)
text_120k = [data["text"] for data in shuffled_train]
label_120k = [data["label"] for data in shuffled_train]

text_12k = text_120k[:50]
label_12k  = label_120k[:50]
counts = Counter(label_12k)
print(label_12k)
print(counts)

insturction_pre = "现在我会给你你一些新闻句子，你帮我总结出这些句子里涉及最多的4种新闻类型\n,并告诉我他们在语料中分别出现的数量"
instruction_suffix = "现在开始总结这些句子里涉及最多的四种新闻类型,并告诉我他们在语料中出现的数量，用中文回答我:"
instuction = insturction_pre + "\n".join(text_12k) + instruction_suffix


start = time.time()
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": instuction}
  ]
)

end = time.time()

print(completion.choices[0].message)
print("time",end-start)
